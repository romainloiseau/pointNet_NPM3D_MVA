import os
import numpy as np
import matplotlib.pyplot as plt
import time
from tqdm import tqdm_notebook
from keras.utils import to_categorical
from sklearn.neighbors import KDTree
from plyfile import PlyData, PlyElement
from ply import write_ply
import joblib as joblib
import keras

def get_cov(points):
    points -= np.mean(points, axis = 0)
    return points.transpose().dot(points) / points.shape[0]

def refine_normals(normals):
    return np.abs(normals)
    
def compute_normals(tree, radius = .75, path = None, save = True):
    cloud = np.array(tree.data)
    npoints = len(cloud)
    if(not path is None):
        print("MODIFIYING PATHS", path)
        normal_path = path[:-4] + "_normals.npy"
        eigen_path = path[:-4] + "_eigens.npy"
        
    if((path is None) or (not os.path.isfile(normal_path))):
        print("COMPUTING NORMALS")
        max_points_in_cloud = 2000000
        if(npoints < max_points_in_cloud):
            try:
                neighborhoods = tree.query_radius(cloud, r = radius)
                print("pass 0")
                cov = np.array([get_cov(cloud[neighborhood]) for neighborhood in neighborhoods])
                print("pass 1")
            except:
                print("EXCEPT")
                cov = np.array([get_cov(cloud[tree.query_radius([point], r = radius)[0]]) for point in tqdm_notebook(cloud)])
        else:
            n_splits = int(10 * npoints / max_points_in_cloud) + 1
            print("SPLITTING {}".format(n_splits))
            cov = np.concatenate([np.array([get_cov(cloud[neighborhood]) for neighborhood in tree.query_radius(cloud[int(i * npoints / n_splits): int((i + 1) * npoints / n_splits)], r = radius)]) for i in tqdm_notebook(range(n_splits))], axis = 0)
        
        print(len(cov), npoints)
        eigen_values, eigen_vectors = np.linalg.eigh(cov)
        mini_eigen_values = np.argmin(eigen_values, axis = -1)

        normals = np.array([vectors[:, mini] for vectors, mini in zip(eigen_vectors, mini_eigen_values)])
        normals = (normals.transpose() / np.sum(normals**2, axis = -1)**.5).transpose()

        sorted_eigenvalues = np.sort(eigen_values)
        
        if(save and not path is None):
            print("SAVING NORMALS")
            np.save(normal_path, normals)
            np.save(eigen_path, sorted_eigenvalues)
    
    else:
        print("LOADING NORMALS", normal_path)
        normals = np.load(normal_path)
        sorted_eigenvalues = np.load(eigen_path)
        
    sorted_eigenvalues = np.maximum(sorted_eigenvalues, 10**(-10))
    return normals, sorted_eigenvalues

def random_rotate_z(cloud, normal = None):
    rotation = 2 * np.pi * np.random.random()
    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                                [np.sin(rotation), np.cos(rotation), 0],
                                [0, 0, 1.]])
    
    if(normal is None):return cloud.dot(rotation_matrix), None
    else:return cloud.dot(rotation_matrix), normal.dot(rotation_matrix)

def preprocess_cloud(cloud):#, normal = None, eigen = None):
    #cloud -= np.mean(cloud, axis = 0)
    #cloud, normal = random_rotate_z(cloud, normal)
    #if(normal is None):return cloud
    #else:
        #normal = refine_normals(normal)
        #if(eigen is None):return np.concatenate([cloud, normal], axis = -1)
        #else:return np.concatenate([cloud, normal, eigen], axis = -1)
    keys = [k for k in cloud.keys()]
    xyz = cloud["cloud"] - np.mean(cloud["cloud"], axis = 0)
    if("normal" in keys):
        xyz, normal = random_rotate_z(xyz, cloud["normal"])
        normal = refine_normals(normal)
        final_cloud = [xyz, normal]
        if("eigen" in keys):
            final_cloud += [cloud["eigen"]]
    else:
        xyz, _ = random_rotate_z(xyz, None)
        final_cloud = [xyz]
    if("reflectance" in keys):
        final_cloud += [np.expand_dims(cloud["reflectance"], -1)]
    return np.concatenate(final_cloud, axis = -1)
    
def from_categorical(label):
    return label.dot(1 + np.arange(label.shape[-1]))

class NPM3DGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, n_points = 4096, batch_size = 8, input_dir = "../Benchmark_MVA/training", train = True, evaluation = False, paths_to_keep = None, use_normals = True, compute_normals = True, normal_radius = .75, sample_uniformly_from_classes = False, use_reflectance = False, max_biggest_eigen = .4):
        'Initialization'
        
        self.class_dict = {0 : "unclassified", 1 : "ground", 2 : "buildings", 3 : "poles", 4 : "pedestrians", 5 : "cars", 6 : "vegetation"}
        self.n_classes = len(self.class_dict) - 1
        
        self.input_dir = input_dir
        self.paths_to_keep = paths_to_keep
        
        self.batch_size = batch_size
        self.n_points = n_points
        self.train = train
        self.evaluation = evaluation
        self.sample_uniformly_from_classes = sample_uniformly_from_classes
        self.normal_radius = normal_radius
        self.use_normals = use_normals
        self.compute_normals = use_normals and compute_normals
        self.max_biggest_eigen = max_biggest_eigen
        self.use_CCcomputed_normals = use_normals and not compute_normals
        self.use_reflectance = use_reflectance
                 
        self.n_channels = 3
        if(self.use_reflectance):self.n_channels += 1        
        if(self.use_normals):self.n_channels += 3
        if(self.compute_normals):self.n_channels += 3
        
        self.prepare_NPM3D()
    
    def __str__(self):
        output = ""
        output += "NPM3DGenerator config\n"
        output += "n_classes         : {}\n".format(self.n_classes)
        output += "batch_size        : {}\n".format(self.batch_size)
        output += "n_points          : {}\n".format(self.n_points)
        output += "n_channels        : {}\n".format(self.n_channels)
        output += "train             : {}\n".format(self.train)
        if(self.use_normals):
            output += "use_normals       : {}\n".format(self.use_normals)
            output += "normal_radius     : {}\n".format(self.normal_radius)
            output += "compute_normals   : {}\n".format(self.compute_normals)
        if(self.train):
            output += "class_weight      : {}".format(np.array2string(self.class_weight, formatter={'float_kind':lambda x: "%.2f" % x}))
        return output
        
    def get_label(self, label):
        return to_categorical(label, num_classes = self.n_classes + 1)[:, 1:] 
        
    def load_point_cloud(self, input_dir):
        data = PlyData.read(input_dir)
        columns = ["x", "y", "z"]
        if(self.use_CCcomputed_normals):columns += ["nx", "ny", "nz"]
        if(self.use_reflectance):columns += ["reflectance"]
        columns += ["class"]
        data = np.array([data.elements[0].data[i] for i in columns[:len(data.elements[0].properties)]]).transpose()
        """
        for i in range(3):
            plt.hist(data[:, i])
            plt.show()
        """
        
        tree_path = input_dir[:-4] + "_tree.joblib"
        print(tree_path)
        if(os.path.isfile(tree_path)):
            print("LOADING TREE")
            tree = joblib.load(tree_path)
            if(np.mean(np.array(tree.data) == data[:, :3]) != 1):
                raise ValueError("THE LOADED TREE ISN'T THE ONE ASSOCIATED TO THIS DATA")
        else:
            print("COMPUTING TREE")
            tree = KDTree(data[:, :3], metric = "euclidean")
            joblib.dump(tree, tree_path)
        print("DONE")
        normal = None
        eigen = None
        reflectance = None
        if(self.use_normals):
            if(self.use_CCcomputed_normals):
                normal = data[:, 3:6]
            else:
                normal, eigen = compute_normals(tree, self.normal_radius, input_dir)
        if(self.use_reflectance):
            plt.hist(data[:, -2])
            plt.show()
            #reflectance = data[:, -2] / 255.
            reflectance = (data[:, -2] - np.min(data[:, -2])) / (np.max(data[:, -2]) - np.min(data[:, -2]))
        label = self.get_label(data[:, -1]) if self.train else None
        return tree, normal, eigen, reflectance, label
    
    def compute_class_weight(self):
        """
        sum_labels = np.mean(np.concatenate(self.labels), axis = 0)
        sum_labels = np.clip(sum_labels, .0001, 1.)
        self.class_weight = 1. / sum_labels
        """
        #self.class_weight = np.array([ 2.48904064,  3.53694259, 62.30171678, 23.44471258,  8.60931716, 7.10979527])
        self.class_weight = np.array([ 2.77362278,  3.99426094, 37.20384495, 15.52374327,  7.15863926, 6.32456057])
    
    def prepare_NPM3D(self):
        self.paths = os.listdir(self.input_dir)
        self.paths = [path for path in self.paths if path.split(".")[-1] == "ply"]
        if(self.use_CCcomputed_normals):self.paths = [path for path in self.paths if path.split(".")[0][-8:] == "_normals"]
        else:self.paths = [path for path in self.paths if path.split(".")[0][-8:] != "_normals"]
        if(not self.paths_to_keep is None):self.paths = [path for i, path in enumerate(self.paths) if i in self.paths_to_keep]
          
        self.trees = []
        if(self.use_normals):self.normals = []
        if(self.compute_normals):self.eigens = []
        if(self.use_reflectance):self.reflectances = []
        if(self.train):self.labels = []
        
        for path in tqdm_notebook(self.paths):
            tree, normal, eigen, reflectance, label = self.load_point_cloud(os.path.join(self.input_dir, path))
            self.trees.append(tree)
            if(self.use_normals):
                 self.normals.append(normal)
                 if(self.compute_normals):
                    eigen = np.concatenate([(eigen.transpose() / eigen[:, -1]).transpose()[:, :-1], np.array([eigen[:, -1] / self.max_biggest_eigen]).transpose()], axis = -1)
                    self.eigens.append(eigen)
            if(self.use_reflectance):self.reflectances.append(reflectance)
            if(self.train):self.labels.append(label)
        
        self.n_points_clouds = np.array([len(tree.data) for tree in self.trees])
        
        self.n_points_total = np.sum(self.n_points_clouds)
        self.n_clouds = len(self.trees)
        
        if(self.train):
            self.compute_class_weight()
            if(self.sample_uniformly_from_classes):
                self.labels_as_int = [label.dot(np.arange(self.n_classes)) for label in self.labels]
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(self.n_points_total / (self.batch_size * self.n_points)))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = np.random.choice(self.n_clouds, self.batch_size, p = self.n_points_clouds / self.n_points_total)

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y
    
    def choose_center_point(self, index = 0, chosen_label = None):
        if(self.train and self.sample_uniformly_from_classes):
            if chosen_label is None:chosen_label = np.random.randint(self.n_classes)
            chosen_indexes = np.where(self.labels_as_int[index] == chosen_label)[0]
            if(len(chosen_indexes) > 0):
                center_points_index = chosen_indexes[np.random.randint(len(chosen_indexes))]
            else:
                if self.evaluation:center_points_index = np.random.randint(self.n_points_clouds[index])
                else:return None
        else:
            center_points_index = np.random.randint(self.n_points_clouds[index])
        return center_points_index
    
    def sample_point_cloud(self, index = 0, center_points_index = None):
        if center_points_index is None:center_points_index = self.choose_center_point(index)
        if center_points_index is None:
            index = min(2, self.n_clouds - 1)
            center_points_index = self.choose_center_point(index, 3)
                
        cloud = np.array(self.trees[index].data)
        dist, ind = self.trees[index].query([cloud[center_points_index]], k = self.n_points)
        dist, ind = dist[0], ind[0]
        
        to_preprocess = {"cloud": cloud[ind]}
        if(self.use_normals):
            to_preprocess["normal"] = self.normals[index][ind]
            if(self.compute_normals):
                to_preprocess["eigen"] = self.eigens[index][ind]
        if(self.use_reflectance):
            to_preprocess["reflectance"] = self.reflectances[index][ind]
            
        cloud = preprocess_cloud(to_preprocess)
       
        """
        if(self.use_normals):
            if(self.compute_normals):cloud = preprocess_cloud(cloud[ind], self.normals[index][ind], self.eigens[index][ind])
            else:cloud = preprocess_cloud(cloud[ind], self.normals[index][ind])
        else:cloud = preprocess_cloud(cloud[ind])
        """
        
        if(self.train):
            label = self.labels[index][ind]
            return cloud, label
        else:
            return cloud, ind, dist
            
    def __data_generation(self, indexes):
        'Generates data containing batch_size samples'
        # Initialization
        # Generate data
        
        clouds = np.empty((self.batch_size, self.n_points, self.n_channels))
        labels = np.empty((self.batch_size, self.n_points, self.n_classes))
        
        for i, index in enumerate(indexes):
            cloud, label = self.sample_point_cloud(index)
            clouds[i] = cloud.copy()
            labels[i] = label.copy()

        return clouds, labels
    
    def predict_point_cloud(self, model, index = 0, epsilon_weights = .5, output_path = None):
        all_indexes = np.arange(self.n_points_clouds[index])
        predictions = np.zeros((self.n_points_clouds[index], self.n_classes))
        weights = np.zeros(self.n_points_clouds[index])
        
        pbar, pbar_value, pbar_update = tqdm_notebook(total=100), 0, 0
        while np.min(weights) < epsilon_weights:
            center_points_indexes = all_indexes[weights < epsilon_weights]
            cloud, ind, dist = self.sample_point_cloud(index = index, center_points_index = center_points_indexes[np.random.randint(len(center_points_indexes))])
            weight = 1. / np.clip(dist, .1 * np.max(dist), 10.)
            
            prediction = model.predict(np.expand_dims(cloud, axis = 0))[0]
            predictions[ind] += (prediction.transpose() * weight).transpose() 
            weights[ind] += weight
            
            int_pbar_value = int(pbar_value)
            pbar_value = 100 * np.mean(weights > epsilon_weights)
            pbar_update = int(pbar_value) - int_pbar_value
            for i in range(pbar_update):
                pbar.update()
        pbar.close()
        
        predictions = (predictions.transpose() / weights).transpose()
        predictions_int = (np.argmax(predictions, axis = -1) + 1).astype(int)
        
        if(output_path is None):output_path = self.paths[index].split(".")[0] + "_prediction.ply"
        
        if(self.use_normals):
            write_ply(output_path,
                      [np.array(self.trees[index].data), self.normals[index], predictions_int],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz', 'class'])
        else:
            write_ply(output_path,
                      [np.array(self.trees[index].data), predictions_int],
                      ['x', 'y', 'z', 'class'])
        
        return predictions, predictions_int
    
class NPM3DGenerator_full(NPM3DGenerator):
    'Generates data for Keras'
    def __init__(self, n_points = 4096, batch_size = 8, input_dir = "../Benchmark/training_10_classes_subsampled_2", train = True, evaluation = False, paths_to_keep = None, use_normals = True, compute_normals = True, normal_radius = .75, sample_uniformly_from_classes = False, dataset_predicted = "full", use_reflectance = True, max_biggest_eigen = .4):
        
        'Initialization'
        
        self.dataset_predicted = dataset_predicted
        assert self.dataset_predicted in ["full", "mva"]
        if(self.dataset_predicted == "full"):
            self.class_dict = {0 : "unclassified",
                               1 : "ground",
                               2 : "building",
                               3 : "pole - road sign - traffic light",
                               4 : "bollard - small pole",
                               5 : "trash can",
                               6 : "barrier",
                               7 : "pedestrian",
                               8 : "car",
                               9 : "natural - vegetation"}
        elif(self.dataset_predicted == "mva"):
            self.class_dict = {0 : "unclassified",
                               1 : "ground",
                               2 : "buildings",
                               3 : "poles",
                               4 : "pedestrians",
                               5 : "cars",
                               6 : "vegetation"}
            
        self.n_classes = len(self.class_dict) - 1
        
        self.input_dir = input_dir
        self.paths_to_keep = paths_to_keep
        
        self.batch_size = batch_size
        self.n_points = n_points
        self.train = train
        self.evaluation = evaluation
        self.sample_uniformly_from_classes = sample_uniformly_from_classes
        self.normal_radius = normal_radius
        self.use_normals = use_normals
        self.max_biggest_eigen = max_biggest_eigen
        self.compute_normals = use_normals and compute_normals
        self.use_CCcomputed_normals = use_normals and not compute_normals
        self.use_reflectance = use_reflectance
        self.n_channels = 3
        if(self.use_reflectance):self.n_channels += 1
        if(self.use_normals):self.n_channels += 3
        if(self.compute_normals):self.n_channels += 3
        
        self.prepare_NPM3D()
    
    def from_coarse_to_global_labels_int(self, x):
        if x in [0, 1, 2]:
            return x
        elif x in [3, 4, 5, 6]:
            return 3
        elif x in [7, 8, 9]:
            return x - 3
        else:
            return 0
        
    def from_coarse_to_global_labels(self, label):
        for i in range(len(label)):
            label[i] = self.from_coarse_to_global_labels_int(label[i])
        return label
        
    def get_label(self, label):
        if(self.dataset_predicted == "full"):
            return to_categorical(label, num_classes = self.n_classes + 1)[:, 1:]
        else:
            return to_categorical(self.from_coarse_to_global_labels(label), num_classes = self.n_classes + 1)[:, 1:]
        
    def compute_class_weight(self):
        """
        sum_labels = np.mean(np.concatenate(self.labels), axis = 0)
        sum_labels = np.clip(sum_labels, .0001, 1.)
        self.class_weight = 1. / sum_labels
        """
        self.class_weight = np.array([2.282537, 5.243769, 33.179123, 169.81613, 32.47143, 15.554446, 24.446503, 13.716806, 7.919863])
