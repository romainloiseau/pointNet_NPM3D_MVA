import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm_notebook
from keras.utils import to_categorical
from sklearn.neighbors import KDTree
from plyfile import PlyData, PlyElement
from ply import write_ply
import keras

def get_cov(points):
    points -= np.mean(points, axis = 0)
    return points.transpose().dot(points) / points.shape[0]

def refine_normals(normals):
    return np.abs(normals)
    
def compute_normals(cloud, tree, radius = .75, path = None, save = True):
    
    if(not path is None):
        print("not path is None", path)
        normal_path = path[:-4] + "_normals.npy"
        print(normal_path)
        eigen_path = path[:-4] + "_eigens.npy"
        
    if((path is None) or (not os.path.isfile(normal_path))):
        print("path is None or not os.path.isfile(normal_path))")
        max_points_in_cloud = 2500000
        if(len(cloud) < max_points_in_cloud):
            try:
                neighborhoods = tree.query_radius(cloud, r = radius)
                cov = np.array([get_cov(cloud[neighborhood]) for neighborhood in neighborhoods])
            except:
                print("EXCEPT")
                cov = np.array([get_cov(cloud[tree.query_radius([point], r = radius)[0]]) for point in tqdm_notebook(cloud)])
        else:
            n_splits = 4 * (int(len(cloud) / max_points_in_cloud) + 1)
            print("SPLITTING {}".format(n_splits))
            cov = np.concatenate([np.array([get_cov(cloud[neighborhood]) for neighborhood in tree.query_radius(cloud[int(i * len(cloud) / n_splits): int((i + 1) * len(cloud) / n_splits)], r = radius)]) for i in tqdm_notebook(range(n_splits))], axis = 0)
        
        print(len(cov), len(cloud))
        eigen_values, eigen_vectors = np.linalg.eigh(cov)
        mini_eigen_values = np.argmin(eigen_values, axis = -1)

        normals = np.array([vectors[:, mini] for vectors, mini in zip(eigen_vectors, mini_eigen_values)])
        normals = refine_normals((normals.transpose() / np.sum(normals**2, axis = -1)**.5).transpose())

        sorted_eigenvalues = np.sort(eigen_values)
        sorted_eigenvalues = np.maximum(sorted_eigenvalues, 10**(-6))
        sorted_eigenvalues = (sorted_eigenvalues.transpose() / sorted_eigenvalues[:, -1]).transpose()[:, :-1]
        
        if(save and not path is None):
            print("save and not path is None")
            np.save(normal_path, normals)
            np.save(eigen_path, sorted_eigenvalues)
    
    else:
        print("else", normal_path)
        normals = np.load(normal_path)
        sorted_eigenvalues = np.load(eigen_path)
        
    return normals, sorted_eigenvalues

def random_rotate_z(cloud, normal = None):
    rotation = 2 * np.pi * np.random.random()
    rotation_matrix = np.array([[np.cos(rotation), -np.sin(rotation), 0],
                                [np.sin(rotation), np.cos(rotation), 0],
                                [0, 0, 1.]])
    
    if(normal is None):return cloud.dot(rotation_matrix), None
    else:return cloud.dot(rotation_matrix), normal.dot(rotation_matrix)

def preprocess_cloud(cloud, normal = None, eigen = None):
    cloud -= np.mean(cloud, axis = 0)
    cloud, normal = random_rotate_z(cloud, normal)
    if(normal is None):return cloud
    else:
        if(eigen is None):return np.concatenate([cloud, normal], axis = -1)
        else:return np.concatenate([cloud, normal, eigen], axis = -1)

def from_categorical(label):
    return label.dot(1 + np.arange(label.shape[-1]))

class NPM3DGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, n_points = 4096, batch_size = 8, input_dir = "../Benchmark_MVA/training", train = True, paths_to_keep = None, use_normals = True, compute_normals = True, normal_radius = .75):
        'Initialization'
        
        self.class_dict = {0 : "unclassified", 1 : "ground", 2 : "buildings", 3 : "poles", 4 : "pedestrians", 5 : "cars", 6 : "vegetation"}
        self.n_classes = len(self.class_dict) - 1
        
        self.input_dir = input_dir
        self.paths_to_keep = paths_to_keep
        
        self.batch_size = batch_size
        self.n_points = n_points
        self.train = train
        self.normal_radius = normal_radius
        self.use_normals = use_normals
        self.compute_normals = use_normals and compute_normals
        self.use_CCcomputed_normals = use_normals and not compute_normals
        
        self.n_channels = 3
        if(self.use_normals):self.n_channels += 3
        if(self.compute_normals):self.n_channels += 2
        
        self.prepare_NPM3D()
        
    def get_label(self, label):
        return to_categorical(label, num_classes = self.n_classes + 1)[:, 1:] 
        
    def load_point_cloud(self, input_dir):
        data = PlyData.read(input_dir)
        columns = ["x", "y", "z"]
        if(self.use_CCcomputed_normals):columns += ["nx", "ny", "nz", "scalar_class"]
        else:columns += ["class"]
        data = np.array([data.elements[0].data[i] for i in columns[:len(data.elements[0].properties)]]).transpose()
        cloud = data[:, :3]
        tree = KDTree(cloud)
        normal = None
        eigen = None
        if(self.use_normals):
            if(self.use_CCcomputed_normals):
                normal = data[:, 3:6]
            else:
                normal, eigen = compute_normals(cloud, tree, self.normal_radius, input_dir)
        label = self.get_label(data[:, -1]) if self.train else None
        return cloud, tree, normal, eigen, label
    
    def compute_class_weight(self):
        sum_labels = np.mean(np.concatenate(self.labels), axis = 0)
        sum_labels = np.clip(sum_labels, .0001, 1.)
        self.class_weight = 1. / sum_labels
    
    def prepare_NPM3D(self):
        self.paths = os.listdir(self.input_dir)
        self.paths = [path for path in self.paths if path.split(".")[-1] == "ply"]
        if(self.use_CCcomputed_normals):self.paths = [path for path in self.paths if path.split(".")[0][-8:] == "_normals"]
        else:self.paths = [path for path in self.paths if path.split(".")[0][-8:] != "_normals"]
        if(not self.paths_to_keep is None):self.paths = [path for i, path in enumerate(self.paths) if i in self.paths_to_keep]
            
        self.clouds = []
        self.trees = []
        if(self.use_normals):self.normals = []
        if(self.compute_normals):self.eigens = []
        if(self.train):self.labels = []
        
        for path in tqdm_notebook(self.paths):
            cloud, tree, normal, eigen, label = self.load_point_cloud(os.path.join(self.input_dir, path))
            self.clouds.append(cloud)
            self.trees.append(tree)
            if(self.use_normals):self.normals.append(normal)
            if(self.compute_normals):self.eigens.append(eigen)
            if(self.train):self.labels.append(label)
        
        self.n_points_clouds = np.array([len(c) for c in self.clouds])
        
        self.n_points_total = np.sum(self.n_points_clouds)
        self.n_clouds = len(self.clouds)
        
        if(self.train):self.compute_class_weight()
    
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
    
    def sample_point_cloud(self, index = 0, center_point = None):
        
        if(center_point is None):center_point = np.random.randint(self.n_points_clouds[index])
        dist, ind = self.trees[index].query([self.clouds[index][center_point]], k = self.n_points)
        dist, ind = dist[0], ind[0]
        
        if(self.use_normals):
            if(self.compute_normals):cloud = preprocess_cloud(self.clouds[index][ind], self.normals[index][ind], self.eigens[index][ind])
            else:cloud = preprocess_cloud(self.clouds[index][ind], self.normals[index][ind])
        else:cloud = preprocess_cloud(self.clouds[index][ind])
        
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
    
    def predict_point_cloud(self, model, index = 0, epsilon_weights = .1, output_path = None):
        all_indexes = np.arange(self.n_points_clouds[index])
        predictions = np.zeros((self.n_points_clouds[index], self.n_classes))
        weights = np.zeros(self.n_points_clouds[index])
        
        pbar, pbar_value, pbar_update = tqdm_notebook(total=100), 0, 0
        while np.min(weights) < epsilon_weights:
            center_points = all_indexes[weights < epsilon_weights]
            cloud, ind, dist = self.sample_point_cloud(original_point_cloud = index, center_point = center_points[np.random.randint(len(center_points))])
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
        
        predictions = (predictions.transpose() * weights).transpose()
        
        if(output_path is None):output_path = self.paths[index].split(".")[0] + "_prediction.ply"
        
        if(self.use_normals):
            write_ply(output_path,
                      [self.clouds[index], self.normals[index], np.argmax(predictions, axis = -1).astype(int)],
                      ['x', 'y', 'z', 'nx', 'ny', 'nz', 'class'])
        else:
            write_ply(output_path,
                      [self.clouds[index], np.argmax(predictions, axis = -1)],
                      ['x', 'y', 'z', 'class'])
        
        return predictions
    
class NPM3DGenerator_full(NPM3DGenerator):
    'Generates data for Keras'
    def __init__(self, n_points = 4096, batch_size = 8, input_dir = "../Benchmark/training_10_classes", train = True, paths_to_keep = None, use_normals = False, compute_normals = True, normal_radius = .25):
        'Initialization'
        
        self.class_dict = {0 : "unclassified", 1 : "ground", 2 : "buildings", 3 : "poles", 4 : "pedestrians", 5 : "cars", 6 : "vegetation"}
        self.n_classes = len(self.class_dict) - 1
        
        self.input_dir = input_dir
        self.paths_to_keep = paths_to_keep
        
        self.batch_size = batch_size
        self.n_points = n_points
        self.train = train
        self.normal_radius = normal_radius
        self.use_normals = use_normals
        self.compute_normals = use_normals and compute_normals
        self.use_CCcomputed_normals = use_normals and not compute_normals
        
        self.n_channels = 3
        if(self.use_normals):self.n_channels += 3
        if(self.compute_normals):self.n_channels += 2
        
        self.prepare_NPM3D()
        
    def get_label(self, label):
        return to_categorical(label, num_classes = self.n_classes + 1)[:, 1:] 
