import os
import numpy as np
from tqdm import tqdm_notebook
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from keras.utils import to_categorical

def get_classes(input_dir):
    classes = []
    for class_ in os.listdir(input_dir):
        subpath = os.path.join(input_dir, class_)
        if(not os.path.isfile(subpath) and class_[0] != "_"):
            classes.append(class_)
    return classes

def get_file_paths(input_dir, classes, split):
    paths, labels = [], []
    for key, class_ in classes.items():
        subpath = os.path.join(os.path.join(input_dir, class_), split)
        for item in os.listdir(subpath):
            file_path = os.path.join(subpath, item)
            if(os.path.isfile(file_path) and item[0] != "."):
                paths.append(file_path)
                labels.append(key)
    return paths, labels

def load_point_cloud(input_dir):
    lines = [l[:-1] for l in open(input_dir,"r").readlines()]
    
    if(lines[0] == "OFF"):
        n_vertices, n_triangles, _ = np.array(lines[1].split(" ")).astype(int)
        delta = 2
    elif(lines[0][:3] == "OFF" and len(lines[0]) > 3):
        n_vertices, n_triangles, _ = np.array(lines[0][3:].split(" ")).astype(int)
        delta = 1
    else:raise ValueError(lines[0] + " " + lines[1] + "   INVALID FIRST LINES")
        
    vertices = lines[delta: n_vertices + delta]
    vertices = np.array([v.split(" ") for v in vertices])[:, :3].astype(float)
    triangles = lines[n_vertices + delta:]
    triangles = np.array([t.split(" ") for t in triangles]).astype(int)
    
    return vertices, triangles

def preprocess_vertices(vertices):
    vertices -= np.mean(vertices, axis = 0)
    vertices /= np.max(np.abs(vertices))
    return vertices

def vizualize_point_cloud(vertices, title = ""):
    fig = plt.figure()
    ax0 = fig.add_subplot(121)
    [ax0.hist(vertices[:, i], alpha = .5) for i in range(3)]
    ax0.set_xlim(-1, 1)
    
    ax = fig.add_subplot(122, projection='3d')
    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2])
    if(title != ""):plt.title(str(title))
    plt.show()
    
def choose_triangles(vertices, triangles, n = 2048):
    p1, p2, p3 = np.array([vertices[triangles[:, i]] for i in [1, 2, 3]])

    v12, v23, v31 = p2 - p1, p3 - p2, p1 - p3

    l12 = np.sqrt(np.sum(v12**2, axis = -1))
    l23 = np.sqrt(np.sum(v23**2, axis = -1))
    l31 = np.sqrt(np.sum(v31**2, axis = -1))

    p = (l12 + l23 + l31) / 2
    a2 = np.abs(p * (p - l12) * (p - l23) * (p - l31))
    a = np.sqrt(a2)
    a /= np.sum(a)

    choice = np.random.choice(len(a), n, p = a)
    
    return choice

def pick_random_points(vertices, triangles, choice):
    final_triangles = triangles[choice]
    p1, p2, p3 = np.array([vertices[final_triangles[:, i]] for i in [1, 2, 3]])
    
    r = np.random.random((len(choice), 2))
    sqrtr1 = np.sqrt(r[:, 0])
    
    final_vertices = ((1 - sqrtr1) * p1.transpose() + sqrtr1 * (1 - r[:, 1]) * p2.transpose() + r[:, 1] * sqrtr1 * p3.transpose()).transpose()
    
    return final_vertices

def load_and_preprocess_point_cloud(path, n = 2048):
    vertices, triangles = load_point_cloud(path)
    vertices = preprocess_vertices(vertices)
    
    assert np.mean(triangles[:, 0]) == 3.
    
    choice = choose_triangles(vertices, triangles, n)
    
    final_vertices = pick_random_points(vertices, triangles, choice)
    
    return final_vertices

def prepare_for_learning(input_dir, n = 2048, verbose = 1):
    classes = get_classes(input_dir)
    class_dict = dict(enumerate(classes))
    if(verbose):print(class_dict)
    n_classes = len(class_dict)   
    
    train_paths, train_labels = get_file_paths(input_dir, class_dict, "train")
    test_paths, test_labels = get_file_paths(input_dir, class_dict, "test")
    
    train_data, test_data = [], []
    
    for path in tqdm_notebook(train_paths):
        try:
            train_data.append(load_and_preprocess_point_cloud(path, n))
        except:
            print("ERROR", path)
            pass
       
    for path in tqdm_notebook(test_paths):
        try:
            test_data.append(load_and_preprocess_point_cloud(path, n))
        except:
            print("ERROR", path)
            pass
       
    train_data = np.array(train_data)
    test_data = np.array(test_data)
    
    if(verbose):
        for i in range(3):
            ind = np.random.randint(len(train_data) - 1)
            vizualize_point_cloud(train_data[ind], title = class_dict[train_labels[ind]])
            
    train_labels = to_categorical(train_labels, n_classes)
    test_labels = to_categorical(test_labels, n_classes)
            
    return train_data, train_labels, test_data, test_labels