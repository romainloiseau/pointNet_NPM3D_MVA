import os
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import to_categorical
import keras
from utils_ModelNet import get_classes, get_file_paths
import tensorflow as tf

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, batch_size = 32, shuffle=True):
        'Initialization'
        
        self.prepare_ModelNet()
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()
        
    def show_samples(self, n_samples = 5):
        
        for i in range(n_samples):
            ind = np.random.randint(0, len(self.test_labels) - 1)
            iclass = self.test_labels[ind]
            plt.figure(figsize = (10, 3))
            plt.subplot(131)
            plt.imshow(self.test_rgb[ind], cmap = "gray")
            plt.title(self.class_dict[iclass])
            plt.axis("off")
            plt.subplot(132)
            plt.imshow(self.test_colored[ind])
            plt.title(self.colorspace)
            plt.axis("off")
            plt.subplot(133)
            plt.imshow(self.test_gray[ind, :, :, 0], cmap = "gray")
            plt.title("GRAY")
            plt.axis("off")
            plt.show()
    
    def prepare_ModelNet(self):
        global_path = "ModelNet10/"
        
        classes = get_classes(global_path)
        self.class_dict = dict(enumerate(classes))
        print(self.class_dict)
        self.n_classes = len(self.class_dict)
        
        test_paths, test_labels = get_file_paths(global_path, self.class_dict, "test")
        train_paths, train_labels = get_file_paths(global_path, self.class_dict, "train")
                          
        print("ALL PATHS LOADED")
        
        self.list_IDs = list(set(np.arange(len(self.labels))) - set(test_ids))
        
        print("SPLIT DONE", len(test_ids))
        
        self.test_rgb = self.preprocess(np.array([self.load(self.list_paths[test_id]) for test_id in test_ids]))
        self.test_labels = np.array([self.labels[test_id] for test_id in test_ids])
        self.test_gray = self.fromRGBtoGRAY(self.test_rgb)
        
        if(self.colorspace == "LAB"):
            self.test_colored = self.fromRGBtoLAB(self.test_rgb)
        else:
            self.test_colored = self.test_rgb
        print("TEST IMAGE LOADED")
                    
    def get_validation_data(self):
        return self.test_gray, {"Output_color" : self.test_colored, "Output_class_" + str(self.n_classes) : keras.utils.to_categorical(self.test_labels, num_classes=self.n_classes)}
    
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def load(self, path):
        image = cv2.resize(np.array(Image.open(path)), (self.h, self.w))
        if(len(image.shape) == 2):
            image = np.array([image, image, image]).transpose(1, 2, 0)
        return image
            
    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples'
        # Initialization
        # Generate data
        if(self.dataset == "cifar10"):
            grays_batch = np.array([self.train_gray[ID] for ID in list_IDs_temp])
            colored_batch = np.array([self.train_colored[ID] for ID in list_IDs_temp])
            y_batch = np.array([self.train_labels[ID] for ID in list_IDs_temp])
            
        else:
            grays_batch = np.empty((self.batch_size, self.w, self.h, 1))
            colored_batch = np.empty((self.batch_size, self.w, self.h, self.n_channels))
            y_batch = np.empty((self.batch_size), dtype=int)
                
            for i, ID in enumerate(list_IDs_temp):
                colored_batch[i,] = self.load(self.list_paths[ID])
                # Store class
                y_batch[i] = self.labels[ID]
                
            colored_batch = self.preprocess(colored_batch)
            grays_batch = self.fromRGBtoGRAY(colored_batch)
            if(self.colorspace == "LAB"):
                colored_batch = self.fromRGBtoLAB(colored_batch)

        return grays_batch, {"Output_color" : colored_batch, "Output_class_" + str(self.n_classes) : keras.utils.to_categorical(y_batch, num_classes=self.n_classes)}
