import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, Lambda, MaxPooling1D, Flatten, Dense, Reshape, RepeatVector, Concatenate, BatchNormalization
from keras import backend as K
import tensorflow as tf
from tensorflow.python.keras.models import model_from_json

def build_point_net(input_shape = (2048, 3), output_shape = 10, refined_points = 25, mode = "segmentation", normalize = False):
    
    assert mode in ["classification", "segmentation"]

    features = Input(input_shape, name = "input_features")
    
    def multiply(input_tensors):
        dot = K.batch_dot(input_tensors[0], input_tensors[1])
        return dot
    
    transform3 = build_T_net((input_shape[0], input_shape[1]), normalize = normalize, name = "T_net_3")(features)
    transformed3 = Lambda(multiply, name = "transformed3")([features, transform3])
    
    conv10 = Conv1D(filters = 64, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv10")(transformed3)
    if normalize:conv10 = MyBN(name = "conv10_bn")(conv10)
    
    conv11 = Conv1D(filters = 64, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv11")(conv10)
    if normalize:conv11 = MyBN(name = "conv11_bn")(conv11)
        
    transform64 = build_T_net((input_shape[0], 64), normalize = normalize, name = "T_net_64")(conv11)
    transformed64 = Lambda(multiply, name = "transformed64")([conv11, transform64])
    
    conv20 = Conv1D(filters = 64, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv20")(transformed64)
    if normalize:conv20 = MyBN(name = "conv20_bn")(conv20)
        
    conv21 = Conv1D(filters = 128, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv21")(conv20)
    if normalize:conv21 = MyBN(name = "conv21_bn")(conv21)
        
    conv22 = Conv1D(filters = 1024, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv22")(conv21)
    if normalize:conv22 = MyBN(name = "conv22_bn")(conv22)
        
    global_features = MaxPooling1D(pool_size = input_shape[0], strides = None, padding = "valid")(conv22)
    global_features = Flatten()(global_features)
    
    if(mode == "classification"):
        dense0 = Dense(512, activation = "relu", name = "dense0")(global_features)
        if normalize:dense0 = MyBN(name = "dense0_bn")(dense0)
            
        dense1 = Dense(256, activation = "relu", name = "dense1")(dense0)
        if normalize:dense1 = MyBN(name = "dense1_bn")(dense1)
            
        dense2 = Dense(output_shape, activation = "softmax")(dense1)
    
        model = Model(inputs = features, outputs = dense2)
        return model
    
    elif(mode == "segmentation"):
        
        input_segmentation = Concatenate()([transformed3, conv21, transformed64, RepeatVector(input_shape[0])(global_features)])
        
        conv30 = Conv1D(filters = 512, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv30")(input_segmentation)
        if normalize:conv30 = MyBN(name = "conv30_bn")(conv30)
            
        conv31 = Conv1D(filters = 256, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv31")(conv30)
        if normalize:conv31 = MyBN(name = "conv31_bn")(conv31)
            
        conv32 = Conv1D(filters = 128, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv32")(conv31)
        if normalize:conv32 = MyBN(name = "conv32_bn")(conv32)
            
        conv33 = Conv1D(filters = 128, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv33")(conv32)
        if normalize:conv33 = MyBN(name = "conv33_bn")(conv33)
            
        conv34 = Conv1D(filters = output_shape, kernel_size = (1), padding = 'valid', strides = (1), activation = "softmax", name = "conv34")(conv33)
        
        model = Model(inputs = features, outputs = conv34)
        return model

from keras.layers import Layer
class TransformLayer(Layer):
    def __init__(self, K, **kwargs):
        self.K = K
        super(TransformLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1], self.K * self.K),
                                      initializer='uniform',
                                      trainable=True)
        self.biais = K.variable(np.eye(self.K).flatten())
        super(TransformLayer, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        dot = K.dot(x, self.kernel)
        dot_plus_biais = K.bias_add(dot, self.biais)
        return dot_plus_biais

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.K * self.K)

def build_T_net(input_shape = (2048, 3), normalize = True, name = ""):
    
    features = Input(input_shape, name = "input_features")
    
    conv10 = Conv1D(filters = 64, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = name + "_conv10")(features)
    if normalize:conv10 = MyBN(name = name + "_conv10_bn")(conv10)
        
    conv11 = Conv1D(filters = 128, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = name + "_conv11")(conv10)
    if normalize:conv11 = MyBN(name = name + "_conv11_bn")(conv11)
        
    conv12 = Conv1D(filters = 1024, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = name + "_conv12")(conv11)
    if normalize:conv12 = MyBN(name = name + "_conv12_bn")(conv12)
    
    global_features = MaxPooling1D(pool_size = input_shape[0], strides = None, padding = "valid")(conv12)
    global_features = Flatten()(global_features)
    
    dense0 = Dense(512, activation = "relu", name = name + "_dense0")(global_features)
    if normalize:dense0 = MyBN(name = name + "_dense0_bn")(dense0)
        
    dense1 = Dense(256, activation = "relu", name = name + "_dense1")(dense0)
    if normalize:dense1 = MyBN(name = name + "_dense1_bn")(dense1)
    
    transform = TransformLayer(input_shape[1])(dense1)
    transform = Reshape((input_shape[1], input_shape[1]), name = name + "_reshape")(transform)
    
    if(name != ""):model = Model(inputs = features, outputs = transform, name = str(name))
    else:model = Model(inputs = features, outputs = transform)
    
    return model

class MyBN(Layer):
    def __init__(self, **kwargs):
        super(MyBN, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        print(input_shape)
        self.bn = tf.layers.BatchNormalization(axis = -1)
        super(MyBN, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        return tf.map_fn(lambda xx: self.bn(xx), x)

    def compute_output_shape(self, input_shape):
        return input_shape
    

def save_model(model, output_path):
    model_json = model.to_json()
    with open(output_path + ".json", "w") as json_file:
        json_file.write(model_json)
    # serialize weights to HDF5
    model.save_weights(output_path + ".h5")
    print("Saved model to disk")
    
def load_model(input_path):
    json_file = open(input_path + '.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    #loaded_model = model_from_json(loaded_model_json, {'TransformLayer': TransformLayer})
    params = input_path.split("_")
    loaded_model = build_point_net(input_shape = (int(params[1]), int(params[2])), output_shape = int(params[3]))
    loaded_model.load_weights(input_path + '.h5')
    print("Loaded model from disk")
    
    return loaded_model