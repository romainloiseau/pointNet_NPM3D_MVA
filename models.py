import numpy as np
from keras.models import Model
from keras.layers import Input, Conv1D, Lambda, MaxPooling1D, Flatten, Dense, Reshape, RepeatVector, Concatenate
from keras import backend as K

def build_point_net(input_shape = (2048, 3), output_shape = 10, mode = "classification"):
    
    assert mode in ["classification", "segmentation"]

    features = Input(input_shape, name = "input_features")
    
    def multiply(input_tensors):
        dot = K.batch_dot(input_tensors[0], input_tensors[1])
        return dot
    
    transform3 = build_T_net((input_shape[0], 3), name = "T_net_3")(features)
    transformed3 = Lambda(multiply, name = "transformed3")([features, transform3])
    
    conv10 = Conv1D(filters = 64, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv10")(transformed3)
    conv11 = Conv1D(filters = 64, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv11")(conv10)
    
    transform64 = build_T_net((input_shape[0], 64), name = "T_net_64")(conv11)
    transformed64 = Lambda(multiply, name = "transformed64")([conv11, transform64])
    
    conv20 = Conv1D(filters = 64, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv20")(transformed64)
    conv21 = Conv1D(filters = 128, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv21")(conv20)
    conv22 = Conv1D(filters = 1024, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv22")(conv21)
    
    global_features = MaxPooling1D(pool_size = input_shape[0], strides = None, padding = "valid")(conv22)
    global_features = Flatten()(global_features)
    
    dense0 = Dense(512, activation = "relu")(global_features)
    dense1 = Dense(256, activation = "relu")(dense0)
    dense2 = Dense(output_shape, activation = "softmax")(dense1)
    
    if(mode == "classification"):
        model = Model(inputs = features, outputs = dense2)
        return model
    
    elif(mode == "segmentation"):
        
        input_segmentation = Concatenate()([transformed64, RepeatVector(input_shape[0])(global_features)])
        
        conv30 = Conv1D(filters = 512, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv30")(input_segmentation)
        conv31 = Conv1D(filters = 256, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv31")(conv30)
        conv32 = Conv1D(filters = 128, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv32")(conv31)
        conv33 = Conv1D(filters = 128, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = "conv33")(conv32)
        conv34 = Conv1D(filters = m, kernel_size = (1), padding = 'valid', strides = (1), activation = "softmax", name = "conv34")(conv33)
        
        model = Model(inputs = features, outputs = dense2)
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

def build_T_net(input_shape = (2048, 3), name = ""):
    
    features = Input(input_shape, name = "input_features")
    
    conv10 = Conv1D(filters = 64, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = name + "_conv10")(features)
    conv11 = Conv1D(filters = 128, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = name + "_conv11")(conv10)
    conv12 = Conv1D(filters = 1024, kernel_size = (1), padding = 'valid', strides = (1), activation = "relu", name = name + "_conv12")(conv11)
    
    global_features = MaxPooling1D(pool_size = input_shape[0], strides = None, padding = "valid")(conv12)
    global_features = Flatten()(global_features)
    
    dense0 = Dense(512, activation = "relu")(global_features)
    dense1 = Dense(256, activation = "relu")(dense0)
    
    transform = TransformLayer(input_shape[1])(dense1)
    transform = Reshape((input_shape[1], input_shape[1]), name = name + "_reshape")(transform)
    
    if(name != ""):model = Model(inputs = features, outputs = transform, name = str(name))
    else:model = Model(inputs = features, outputs = transform)
    
    return model