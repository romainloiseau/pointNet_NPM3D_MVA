import tensorflow as tf

class PointNet(tf.keras.Model):
    
    def __init__(self,nb_classe=10,H=224,W=224):
        super(ColorNetwork, self).__init__()
        
        self.nb_classe = nb_classe
        self.inpu = tf.keras.layers.Input(shape=(H,W,1))
        
        #Low level features network 1
        self.conv10 = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3, 3), strides=(2, 2),
                                             activation = 'relu', padding = "same", name = "conv10")
        self.conv11 = tf.keras.layers.Conv2D(filters = 128, kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv11")
        self.conv12 = tf.keras.layers.Conv2D(filters = 128, kernel_size=(3, 3), strides=(2, 2),
                                             activation = 'relu', padding = "same", name = "conv12")
        self.conv13 = tf.keras.layers.Conv2D(filters = 256, kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv13")
        self.conv14 = tf.keras.layers.Conv2D(filters = 256, kernel_size=(3, 3), strides=(2, 2),
                                             activation = 'relu', padding = "same", name = "conv14")
        self.conv15 = tf.keras.layers.Conv2D(filters = 512, kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv15")
        #mid level features network
        self.conv20 = tf.keras.layers.Conv2D(filters = 512, kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv20")
        self.conv21 = tf.keras.layers.Conv2D(filters = 256, kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv21")
        #Low level features network 2
        #identique
    
        #Features Network
        self.conv30 = tf.keras.layers.Conv2D(filters = 512, kernel_size=(3, 3), strides=(2, 2),
                                             activation = 'relu', padding = "same", name = "conv30")
        self.conv31 = tf.keras.layers.Conv2D(filters = 512,kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv31")
        self.conv32 = tf.keras.layers.Conv2D(filters = 512,kernel_size=(3, 3), strides=(2, 2),
                                             activation = 'relu', padding = "same", name = "conv32")
        self.conv33 = tf.keras.layers.Conv2D(filters = 512,kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv33")
        self.flat34 = tf.keras.layers.Flatten(name = "flat34")
        self.dens35 = tf.keras.layers.Dense(1024, activation = 'relu', name = "dens35")
        self.dens36 = tf.keras.layers.Dense(512, activation = 'relu', name = "dens36")
    
        #classification 
        self.cl1= tf.keras.layers.Dense(256, activation = 'relu', name = "cl1")
        self.cl2= tf.keras.layers.Dense(nb_classe, activation = 'softmax', name = "Output_class_" + str(nb_classe))
    
        #feature Network partie replicate
        self.dens40 = tf.keras.layers.Dense(256, activation = 'relu', name = "dens40")
        self.dens41 = tf.keras.layers.RepeatVector(int(H/8.0) * int(W/8.0), name = "dens41") #tf.keras.layers.RepeatVector(28)
        self.dens42 = tf.keras.layers.Reshape((int(H/8.0), int(W/8.0), 256), name = "dens42") #tf.keras.layers.RepeatVector(28)

        #fusion
        self.con50 = tf.keras.layers.Concatenate()

        #Colorization Network
        self.conv60 = tf.keras.layers.Conv2D(filters = 256, kernel_size=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv60")
        self.conv61 = tf.keras.layers.Conv2D(filters = 128, kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv61")
        self.conv62 = tf.keras.layers.UpSampling2D((2, 2), name = "conv62")
        self.conv63 = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv63")
        self.conv64 = tf.keras.layers.Conv2D(filters = 64, kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv64")
        self.conv65 = tf.keras.layers.UpSampling2D((2, 2), name = "conv65")
        self.conv66 = tf.keras.layers.Conv2D(filters = 32, kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'relu', padding = "same", name = "conv66")
        self.conv67 = tf.keras.layers.Conv2D(filters = 2, kernel_size=(3, 3), strides=(1, 1),
                                             activation = 'sigmoid', padding = "same", name = "conv67")

        #last concat
        self.con70 = tf.keras.layers.Concatenate(name = "Output_color")

        #lats upsampling
        self.conv80 = tf.keras.layers.UpSampling2D((2, 2), name = "conv80")
        
    def computeLowLevels(self, array):
        x = self.conv10(array)
        x = self.conv11(x)
        x = self.conv12(x)
        x = self.conv13(x)
        x = self.conv14(x)
        x = self.conv15(x)
        return x
    
    def computeMidLevels(self, array):
        x = self.conv20(array)
        x = self.conv21(x)
        return x
    
    def computeFeatures(self, array):
        x = self.conv30(array)
        x = self.conv31(x)
        x = self.conv32(x)
        x = self.conv33(x)
        x = self.flat34(x)
        x = self.dens35(x)
        x = self.dens36(x)
        return x
    
    def computeClassif(self, array):
        x = self.cl1(array)
        x = self.cl2(x)
        return x
    
    def computeReplicate(self, array):
        x = self.dens40(array)
        x = self.dens41(x)
        x = self.dens42(x)
        return x
    
    def computeColorization(self, array):
        x = self.conv60(array)
        x = self.conv61(x)
        x = self.conv62(x)
        x = self.conv63(x)
        x = self.conv64(x)
        x = self.conv65(x)
        x = self.conv66(x)
        x = self.conv67(x)
        x = self.conv80(x)
        return x
    
    def call(self):
        gray_Image = self.inpu
        
        low = self.computeLowLevels(gray_Image)
        mid = self.computeMidLevels(low)
        
        def resiz(gray_Image):
            return tf.image.resize_images(gray_Image, (224, 224))

        gray_Image_scaled = tf.keras.layers.Lambda(resiz)(gray_Image)
        low_scaled = self.computeLowLevels(gray_Image_scaled)
        
        feature = self.computeFeatures(low_scaled)
        
        Output_class = self.computeClassif(feature)

        featureReplicate = self.computeReplicate(feature)
        
        fullFeatureMaps = self.con50([mid, featureReplicate])
        
        colored = self.computeColorization(fullFeatureMaps)

        Output_color = self.con70([gray_Image, colored])
        
        model = tf.keras.models.Model(inputs = [gray_Image], outputs = [Output_color, Output_class])

        alpha = 1 / self.nb_classe
        adad = tf.keras.optimizers.Adadelta(lr = 1.0, rho = 0.75, epsilon = None, decay = 0.0)#rho=0.95 par default
        model.compile(adad,
                      {"Output_color": "mse", "Output_class_" + str(self.nb_classe): "categorical_crossentropy"},
                      loss_weights = [1.0, alpha])

        return model