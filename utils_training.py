import numpy as np
from keras import backend as K
import tensorflow as tf

def my_categorical_crossentropy(y_true, y_pred):
    return K.mean(K.sum(y_true, axis = -1)) * K.categorical_crossentropy(y_true, y_pred)

class mIoU(object):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def miou(self, y_true, y_pred):
        # Wraps np_mean_iou method and uses it as a TensorFlow op.
        # Takes numpy arrays as its arguments and returns numpy arrays as
        # its outputs.
        return tf.py_func(self.np_miou, [y_true, y_pred], tf.float32)

    def np_miou(self, y_true, y_pred):
        # Compute the confusion matrix to get the number of true positives,
        # false positives, and false negatives
        # Convert predictions and target from categorical to integer format
        
        valid = (np.sum(y_true, axis = -1).ravel()).astype(bool)
        target = np.argmax(y_true, axis=-1).ravel()[valid]
        predicted = np.argmax(y_pred, axis=-1).ravel()[valid]
        
        # Trick from torchnet for bincounting 2 arrays together
        # https://github.com/pytorch/tnt/blob/master/torchnet/meter/confusionmeter.py
        x = predicted + self.num_classes * target
        bincount_2d = np.bincount(x.astype(np.int32), minlength=self.num_classes**2)
        assert bincount_2d.size == self.num_classes**2
        conf = bincount_2d.reshape((self.num_classes, self.num_classes))

        # Compute the IoU and mean IoU from the confusion matrix
        true_positive = np.diag(conf)
        false_positive = np.sum(conf, 0) - true_positive
        false_negative = np.sum(conf, 1) - true_positive

        # Just in case we get a division by 0, ignore/hide the error and set the value to 0
        with np.errstate(divide='ignore', invalid='ignore'):
            iou = true_positive / (true_positive + false_positive + false_negative)
        iou[np.isnan(iou)] = 0
        return np.mean(iou).astype(np.float32)
    
def plot_history(history):
    history_dict = history.history
    plt.figure(figsize = (15, 5))
    for i, metric in enumerate(["loss", "miou", "acc"]):
        plt.subplot(131 + i)
        plt.title(metric)
        plt.plot(history_dict[metric], label = "train")
        plt.plot(history_dict["val_{}".format(metric)], label = "test")
        plt.legend(loc = "best")
    plt.show()