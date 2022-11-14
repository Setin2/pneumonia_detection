import os.path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import numpy as np

def conv2d():
    """
     Our own implementation of a convolutional layer
    """
    return None

def linear():
    """
     Our own implementation of a dense layer
    """
    return None

def maxPooling2D():
    """
     Our own implementation of a pooling layer
    """
    return None

if __name__ == "__main__":
    resolution = 224  # 224x224x3

    # get training data
    data_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, rotation_range=45,
                                        horizontal_flip=True, vertical_flip=True, validation_split=.2, brightness_range=[0.4, 1.5])
    training_set = data_generator.flow_from_directory('./train/train', target_size=(
        resolution, resolution), batch_size=32, class_mode='binary', subset='training')

    # testing data is encoded in .txt files