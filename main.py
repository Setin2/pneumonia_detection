import os.path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import numpy as np

def custom_keras_generator():
    """
     Read the images in .txt format
    """
    return None

# link to documentation of creating our own keras layers: https://keras.io/guides/making_new_layers_and_models_via_subclassing/
# we are allowed to use backend functions like K.dot(x, y)orK.conv2d(x, kernel)
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

def create_model(resolution, load_previous_model=True):
  """ Return a keras model

  Either load a  preexisting model, if there is one, or create a new model from scratch
  We  can use flatten function and such
  """
  if os.path.isfile('model.h5') and load_previous_model:
    return tf.keras.models.load_model('model.h5')
  else:
    # here we need to implement the model
    model = keras.models.Sequential([
      # here we need to use the layers we will implement
    ])

    # compiling the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # print the summary of the model
    print(model.summary())
    # save the model architecture into image file
    plot_model(model, show_shapes=True, to_file='model_plot.png')

    return model

if __name__ == "__main__":
    resolution = 224  # 224x224x3

    # get training data
    data_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, rotation_range=45,
                                        horizontal_flip=True, vertical_flip=True, validation_split=.2, brightness_range=[0.4, 1.5])
    training_set = data_generator.flow_from_directory('./train/train', target_size=(
        resolution, resolution), batch_size=32, class_mode='binary', subset='training')

    # testing data is encoded in .txt files
    # each file contains a 224x224 matrix of the pixelvalues of an image
    # the matrix is 2D, so we only have 1 color channel (fine, since we will train the model on 1 color channel anyways)