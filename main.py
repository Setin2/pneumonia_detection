import sys
import math
import os.path
import numpy as np
import pandas as pd
import scipy.ndimage
import tensorflow as tf
from tensorflow import keras
from keras import activations
from skimage import transform
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils.vis_utils import plot_model
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.attention.multi_head_attention import activation

MODEL_NAME = "model.h5"

class CustomDataGenerator(tf.keras.utils.Sequence):
    ''' 
        Custom DataGenerator to load text images 
    '''
    def __init__(self, path, rescale=None, shear_range=None, zoom_range=None, rotation_range=None, 
                horizontal_flip=False, vertical_flip=False, batch_size=10, num_classes=2):
        self.data_frame = None
        self.get_data_frame(path)
        self.n = len(self.data_frame)
        self.batch_size = batch_size
        self.num_classes = num_classes

        self.rescale = rescale
        self.shear_range = shear_range
        self.zoom_range = zoom_range
        self.rotation_range = rotation_range
        self.horizontal_flip = horizontal_flip
        self.vertical_flip = vertical_flip
        print(f"Found {self.data_frame.shape[0]} images belonging to {self.num_classes} classes")

    def __len__(self):
        ''' 
            return the number of batches
        '''
        return math.ceil(self.n / self.batch_size)
    
    def __data_augmentation(self, image):
        ''' 
            apply data augmentation to an image 
        '''
        if self.vertical_flip: image = np.flip(image, axis=0)
        if self.horizontal_flip: image = np.flip(image, axis=1)
        if self.rescale:
            image *= self.rescale
        if self.shear_range:
            transform_matrix = transform.AffineTransform(shear=self.shear_range)
            image = transform.warp(image, inverse_map=transform_matrix)
        # following augmentations are inspired by the source code for the keras ImageDataGenerator
        # https://github.com/keras-team/keras/blob/v2.11.0/keras/preprocessing/image.py#L1166-L2144
        if self.zoom_range:
            zx, zy = np.random.uniform(1 - self.zoom_range, 1 + self.zoom_range, 2)
            zoom_matrix = np.array([[zx, 0, 0], [0, zy, 0], [0, 0, 1]])
            image = np.dot(image, zoom_matrix)
        if self.rotation_range:
            theta = np.random.uniform(-self.rotation_range, self.rotation_range)
            theta = np.deg2rad(theta)
            rotation_matrix = np.array(
                [
                    [np.cos(theta), -np.sin(theta), 0],
                    [np.sin(theta), np.cos(theta), 0],
                    [0, 0, 1]
                ]
            )
            image = np.dot(image, rotation_matrix)
        return image

    def __get_image(self, file_path):
        '''
            return an image from a specified path as a numpy array
        '''
        image = []
        with open(file_path) as file:
            for line in file: image.append([float(x) for x in line.split()])
        # convert to 3 color channel
        image = np.stack((image,)*3, axis=-1)
        image = self.__data_augmentation(image)
        return image

    def __getitem__(self, idx):
        x = self.data_frame["filenames"][idx * self.batch_size:(idx + 1) * self.batch_size]
        y = self.data_frame["labels"][idx * self.batch_size:(idx + 1) * self.batch_size]

        x = [self.__get_image(file_name) for file_name in x] 
        y = [label for label in y]

        return tf.convert_to_tensor(x), tf.convert_to_tensor(y)
    
    def get_data_frame(self, path):
        """
            Get the path to all files in the directories and theirs labels
        """
        df = pd.DataFrame()
        filenames = []
        labels = []
        image_class = 0
        
        for directory in os.listdir(path):
            if not directory.startswith("."):
                images = os.listdir(os.path.join(path, directory))
            
                for image in images:
                    filenames.append(os.path.join(path, directory, image))
                    labels.append(image_class)
            
        df["filenames"] = filenames
        df["labels"] = labels
        
        self.data_frame = df

class Conv2D(keras.layers.Layer):
    """
     Custom implementation of a convolutional layer
    """

    def __init__(self, filters, kernel, activation, **kwargs):
        self.filters = filters
        self.k_h, self.k_w = kernel
        self.kernel = kernel
        self.activation = activations.get(activation)
        super(Conv2D, self).__init__(**kwargs)

    def build(self, input_shape):
        _, self.h, self.w, self.c = input_shape

        self.out_h = self.h - self.k_h + 1
        self.out_w = self.w - self.k_w + 1

        self.kernel_size = self.k_h * self.k_w * self.c
        self.kernels = self.add_weight(name='kernel', shape=[
                                       self.k_h, self.k_w, self.c, self.filters],
                                       initializer='glorot_uniform',
                                       trainable=True)

        super(Conv2D, self).build(input_shape)

    def call(self, inputs):
        outputs = K.conv2d(inputs, self.kernels, strides=(
            1, 1), padding='same', data_format=None, dilation_rate=(1, 1))

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.out_h, self.out_w, self.filters)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "filters": self.filters,
            "kernel_size": self.kernel_size
        })
        return config

class Linear(keras.layers.Layer):
    """
      Custom implementation of a dense layer
    """

    def __init__(self,n_neurons, activation, **kwargs):
        super(Linear, self).__init__(**kwargs)
        self.n_neurons = n_neurons
        self.activation = activations.get(activation)

    def build(self, input_shape):
        self.w = self.add_weight(shape=(input_shape[-1], self.n_neurons),initializer="random_normal",trainable=True, name='w')
        self.b = self.add_weight(shape=(self.n_neurons,), initializer="random_normal", trainable=True, name='b')

    # takes 1 argument, the input dataset
    def call(self, inputs):
        outputs = tf.matmul(inputs, self.w) + self.b
        outputs = self.activation(outputs)
        return outputs
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "n_neurons": self.n_neurons,
            "activation": self.activation,
        })
        return config


class MaxPooling2D(keras.layers.Layer):
    """
     Custom implementation of a pooling layer
    """

    def __init__(self, pool_size=(2, 2)):
        super(MaxPooling2D, self).__init__()
        self.pool_size = pool_size

    def call(self, pools):
        pooled = K.pool2d(x=pools,
                          pool_size=self.pool_size,
                          strides=(2, 2),
                          padding='valid',
                          pool_mode='max')

        return pooled
    
    def get_config(self):
        config = super().get_config()
        config.update({
            "pool_size": self.pool_size
        })
        return config

def create_model(resolution, load_previous_model=True):
    """ Return a keras model
    Either load a  preexisting model, if there is one, or create a new model from scratch
    """
    if os.path.isfile(MODEL_NAME) and load_previous_model:
        return tf.keras.models.load_model(MODEL_NAME, custom_objects={'Linear': Linear})
    else:
        # here we need to implement the model
        model = keras.models.Sequential([
            keras.layers.Input(shape=[resolution, resolution, 3]),
            # here we need to use the layers we will implement
            # since Conv2D is the first layer of the neural network, we should also specify the size of the input
            Conv2D(filters=16, kernel=(3, 3), activation='relu'),
            # apply pooling
            MaxPooling2D(),
            keras.layers.Dropout(0.2),
            # and repeat the process
            Conv2D(filters=16, kernel=(3, 3), activation='relu'),
            # keras.layers.Conv2D(32, (3, 3), activation='relu'),
            MaxPooling2D(),
            keras.layers.Dropout(0.2),

            Conv2D(filters=64, kernel=(3, 3), activation='relu'),
            # keras.layers.Conv2D(64, (3, 3), activation='relu'),
            MaxPooling2D(),
            keras.layers.Dropout(0.2),
            # flatten the result to feed it to the dense layer
            keras.layers.Flatten(),

            # model.add(keras.layers.Flatten())
            # and define 512 neurons for processing the output coming by the previous layers
            Linear(n_neurons=512, activation="relu"),
            # keras.layers.Dense(512, activation='relu'),
            # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
            Linear(n_neurons=1, activation="sigmoid"),
            # keras.layers.Dense(1, activation='sigmoid'),
        ])

        model.build((None, resolution, resolution, 3))

        # compiling the model
        model.compile(loss='binary_crossentropy',
                      optimizer='adam', metrics=['accuracy'])

        # print the summary of the model
        print(model.summary())
        # save the model architecture into image file
        plot_model(model, show_shapes=True, to_file='model_plot.png')

        return model

def train(model, training_set, epochs, save):
    history = model.fit(training_set, steps_per_epoch=len(training_set), epochs=epochs)
    if save: model.save('model.h5', include_optimizer=False)
    return history

def model_eval(model, testing_set):
  # evaluate model
  results = model.evaluate(testing_set)
  # print evaluation results
  print("test loss, test acc:", results)

  # create model predictions
  predictions = model.predict(testing_set)
  # normalize the predictions
  predictions[predictions <= 0.5] = 0
  predictions[predictions > 0.5] = 1

  # print predictions
  # print("Predictions: ", predictions)

def plot_learning_curve(history, figure_name):
  history_dict = history.history
  loss_values = history_dict['loss']
  accuracy = history_dict['accuracy']

  epochs = range(1, len(loss_values) + 1)
  fig, ax = plt.subplots(1, 2, figsize=(14, 6))

  # plot accuracy
  ax[0].plot(epochs, accuracy, label='Training accuracy')
  ax[0].set_title('Training Accuracy')
  ax[0].set_xlabel('Epochs')
  ax[0].set_ylabel('Accuracy')
  ax[0].legend()

  # plot loss
  ax[1].plot(epochs, loss_values, label='Training loss')
  ax[1].set_title('Training Loss')
  ax[1].set_xlabel('Epochs')
  ax[1].set_ylabel('Loss')
  ax[1].legend()

  plt.savefig(figure_name)

if __name__ == "__main__":
    resolution = 224  # 224x224x3

    # get training data
    data_generator = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, rotation_range=45,
                                        horizontal_flip=True, vertical_flip=True, validation_split=.2, brightness_range=[0.4, 1.5])
    training_set = data_generator.flow_from_directory('./train/train', target_size=(
        resolution, resolution), batch_size=20, class_mode='binary', subset='training')

    testing_generator = CustomDataGenerator("./test_encoded/test_encoded", rescale=1./255, shear_range=0.2, zoom_range=0.2, rotation_range=45, 
                                            horizontal_flip=True, vertical_flip=True)

    model = create_model(resolution, load_previous_model=True)
    history = train(model, training_set, epochs=1, save=True)
    model_eval(model, testing_generator)
    plot_learning_curve(history, "results")
