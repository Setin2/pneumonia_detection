import os.path
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.vis_utils import plot_model
import numpy as np
from keras import activations

def custom_keras_generator():
    """
     Read the images in .txt format
    """
    return None

# link to documentation of creating our own keras layers: https://keras.io/guides/making_new_layers_and_models_via_subclassing/
# we are allowed to use backend functions like K.dot(x, y)or K.conv2d(x, kernel)
class Conv2D(keras.layers.Layer):
    """
     Custom implementation of a convolutional layer
    """
    def __init__(self):
      empty = True
    
    def call(self, input):
      return input

class Linear(keras.layers.Layer):
    """
      Custom implementation of a dense layer
    """
    def __init__(self, n_inputs, n_neurons, activation):
        super(Linear, self).__init__()
        # randomly generate weights
        weights_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value = weights_init(shape=(n_inputs, n_neurons), dtype="float32"), trainable=True,
        )
        # generate bias
        bias_initializer = tf.zeros_initializer()
        self.b = tf.Variable(
            initial_value = bias_initializer(shape=(n_neurons,), dtype="float32"), trainable=True
        )
        # get activation function
        self.activation = activations.get(activation)
    
    # takes 1 argument, the input dataset
    def call(self, inputs):
        outputs = tf.matmul(inputs, self.w) + self.b
        outputs = self.activation(outputs) 
        return outputs

class MaxPooling2D(keras.layers.Layer):
    """
     Custom implementation of a pooling layer
    """
    def __init__(self, pool_size=(2, 2)):
      super(MaxPooling2D, self).__init__()
      self.pool_size = pool_size
    
    def call(self, pools):
      # store the max values to pooled
      pooled = []
      for tensor in tf.shape(pools):
        pooled.append(tf.reduce_max(tensor))

      # reshape and convert back to tensor
      pooled = np.array(pooled).reshape(self.pool_size)
      pooled = tf.convert_to_tensor(pooled, dtype=tf.float32)

      return pooled

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
      # since Conv2D is the first layer of the neural network, we should also specify the size of the input
      keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(resolution, resolution, 3)),
      # apply pooling
      keras.layers.MaxPooling2D(2, 2),
      keras.layers.Dropout(0.2),
      # and repeat the process
      keras.layers.Conv2D(32, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D(2, 2),
      keras.layers.Dropout(0.2),

      keras.layers.Conv2D(64, (3, 3), activation='relu'),
      keras.layers.MaxPooling2D(2, 2),
      keras.layers.Dropout(0.2),
      # flatten the result to feed it to the dense layer
      keras.layers.Flatten(),
      # and define 512 neurons for processing the output coming by the previous layers
      Linear(43264, 512, activation="relu"),
      # a single output neuron. The result will be 0 if the image is a cat, 1 if it is a dog
      Linear(512, 1, activation="sigmoid")
    ])

    # compiling the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # print the summary of the model
    print(model.summary())
    # save the model architecture into image file
    plot_model(model, show_shapes=True, to_file='model_plot.png')

    return model

def test_pooling_layer():
    conv_output = np.array([[10, 12,  8,  7],[ 4, 11,  5,  9],[18, 13,  7,  7],[ 3, 15,  2,  2]])
    conv_output = tf.convert_to_tensor(conv_output, dtype=tf.float32)
    pooling_layer = MaxPooling2D()
    y = pooling_layer(conv_output)
    print(y)

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
    model = create_model(resolution, load_previous_model=False)
    model.fit(training_set, steps_per_epoch=len(training_set), epochs=10)
