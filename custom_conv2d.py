from keras import backend as K
from keras.layers import Layer


class custom_conv2d(Layer):
    def __init__(self, filters, kernel, activation, **kwargs):
        self.filters = filters
        self.k_h, self.k_w = kernel
        self.activation = activation
        super(custom_conv2d, self).__init__(**kwargs)

    def build(self, input_shape):
        _, self.h, self.w, self.c = input_shape

        self.out_h = self.h - self.k_h + 1
        self.out_w = self.w - self.k_w + 1

        self.kernel_size = self.k_h * self.k_w * self.c
        self.kernels = self.add_weight(name='kernel', shape=[
                                       self.k_h, self.k_w, self.c, self.filters],
                                       initializer='glorot_uniform',
                                       trainable=True)

        super(custom_conv2d, self).build(input_shape)

    def call(self, inputs):
        # flatten the kernels
        kernel = K.reshape(self.kernels, [self.kernel_size, self.filters])

        outputs = K.conv2d(inputs, kernel, strides=(
            1, 1), padding='same', data_format=None, dilation_rate=(1, 1))

        if self.activation is not None:
            return self.activation(outputs)

        return outputs

    def compute_output_shape(self, input_shape):
        return (None, self.out_h, self.out_w, self.filters)
