import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.initializers import RandomNormal
_leaky_relu = lambda net: tf.nn.leaky_relu(net, alpha=0.3)

# Resnet block
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    # first layer convolutional laycer
    g = layers.Conv1D(n_filters, 16, padding='same', kernel_initializer=init)(input_layer)
    g = layers.BatchNormalization()(g)
    g = layers.Activation('relu')(g)

    # second convolutional layer
    g = layers.Conv1D(n_filters, 16, padding='same', kernel_initializer=init)(g)
    g = layers.BatchNormalization()(g)

    # concatenate merge channel-wise with input layer
    g = layers.Concatenate()([g, input_layer])
    return g

# CNN block
def conv1d_block(filters, k_size, strides, padding, input):
    # init = RandomNormal(stddev=0.02)
    g = layers.Conv1D(filters, k_size, strides=strides, padding=padding)(input)
    g = layers.BatchNormalization()(g)
    g = layers.LeakyReLU(0.3)(g)
    return g

# Gradient reverse layer
@tf.custom_gradient
def GradientReversalOperator(x):
    def grad(dy):
        return -1 * dy

    return x, grad
class GradientReversalLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(GradientReversalLayer, self).__init__()

    def call(self, inputs):
        return GradientReversalOperator(inputs)

