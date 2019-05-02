import tensorflow as tf
import numpy as np
from tensorpack import *

def conv(name, input_layer, output_dim, use_bias = True):
    return Conv2D(name, input_layer, output_dim, 3, stride=1, nl=tf.identity, use_bias=use_bias, W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / output_dim)))

def fully_connected(scope, layer, out_dim):
    return FullyConnected(scope, layer, out_dim=out_dim, nl=tf.identity)

'''
def conv2d(scope, input_layer, output_dim, use_bias=False, filter_size=3, strides=[1, 1, 1, 1]):

    input_dim = input_layer.get_shape().as_list()[-1]

    with tf.variable_scope(scope):
        conv_filter = tf.get_variable(
            'conv_weight',
            shape=[filter_size, filter_size, input_dim, output_dim],
            dtype=tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002)
        )
        conv = tf.nn.conv2d(input_layer, conv_filter, strides, 'SAME')

        if use_bias:
            bias = tf.get_variable(
                'conv_bias',
                shape=[output_dim],
                dtype=tf.float32,
                initializer=tf.constant_initializer(0.0)
            )

            output_layer = tf.nn.bias_add(conv, bias)
            output_layer = tf.reshape(output_layer, conv.get_shape())
        else:
            output_layer = conv

        return output_layer
'''
def batch_norm(scope, input_layer, is_training, reuse):
    output_layer = tf.contrib.layers.batch_norm(
        input_layer,
        decay=0.9,
        scale=True,
        epsilon=1e-5,
        is_training=is_training,
        reuse=reuse,
        scope=scope
    )

    return output_layer


def lrelu(input_layer):
    output_layer = tf.nn.relu(input_layer)
    return output_layer

'''
def fully_connected(scope, input_layer, output_dim):
    input_dim = input_layer.get_shape().as_list()[-1]

    with tf.variable_scope(scope):
        fc_weight = tf.get_variable(
            'fc_weight',
            shape=[input_dim, output_dim],
            dtype=tf.float32,
            initializer=tf.contrib.layers.variance_scaling_initializer(),
            regularizer=tf.contrib.layers.l2_regularizer(scale=0.0002)
        )

        fc_bias = tf.get_variable(
            'fc_bias',
            shape=[output_dim],
            dtype=tf.float32,
            initializer=tf.constant_initializer(0.0)
        )

        output_layer = tf.matmul(input_layer, fc_weight) + fc_bias

        return output_layer
'''

def avg_pool(scope, input_layer, ksize=None, strides=[1, 2, 2, 1]):
    if ksize is None:
        ksize = strides

    with tf.variable_scope(scope):
        output_layer = tf.nn.avg_pool(input_layer, ksize, strides, 'VALID')
        return output_layer


def Global_Average_Pooling(x, stride=1):
    width = np.shape(x)[1]
    height = np.shape(x)[2]
    pool_size = [width, height]
    return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride)