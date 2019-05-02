import tensorflow as tf
import argparse
import os
import numpy as np
from tensorpack import *
from tensorpack.tfutils.symbolic_functions import *
from tensorpack.tfutils.summary import *

def conv(name, l, channel, stride):
    return Conv2D(name, l, channel, 3, stride=stride, nl=tf.identity, use_bias=False,
                  W_init=tf.random_normal_initializer(stddev=np.sqrt(2.0 / 9 / channel)))


def add_layer(name, l, growthRate):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        c = BatchNorm('bn1', l)
        c = tf.nn.relu(c)
        c = conv('conv1', c, growthRate, 1)
        c = Dropout('dropout', c, rate=0.2)
        l = tf.concat([c, l], 3)
    return l


def add_transition(name, l):
    shape = l.get_shape().as_list()
    in_channel = shape[3]
    with tf.variable_scope(name) as scope:
        l = BatchNorm('bn1', l)
        l = tf.nn.relu(l)
        l = Conv2D('conv1', l, in_channel, 1, stride=1, use_bias=False, nl=tf.nn.relu)
        l = Dropout('dropout', l, rate=0.2)
        l = AvgPooling('pool', l, 2)
    return l

def Drop_Out(x, rate, training):
    return tf.layers.dropout(input=x, rate=rate, training=training)