#!/usr/bin/env python3

""" A function that is used to the tensor output of the layer
"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """
    args:
    prev: output of the previous layer
    n: number of of nodes in the layer to create
    activation: activation function for a layer to use

    returns:
    tensor output
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(
        mode="FAN_AVG"
        )
    w = tf.Variable(
        initializer([prev.get_shape().as_list()[1], n]), name='weights'
    )
    b = tf.Variable(tf.zeros([n]), name='biases')
    layer = tf.add(tf.matmul(prev, w), b)
    if activation is not None:
        layer = activation(layer)
    return layer
