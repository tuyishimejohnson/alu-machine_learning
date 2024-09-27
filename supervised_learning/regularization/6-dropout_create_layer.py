#!/usr/bin/env python3

""" a function that creates a layer of a neural network using dropout
"""

import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob):
    """
    args:
    prev:tensor containing the output of the previous layer
    n:number of nodes the new layer should contain
    activation:activation function that should be used on the layer
    keep_prob:probability that a node will be kept
    returns:
    the output of the new layer
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer
    )
    output = layer(prev)
    dropout = tf.nn.dropout(output, keep_prob)
    return dropout
