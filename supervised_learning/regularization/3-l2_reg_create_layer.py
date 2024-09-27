#!/usr/bin/env python3

""" a function that creates a tensorflow
layer that includes L2 regularization
"""
import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """
    args:
    prev:tensor containing the output of the previous layer
    n:number of nodes the new layer should contain
    activation:activation function that should be used on the layer
    lambtha:L2 regularization parameter

    returns:
     the output of the new layer
    """
    regularizer = tf.contrib.layers.l2_regularizer(lambtha)
    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"
        ),
        kernel_regularizer=regularizer
    )
    return layer(prev)
