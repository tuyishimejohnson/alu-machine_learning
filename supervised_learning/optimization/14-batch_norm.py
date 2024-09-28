#!/usr/bin/env python3

""" A function that creates a batch
normalization layer for a neural network in tensorflow:
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    args:
    prev is the activated output of the previous layer
n is the number of nodes in the layer to be created
activation is the activation function that should be used on the output of the layer
    returns:
      the updated value for alpha

    """
    dense_layer = tf.layers.Dense(
        units=n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    )
    
    Z = dense_layer(prev)
    
    mean, variance = tf.nn.moments(Z, axes=[0])
    
    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')
    
    Z_batch_norm = tf.nn.batch_normalization(
        Z, mean, variance, beta, gamma, epsilon=1e-8
    )
    
    if activation is not None:
        return activation(Z_batch_norm)
    else:
        return Z_batch_norm