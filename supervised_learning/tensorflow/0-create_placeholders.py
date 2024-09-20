#!/usr/bin/env python3

""" A function that is used to calculate the
    placeholder of two neural networks
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """
    args:
    nx: number of feature columns in data
    classes: number of classes in classifier

    returns:
    x: a placeholder for input (Neural network)
    y: a placeholder for one-hot labels for input data
    """
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")

    return x, y
