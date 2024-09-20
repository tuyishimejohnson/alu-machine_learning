#!/usr/bin/env python3

""" A function that is used to calculate the
    placeholder of two neural networks
"""

import tensorflow as tf


def create_placeholders(nx, classes):
    """ A function that is used to calculate the
    placeholder of two neural networks
"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")

    return x, y
