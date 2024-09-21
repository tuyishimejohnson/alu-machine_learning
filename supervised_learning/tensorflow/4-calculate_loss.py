#!/usr/bin/env python3

""" A function that calculates the softmax cross-entropy
loss of a prediction
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    args:
    y: the placeholder for labels of input data
    y_pred: tensor containing the network’s predictions

    returns:
    a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
