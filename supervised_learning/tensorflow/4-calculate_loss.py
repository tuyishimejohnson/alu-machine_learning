#!/usr/bin/env python3

""" A function that calculates the softmax cross-entropy 
loss of a prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    args:
    y: the placeholder for labels of input data
    y_pred: tensor containing the network’s predictions

    returns:
    a tensor containing the loss of the prediction
    """
    return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_pred))
