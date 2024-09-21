#!/usr/bin/env python3

""" A function that calculates the accuracy of a prediction
"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    args:
    y: the placeholder for labels of input data
    y_pred: tensor containing the network’s predictions

    returns:
    accuracy = correct_predictions / all_predictions
    """
    correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(y_pred, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))
    return accuracy
