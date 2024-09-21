#!/usr/bin/env python3

""" A function that  that creates the training operation for the network
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    args:
    loss:the loss of the network’s prediction
    alpha:the learning rate

    returns:
     an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha)
    train_op = optimizer.minimize(loss)
    return train_op
