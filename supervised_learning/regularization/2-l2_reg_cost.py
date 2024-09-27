#!/usr/bin/env python3

""" updates the weights and biases of
a neural network using gradient descent with L2 regularization
"""

import tensorflow as tf


def l2_reg_cost(cost):
    """
    args:
    cost:tensor containing the cost of the network
    without L2 regularization

    returns:
    tensor containing the cost of the network
    accounting for L2 regularization
    """
    l2_term = tf.add_n([
        tf.nn.l2_loss(var) for var in tf.trainable_variables()
        if 'kernel' in var.name
    ])
    l2_cost = cost + l2_term
    return l2_cost
