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
    return cost + tf.losses.get_regularization_losses()