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
    l2_term = 0
    for i in range(1, L + 1):
        l2_term += tf.nn.l2_loss(weights['W' + str(i)])
    
    l2_cost = cost + (lambtha / (2 * m)) * l2_term
    return l2_cost
