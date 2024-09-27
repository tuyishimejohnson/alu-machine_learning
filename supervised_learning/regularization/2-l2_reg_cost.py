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
    l2_loss = tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    
    # Add the L2 regularization loss to the original cost
    total_cost = cost + l2_loss
    
    return total_cost