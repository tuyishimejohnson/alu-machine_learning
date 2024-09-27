#!/usr/bin/env python3

""" A function that calculates the cost of
a neural network with L2 regularization
"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """
    args:
    cost:cost of the network without L2 regularization
    lambtha:regularization parameter
    weights:dictionary of the weights and biases
      (numpy.ndarrays) of the neural network
    L: number of layers in the neural network
    m:number of data points used

    returns:
     the cost of the network accounting for L2 regularization
    """
    l2_term = 0
    for i in range(1, L + 1):
        l2_term += np.sum(np.square(weights['W' + str(i)]))

    l2_cost = cost + (lambtha / (2 * m)) * l2_term
    return l2_cost
