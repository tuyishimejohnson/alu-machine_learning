#!/usr/bin/env python3

""" updates the weights and biases of
a neural network using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    args:
    Y one-hot numpy.ndarray of shape (classes, m)
    that contains the correct labels for the data
    classes:number of classes
    m:number of data points
    weights:dictionary of the weights and biases of the neural network
    cache:dictionary of the outputs of each layer of the neural network
    alpha:learning rate
    lambtha:L2 regularization parameter
    L:number of layers of the network

    returns:
     the cost of the network accounting for L2 regularization
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = (1 / m) * np.dot(dZ, A_prev.T) + (lambtha / m) * W
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA_prev = np.dot(W.T, dZ)
            dZ = dA_prev * (1 - A_prev ** 2)
        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
