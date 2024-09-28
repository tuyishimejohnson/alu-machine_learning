#!/usr/bin/env python3
""" A function that conducts forward propagation using Dropout
"""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """
    args:
    X:numpy.ndarray of shape (nx, m) containing
     the input data for the network
    nx:number of input features
    m:number of data points
    weights:dictionary of the weights and biases of the neural network
    L:the number of layers in the network
    keep_prob:probability that a node will be kept
    All layers except the last should use the tanh activation function
    The last layer should use the softmax activation function
    """
    cache = {}
    cache['A0'] = X
    for i in range(1, L + 1):
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]
        A = cache['A' + str(i - 1)]
        Z = np.matmul(W, A) + b
        if i < L:
            A = np.tanh(Z)
            D = np.random.rand(A.shape[0], A.shape[1])
            D = np.where(D < keep_prob, 1, 0)
            A = np.multiply(A, D)
            A = A / keep_prob
            cache['D' + str(i)] = D
        else:
            A = np.exp(Z) / np.sum(np.exp(Z), axis=0, keepdims=True)
        cache['A' + str(i)] = A
    return cache
