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
        A_prev = cache['A' + str(i - 1)]

        Z = np.dot(W, A_prev) + b

        if i == L:
            # Softmax activation for the last layer
            t = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = t / np.sum(t, axis=0, keepdims=True)
        else:
            # Tanh activation for hidden layers
            A = np.tanh(Z)
            # Dropout mask
            D = np.random.rand(A.shape[0], A.shape[1]) < keep_prob
            A = np.multiply(A, D)
            A /= keep_prob
            cache['D' + str(i)] = D

        cache['A' + str(i)] = A

    return cache
