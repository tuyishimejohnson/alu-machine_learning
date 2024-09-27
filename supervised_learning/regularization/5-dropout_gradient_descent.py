#!/usr/bin/env python3

""" updates the weights and biases of
a neural network using gradient descent with L2 regularization
"""

import numpy as np


def dropout_gradient_descent(Y, weights,
                             cache, alpha,
                             keep_prob, L):
    """
    Updates the weights of a neural network with
    Dropout regularization using gradient descent.

    Returns:
    None
    """
    m = Y.shape[1]
    dZ = cache['A' + str(L)] - Y

    for i in range(L, 0, -1):
        A_prev = cache['A' + str(i - 1)]
        W = weights['W' + str(i)]
        b = weights['b' + str(i)]

        dW = (1 / m) * np.dot(dZ, A_prev.T)
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        if i > 1:
            dA_prev = np.dot(W.T, dZ)
            dA_prev = dA_prev * cache['D' + str(i - 1)]
            dA_prev /= keep_prob
            dZ = dA_prev * (1 - A_prev ** 2)

        weights['W' + str(i)] = W - alpha * dW
        weights['b' + str(i)] = b - alpha * db
