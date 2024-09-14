#!/usr/bin/env python3
""" a class for Neural Network
"""

import numpy as np


class DeepNeuralNetwork:
    """ Class that defines a deep neural network performing
        binary classification.
    """

    def __init__(self, nx, layers):
        """
        initialize a DeepNeuralNetwork instance.
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')
        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError('layers must be a list of positive integers')
        if not all(isinstance(layer, int) and layer > 0 for layer in layers):
            raise TypeError('layers must be a list of positive integers')

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for l in range(self.L):
            if l == 0:
                self.weights['W1'] = np.random.randn(layers[l], nx) * np.sqrt(2 / nx)
            else:
                self.weights[f'W{l + 1}'] = np.random.randn(layers[l], layers[l - 1]) * np.sqrt(2 / layers[l - 1])
            self.weights[f'b{l + 1}'] = np.zeros((layers[l], 1))
