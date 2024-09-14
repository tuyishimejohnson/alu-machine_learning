#!/usr/bin/env python3
""" a class for Neural Network
"""

import numpy as np


class NeuralNetwork:
    """ Class that defines a neural network with one hidden layer performing
        binary classification.
    """

    def __init__(self, nx, nodes):
        """ Instantiation function

        Args:
            nx (int): size of the input layer
            nodes (_type_): _description_
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(nodes, int):
            raise TypeError('nodes must be an integer')
        if nodes < 1:
            raise ValueError('nodes must be a positive integer')

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    # getter functions
    @property
    def W1(self):
        """ weights for hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """bias one """
        return self.__b1

    @property
    def A1(self):
        """ activated value one """
        return self.__A1

    @property
    def W2(self):
        """ weights for hidden layer two """
        return self.__W2

    @property
    def b2(self):
        """ bias two """
        return self.__b2

    @property
    def A2(self):
        """ activated value two """
        return self.__A2

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        """
        Z1 = np.dot(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.dot(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return self.__A1, self.__A2