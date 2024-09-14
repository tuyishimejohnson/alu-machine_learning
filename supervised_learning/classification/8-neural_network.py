#!/usr/bin/env python3

""" A class Neuron that defines a single neuron
     for binary classification """

import numpy as np


class NeuralNetwork:
    """
        Initializes a constructor with nx variable
        Parameters:
        nx: integer
        W: weight of neuron
        A: activated output
        b: bias of neuron
        """
    def __init__(self, nx, nodes):
        self.nx = nx

        if not isinstance(self.nx, int):
            raise TypeError("nx must be an integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.W1 = np.random.randn(nodes, nx)
        self.b1 = np.zeros((nodes, 1))
        self.A1 = np.zeros((nodes, 1))
        self.W2 = np.random.randn(1, nodes)
        self.b2 = 0
        self.A2 = 0
