#!/usr/bin/env python3

""" A class Neuron that defines a single neuron
     for binary classification """

import numpy as np


class Neuron:
    """
        Initializes a constructor with nx variable
        Parameters:
        nx: integer
        W: weight of neuron
        A: activated output
        b: bias of neuron
        """
    def __init__(self, nx):
        self.nx = nx

        if not isinstance(self.nx, int):
            raise TypeError("nx must be an integer")
        if self.nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.randn(1, nx)
        self.b = 0
        self.A = 0
