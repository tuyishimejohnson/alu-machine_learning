#!/usr/bin/env python3

""" A class Neuron that defines a single neuron
     for binary classification """

import numpy as np


class Neuron:
    """
        Initializes a constructor with nx variable
        to make private instance attributes
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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ Private getter for weight  """
        return self.__W

    @property
    def b(self):
        """ private getter for bias """
        return self.__b

    @property
    def A(self):
        """ private getter for activation output """
        return self.__A