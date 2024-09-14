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
        Raises TypeError
        Raises ValueError
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
        """ private getter for weight  """
        return self.__W

    @property
    def b(self):
        """ private getter for bias """
        return self.__b

    @property
    def A(self):
        """ private getter for activation output """
        return self.__A

    def forward_prop(self, X):
        """ A function to calculate the forward propagation """
        z = np.dot(self.W, X) + self.__b
        self.__A = 1/(1 + np.exp(-z))
        return self.__A

    def cost(self, Y, A):

        m = Y.shape[1]  # Number of examples

        # Compute the cross-entropy loss for each example, ensuring numerical stability
        # by using a small epsilon value to avoid log(0).
        epsilon = 1e-7  # Small value to prevent division by zero in the log function
        cost = -np.sum(Y * np.log(A + epsilon) + (1 - Y) * np.log(1 - A + epsilon)) / m

        return cost
