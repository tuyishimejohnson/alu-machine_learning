#!/usr/bin/env python3

"""
A class GRUCell that represents a gated recurrent unit
"""

import numpy as np


class GRUCell:
    """
    Creates a constructor function.
    """
    def __init__(self, i, h, o):
        """
        Initialize the GRUCell instance.

        Parameters:
        - i: Dimensionality of the input data.
        - h: Dimensionality of the hidden state.
        - o: Dimensionality of the outputs.
        """
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Perform forward propagation for one time step.

        Parameters:
        - h_prev: numpy.ndarray of shape (m, h)
        - x_t: numpy.ndarray of shape (m, i)

        Returns:
        - h_next: numpy.ndarray, the next hidden state.
        - y: numpy.ndarray, the output of the cell.
        """
        m, _ = x_t.shape

        h_x = np.concatenate((h_prev, x_t), axis=1)

        z_t = self.sigmoid(np.dot(h_x, self.Wz) + self.bz)
        r_t = self.sigmoid(np.dot(h_x, self.Wr) + self.br)
        r_h_prev = r_t * h_prev
        r_h_x = np.concatenate((r_h_prev, x_t), axis=1)
        h_hat = np.tanh(np.dot(r_h_x, self.Wh) + self.bh)
        h_next = (1 - z_t) * h_prev + z_t * h_hat
        y_linear = np.dot(h_next, self.Wy) + self.by
        y = self.softmax(y_linear)

        return h_next, y

    @staticmethod
    def sigmoid(x):
        """Compute the sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """Compute the softmax activation function."""
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / exp_x.sum(axis=1, keepdims=True)
