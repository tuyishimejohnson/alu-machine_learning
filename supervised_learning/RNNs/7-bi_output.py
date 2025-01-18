#!/usr/bin/env python3

"""
BidirectionalCell that represents
a bidirectional cell of an RNN
"""

import numpy as np


class BidirectionalCell:
    """
    Represents a bidirectional cell of an RNN.

    Attributes:
    - Whf: Weight matrix for the forward hidden states.
    - bhf: Bias vector for the forward hidden states.
    - Whb: Weight matrix for the backward hidden states.
    - bhb: Bias vector for the backward hidden states.
    - Wy: Weight matrix for the outputs.
    - by: Bias vector for the outputs.
    """

    def __init__(self, i, h, o):
        """
        Class constructor.

        Parameters:
        - i: Dimensionality of the data.
        - h: Dimensionality of the hidden states.
        - o: Dimensionality of the outputs.
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.bhf = np.zeros((1, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.bhb = np.zeros((1, h))
        self.Wy = np.random.normal(size=(2 * h, o))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        Calculates the hidden state in the forward direction for one time step.

        Parameters:
        - h_prev: numpy.ndarray of shape (m, h), previous hidden state.
        - x_t: numpy.ndarray of shape (m, i), data input for the cell.

        Returns:
        - h_next: numpy.ndarray of shape (m, h), the next hidden state.
        """
        concatenated = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(concatenated, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward
        direction for one time step.

        Parameters:
        - h_next: numpy.ndarray of shape (m, h), next hidden state.
        - x_t: numpy.ndarray of shape (m, i), data input for the cell.

        Returns:
        - h_prev: numpy.ndarray of shape (m, h), the previous hidden state.
        """
        concatenated = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(concatenated, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """
        Calculates all outputs for the RNN.

        Parameters:
        - H: numpy.ndarray of shape (t, m, 2 * h),
        concatenated hidden states from both directions.

        Returns:
        - Y: numpy.ndarray of shape (t, m, o), the outputs.
        """
        t, m, _ = H.shape
        Y = np.zeros((t, m, self.Wy.shape[1]))
        for time_step in range(t):
            h_t = H[time_step]
            y_t = np.matmul(h_t, self.Wy) + self.by
            Y[time_step] = np.exp(y_t) / np.exp(y_t).sum(axis=1, keepdims=True)

        return Y
