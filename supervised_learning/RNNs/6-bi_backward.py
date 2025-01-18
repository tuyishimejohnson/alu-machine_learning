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
        # Concatenate h_prev and x_t
        concatenated = np.concatenate((h_prev, x_t), axis=1)
        # Compute the next hidden state
        h_next = np.tanh(np.matmul(concatenated, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        Calculates the hidden state in the backward direction for one time step.

        Parameters:
        - h_next: numpy.ndarray of shape (m, h), next hidden state.
        - x_t: numpy.ndarray of shape (m, i), data input for the cell.

        Returns:
        - h_prev: numpy.ndarray of shape (m, h), the previous hidden state.
        """
        # Concatenate h_next and x_t
        concatenated = np.concatenate((h_next, x_t), axis=1)
        # Compute the previous hidden state
        h_prev = np.tanh(np.matmul(concatenated, self.Whb) + self.bhb)
        return h_prev
