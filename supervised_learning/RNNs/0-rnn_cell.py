#!/usr/bin/env python3

import numpy as np

class RNNCell:
    def __init__(self, i, h, o):
        """
        Constructor for the RNNCell class.

        Parameters:
        - i (int): Dimensionality of the input data.
        - h (int): Dimensionality of the hidden state.
        - o (int): Dimensionality of the outputs.
        """
        self.Wh = np.random.normal(size=(i + h, h))  # Weights for hidden state and input concatenation
        self.Wy = np.random.normal(size=(h, o))      # Weights for the output
        self.bh = np.zeros((1, h))                   # Bias for the hidden state
        self.by = np.zeros((1, o))                   # Bias for the output

    def forward(self, h_prev, x_t):
        """
        Performs forward propagation for one time step.

        Parameters:
        - h_prev (numpy.ndarray): Previous hidden state, shape (m, h).
        - x_t (numpy.ndarray): Input data at time step t, shape (m, i).

        Returns:
        - h_next (numpy.ndarray): Next hidden state, shape (m, h).
        - y (numpy.ndarray): Output of the cell, shape (m, o).
        """
        # Concatenate h_prev and x_t along the last axis
        concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state using tanh activation
        h_next = np.tanh(np.matmul(concat, self.Wh) + self.bh)

        # Compute output using softmax activation
        y_linear = np.matmul(h_next, self.Wy) + self.by
        y = np.exp(y_linear) / np.sum(np.exp(y_linear), axis=1, keepdims=True)

        return h_next, y
