#!/usr/bin/env python3

"""
A class class LSTMCell that represents an LSTM unit
"""


import numpy as np

class LSTMCell:

    """
    Initialize the LSTMCell.
    """
    def __init__(self, i, h, o):
        """
        Parameters:
        - i: Dimensionality of the input data.
        - h: Dimensionality of the hidden state.
        - o: Dimensionality of the outputs.
        """
        # Forget gate weights and biases
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))

        # Update gate (input gate) weights and biases
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))

        # Intermediate cell state weights and biases
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))

        # Output gate weights and biases
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))

        # Output weights and biases
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, c_prev, x_t):
        """
        Perform forward propagation for one time step.

        Parameters:
        - h_prev: numpy.ndarray of shape (m, h), the previous hidden state.
        - c_prev: numpy.ndarray of shape (m, h), the previous cell state.
        - x_t: numpy.ndarray of shape (m, i), the data input for the cell.

        Returns:
        - h_next: numpy.ndarray of shape (m, h), the next hidden state.
        - c_next: numpy.ndarray of shape (m, h), the next cell state.
        - y: numpy.ndarray of shape (m, o), the output of the cell.
        """
        # Concatenate the previous hidden state and the input
        h_x = np.concatenate((h_prev, x_t), axis=1)

        # Forget gate
        f_t = self.sigmoid(np.dot(h_x, self.Wf) + self.bf)

        # Update gate
        u_t = self.sigmoid(np.dot(h_x, self.Wu) + self.bu)

        # Candidate cell state
        c_tilde = np.tanh(np.dot(h_x, self.Wc) + self.bc)

        # Next cell state
        c_next = f_t * c_prev + u_t * c_tilde

        # Output gate
        o_t = self.sigmoid(np.dot(h_x, self.Wo) + self.bo)

        # Next hidden state
        h_next = o_t * np.tanh(c_next)

        # Compute the output
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y

    @staticmethod
    def sigmoid(x):
        """
        Compute the sigmoid activation function.
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """
        Compute the softmax activation function.
        """
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / e_x.sum(axis=1, keepdims=True)
