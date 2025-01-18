#!/usr/bin/env python3


import numpy as np

def rnn(rnn_cell, X, h_0):
    """
    Performs forward propagation for a simple RNN.

    Parameters:
    - rnn_cell: an instance of RNNCell that will be used for the forward propagation
    - X: numpy.ndarray of shape (t, m, i) containing the input data
        * t: maximum number of time steps
        * m: batch size
        * i: dimensionality of the data
    - h_0: numpy.ndarray of shape (m, h) containing the initial hidden state
        * h: dimensionality of the hidden state

    Returns:
    - H: numpy.ndarray of shape (t + 1, m, h) containing all of the hidden states
    - Y: numpy.ndarray of shape (t, m, o) containing all of the outputs
        * o: dimensionality of the outputs
    """
    t, m, _ = X.shape
    h = h_0.shape[1]
    H = np.zeros((t + 1, m, h))
    H[0] = h_0
    outputs = []

    for step in range(t):
        h_next, y = rnn_cell.forward(H[step], X[step])
        H[step + 1] = h_next
        outputs.append(y)

    Y = np.stack(outputs, axis=0)
    return H, Y
