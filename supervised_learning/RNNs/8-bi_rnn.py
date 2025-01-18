#!/usr/bin/env python3
"""
that performs forward propagation for a bidirectional RNN
"""

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    Performs forward propagation for a bidirectional RNN.

    Parameters:
    - bi_cell: An instance of BidirectionalCell
    used for forward propagation.
    - X: numpy.ndarray of shape (t, m, i), input data for the RNN.
      - t: Maximum number of time steps.
      - m: Batch size.
      - i: Dimensionality of the data.
    - h_0: numpy.ndarray of shape (m, h), initial hidden
    state in the forward direction.
      - h: Dimensionality of the hidden state.
    - h_t: numpy.ndarray of shape (m, h), initial hidden
    state in the backward direction.

    Returns:
    - H: numpy.ndarray of shape (t, m, 2 * h), all concatenated hidden states.
    - Y: numpy.ndarray of shape (t, m, o), all outputs.
    """
    t, m, i = X.shape
    h = h_0.shape[1]

    H_forward = np.zeros((t, m, h))
    H_backward = np.zeros((t, m, h))

    h_prev = h_0
    for time_step in range(t):
        h_prev = bi_cell.forward(h_prev, X[time_step])
        H_forward[time_step] = h_prev

    h_next = h_t
    for time_step in reversed(range(t)):
        h_next = bi_cell.backward(h_next, X[time_step])
        H_backward[time_step] = h_next

    H = np.concatenate((H_forward, H_backward), axis=2)
    Y = bi_cell.output(H)  # Shape (t, m, o)

    return H, Y
