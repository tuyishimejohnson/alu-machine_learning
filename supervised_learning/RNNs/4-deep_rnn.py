#!/usr/bin/env python3
"""
A function to perform forward propagation
"""


import numpy as np

def deep_rnn(rnn_cells, X, h_0):
    """
    Perform forward propagation for a deep RNN.
    Parameters:
    - rnn_cells: List of RNNCell instances
    - X: numpy.ndarray of shape (t, m, i), data input for the RNN.
        - t: Maximum number of time steps.
        - m: Batch size.
        - i: Dimensionality of the data.
    - h_0: numpy.ndarray of shape (l, m, h), initial hidden state.
        - l: Number of layers.
        - h: Dimensionality of the hidden state.

    Returns:
    - H: numpy.ndarray containing all the hidden states.
    - Y: numpy.ndarray containing all the outputs.
    """
    t, m, i = X.shape
    l, _, h = h_0.shape
    H = np.zeros((t + 1, l, m, h))
    Y = []

    # Set the initial hidden states
    H[0] = h_0

    # Iterate over each time step
    for step in range(t):
        x_t = X[step]  # Input at time step t
        h_prev_layer = x_t  # Input to the first layer
        for layer in range(l):
            # Perform forward propagation for the current layer
            h_next, y_t = rnn_cells[layer].forward(H[step, layer], h_prev_layer)
            # Update hidden state for this layer
            H[step + 1, layer] = h_next
            # Pass this layer's hidden state as input to the next layer
            h_prev_layer = h_next

        # Store the output (y_t comes from the last layer)
        Y.append(y_t)

    # Convert Y to a numpy array of shape (t, m, o)
    Y = np.array(Y)

    return H, Y
