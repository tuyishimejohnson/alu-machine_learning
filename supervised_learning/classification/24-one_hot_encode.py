#!/usr/bin/env python3
""" a function to convert a numeric vector into a matrix """

import numpy as np


def one_hot_encode(Y, classes):
    """
    Converts a numeric label vector into a one-hot matrix.
    """
    if not isinstance(Y, np.ndarray) or not isinstance(classes, int):
        return None
    if len(Y.shape) != 1 or classes <= 0:
        return None

    try:
        one_hot = np.zeros((classes, Y.shape[0]))
        one_hot[Y, np.arange(Y.shape[0])] = 1
        return one_hot
    except Exception:
        return None
