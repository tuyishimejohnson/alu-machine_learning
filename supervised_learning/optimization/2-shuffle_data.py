#!/usr/bin/env python3

""" A function that calculates
that normalizes (standardizes) a matrix
"""

import numpy as np


def shuffle_data(X, Y):
    """
    args:
    X:first numpy.ndarray of shape (m, nx) to shuffle
    m:number of data points
    nx:number of features in X
    Y:second numpy.ndarray of shape (m, ny) to shuffle
    m:same number of data points as in X
    ny:number of features in Y
    returns:
      the shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])
    return X[permutation], Y[permutation]
