#!/usr/bin/env python3

"""
A function to calculate the mean and covariance of a dataset
"""


import numpy as np

def mean_cov(X):
    """
    Calculate the mean and covariance of a data set.
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")

    n = X.shape[0]
    mean = np.mean(X, axis=0).reshape(1, -1)

    X = X - mean
    cov = np.dot(X.T, X) / (n - 1)

    return mean, cov
