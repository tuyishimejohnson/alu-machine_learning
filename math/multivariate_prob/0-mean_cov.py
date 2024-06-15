#!/usr/bin/env python3
import numpy as np

"""
A function that calculates the mean and covariance
of a dataset.
"""


def mean_cov(X):
    """
    X: it must be a 2D array
    n: the number of datapoints
    d: the number of dimensions in each datapoint
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