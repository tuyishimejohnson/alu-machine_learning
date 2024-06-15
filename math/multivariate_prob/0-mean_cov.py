#!/usr/bin/env python3
import numpy as np


def mean_cov(X):
    """
    Calculate the mean and covariance of a data set.

    Parameters:
    X (numpy.ndarray of shape (n, d)): The data set where
        n is the number of data points and
        d is the number of dimensions in each data point.

    Raises:
    TypeError: If X is not a 2D numpy.ndarray.
    ValueError: If X does not contain multiple data points.

    Returns:
    mean (numpy.ndarray of shape (1, d)): The mean of the data set.
    cov (numpy.ndarray of shape (d, d)): The covariance matrix of the data set.
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
