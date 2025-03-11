#!/usr/bin/env python3
"""Initialization of cluster centroids for K-means"""

import numpy as np


def initialize(X, k):
    """Function that initializes cluster centroids for K-means

    X is a numpy.ndarray of shape (n, d) containing the
        dataset that will be used for K-means clustering
    """

    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(k) is not int or k <= 0:
        return None
    n, d = X.shape
    minimum = np.min(X, axis=0)
    maximum = np.max(X, axis=0)
    centroid = np.random.uniform(minimum, maximum, size=(k, d))
    return centroid
