#!/usr/bin/env python3

""" A function that calculates
the normalization (standardization) constants of a matrix
"""
import numpy as np


def normalization_constants(X):
    """
    args:
    X:numpy.ndarray of shape (m, nx) to normalize
    m:number of data points
    nx:number of features

    returns:
     the mean and standard deviation of each feature, respectively
    """
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    return mean, std
