#!/usr/bin/env python3

""" A function that calculates
that normalizes (standardizes) a matrix
"""


def normalize(X, m, s):
    """
    args:
    X:numpy.ndarray of shape (d, nx) to normalize
    d:number of data points
    nx:number of features
    m:numpy.ndarray of shape (nx,) that contains the mean of all features of X
    s:numpy.ndarray of shape (nx,) that contains the standard deviation of all features of X
    returns:
     The normalized X matrix
    """
    return (X - m) / s