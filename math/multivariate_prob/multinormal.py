#!/usr/bin/env python3
"""
A class that represents a Multivariate Normal distribution.
"""


import numpy as np


class MultiNormal:
    """
    if x not np.ndarray: Typerrror, x must be a np.ndarray
    if x not shape of (d, 1): valueError, x must have the shape ({d}, 1)
    """
    def __init__(self, data):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        self.mean = np.mean(data, axis=1, keepdims=True)
        diff = data - self.mean
        self.cov = np.dot(diff, diff.T) / (data.shape[1] - 1)

    def pdf(self, x):
        """
        a method to calculate the PDF at a data point.
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError("x must have the shape ({}, 1)".format(d))
        check_d, one = x.shape
        if check_d != d or one != 1:
            raise ValueError("x must have the shape ({}, 1)".format(d))

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        mult = np.matmul(np.matmul((x - self.mean).T, inv), (x - self.mean))
        pdf *= np.exp(-0.5 * mult)
        pdf = pdf[0][0]
        return pdf
