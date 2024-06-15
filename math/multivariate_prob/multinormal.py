#!/usr/bin/env python3

"""
A class that represents a Multivariate Normal distribution
"""
import numpy as np


class MultiNormal:

    """
    n: the number of data points
    d: the number of dimensions in each data point
    TypeError: data must be a 2D numpy.ndarray
    If n < 2, ValueError: data must contain multiple data points
    """
    def __init__(self, data):
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")

        d, n = data.shape
        self.mean = np.mean(data, axis=1).reshape(d, 1)

        deviation = data - self.mean
        self.cov = deviation @ deviation.T / (n - 1)

    """
    A method that is used to calculate the PDF at a datapoint
    """
    def pdf(self, x):
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        d = self.mean.shape[0]
        if x.shape != (d, 1):
            raise ValueError(f"x must have the shape ({d}, 1)")

        cov_inv = np.linalg.inv(self.cov)
        cov_det = np.linalg.det(self.cov)
        denominator = np.sqrt((2 * np.pi) ** d * cov_det)
        exponent = -0.5 * ((x - self.mean).T @ cov_inv @ (x - self.mean))

        return float(np.exp(exponent) / denominator)