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
        
    def pdf(self, x):
        """
        A method to calculates the PDF at a data point.
        """
        if type(x) is not np.ndarray:
            raise TypeError("x must be a numpy.ndarray")
        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError(f"x must have the shape {d}")
        check_d, one = x.shape
        if check_d != d or one != 1:
            raise ValueError(f"x must have the shape {d}")

        det = np.linalg.det(self.cov)
        inv = np.linalg.inv(self.cov)
        pdf = 1.0 / np.sqrt(((2 * np.pi) ** d) * det)
        mult = np.matmul(np.matmul((x - self.mean).T, inv), (x - self.mean))
        pdf *= np.exp(-0.5 * mult)
        pdf = pdf[0][0]
        return pdf