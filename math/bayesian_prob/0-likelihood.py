#!/usr/bin/env python3

"""
This function is used to calculate the likelihood
of obtaining data.
"""

import numpy as np


def factorial(n):
    """Calculate the factorial of n (n!)"""
    if n == 0 or n == 1:
        return 1
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result


def binomial_coefficient(n, x):
    """Calculate the binomial coefficient (n choose x)"""
    return factorial(n) / (factorial(x) * factorial(n - x))


def likelihood(x, n, P):
    """Calculate the likelihood of the given data"""
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that"
                         " is greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    # Calculate the binomial coefficient
    binom_coeff = binomial_coefficient(n, x)

    # Calculate the likelihood for each probability in P
    likelihoods = binom_coeff * (P ** x) * ((1 - P) ** (n - x))

    return likelihoods
