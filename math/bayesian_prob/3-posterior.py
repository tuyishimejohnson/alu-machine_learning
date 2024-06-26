#!/usr/bin/env python3

"""
A function to calculate the posterior probability for the various hypothetical
probabilities of developing severe side effects given the data
"""


import numpy as np
likelihood = __import__("0-likelihood").likelihood
intersection = __import__("1-intersection").intersection
marginal = __import__("2-marginal").marginal


def posterior(x, n, P, Pr):
    """
    Params:
    x: number of patients with side effects
    n: total number of patients
    p: 1D numpy.ndarray
    Pr: 1D numpy.ndarray with beliefs of P
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")
    if not isinstance(x, int) or x < 0:
        raise ValueError("x must be an integer that is"
                         " greater than or equal to 0")
    if x > n:
        raise ValueError("x cannot be greater than n")
    if not isinstance(P, np.ndarray) or P.ndim != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    intersections = intersection(x, n, P, Pr)
    marginal_prob = marginal(x, n, P, Pr)

    posterior_prob = intersections / marginal_prob
    return posterior_prob
