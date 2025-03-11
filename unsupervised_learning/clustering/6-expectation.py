#!/usr/bin/env python3
"""
Module for calculating the expectation step in the EM algorithm for a GMM
"""

import numpy as np


def expectation(X, pi, m, S):
    """
    Calculates the expectation step in the EM algorithm for a GMM
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None

    n, d = X.shape
    k = pi.shape[0]

    if m.shape[0] != k or m.shape[1] != d:
        return None, None
    if S.shape[0] != k or S.shape[1] != d or S.shape[2] != d:
        return None, None
    if not np.isclose(np.sum(pi), 1):
        return None, None

    pdf = __import__('5-pdf').pdf

    g = np.zeros((k, n))
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    g_sum = np.sum(g, axis=0)
    g /= g_sum

    log_likelihood = np.sum(np.log(g_sum))

    return g, log_likelihood
