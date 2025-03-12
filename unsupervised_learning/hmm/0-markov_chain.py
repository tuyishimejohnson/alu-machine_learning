#!/usr/bin/env python3
'''
A function that determines the probability of a markov chain
'''


import numpy as np


def markov_chain(P, s, t=1):
    '''
        Determines the probability of a markov chain
    '''
    # check that P is the correct type and dimensions
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    # save value of n and check that P is square
    n, n_check = P.shape
    if n != n_check:
        return None
    # check that s is the correct type and dimensions
    if type(s) is not np.ndarray or len(s.shape) != 2:
        return None
    # check that the shape of s matches (1, n)
    one, n_check = s.shape
    if one != 1 or n_check != n:
        return None
    # check that t is the correct type and is non-negative
    if type(t) is not int or t < 0:
        return None
    result = s
    for i in range(t):
        result = np.matmul(result, P)
    return result
