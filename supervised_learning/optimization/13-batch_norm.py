#!/usr/bin/env python3

""" A function that that normalizes an unactivated output
of a neural network using batch normalization:
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """
    args:
    prev is the activated output of the previous layer
    n is the number of nodes in the layer to be created
    returns:
      the updated value for alpha

    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    Z_normalized = (Z - mean) / np.sqrt(variance + epsilon)
    Z_batch_norm = gamma * Z_normalized + beta
    
    return Z_batch_norm
