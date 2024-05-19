#!/usr/bin/env python3
"""A function that multiplies two matrices"""
import numpy as np

def np_matmul(mat1, mat2):
    """
    Parameters:
    mat1: first matrix.
    mat2: second matrix.
    """
    return np.dot(mat1, mat2)
