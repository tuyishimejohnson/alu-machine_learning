#!/usr/bin/env python3
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """
    concatenate: used to join two matrices
    axis: arrays with more than one dimension. default is 0
    """
    return np.concatenate((mat1, mat2), axis=axis)
