#!/usr/bin/env python3
import numpy as np
"""A function that concatenates two matrices along with a specific axis"""


def np_cat(mat1, mat2, axis=0):
    """
    np: numpy module imported as np
    concatenate: used to join two matrics
    axis: arrays with more than one dimension
    """
    return np.concatenate((mat1, mat2), axis=axis)
