#!/usr/bin/env python3
"""
A function def from_numpy(array): that creates a pd.DataFrame
from a np.ndarray
"""

import pandas as pd
import numpy as np

def from_numpy(array):
    """
    array: np.ndarray from which you should create the pd.DataFrame
    The columns of the pd.DataFrame should be labeled in alphabetical order and
    capitalized. There will not be more than 26 columns.

    Returns:
    the newly created pd.DataFrame
    """

    columns = []
    for i in range(array.shape[1]):
        columns.append(chr(65 + i))

    return pd.DataFrame(array, columns=columns)
    

A = np.random.randn(5, 8)
print(from_numpy(A))