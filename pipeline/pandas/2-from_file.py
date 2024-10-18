#!/usr/bin/env python3

"""
a function that loads data from a file as a pd.DataFrame
"""

import pandas as pd

def from_file(filename, delimiter):
    """
    filename: file to load from
    delimiter: column separator

    Returns:
    the loaded pd.DataFrame
    """
    return pd.read_csv(filename, delimiter=delimiter)
