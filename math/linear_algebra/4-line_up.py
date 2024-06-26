#!/usr/bin/env python3
"""A function that returns the sum of two arrays in matrix"""


def add_arrays(arr1, arr2):
    """
    first condition: checking if length of arr1 != to the length of arr2
    second condition: add arr1 and arr2 with a zip
    """
    if len(arr1) != len(arr2):
        return None
    else:
        return [a + b for a, b in zip(arr1, arr2)]


if __name__ == "__main__":
    add_arrays = __import__('4-line_up').add_arrays
