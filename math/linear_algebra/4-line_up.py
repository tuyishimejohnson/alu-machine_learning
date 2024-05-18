#!/usr/bin/env python3
""" function that returns the surm of two arrays in matrix
    """

def add_arrays(arr1, arr2):
    if len(arr1) != len(arr2):
        return None
    else:
        for (a, b) in zip(arr1, arr2):
            return [a + b]



if __name__ == "__main__":
    add_arrays = __import__('4-line_up').add_arrays
