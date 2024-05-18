#!/usr/bin/env python3
"""A function that concatenates two arrays and return a maatrix"""
def cat_matrices2D(m1, m2, axis=0):
    """
    m1: matrix one
    m2: matrix two
    """
    try: return [a+b for a, b in zip(m1, m2)] if axis else m1+m2
    except: return None


if __name__ == "__main__":
    cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D
