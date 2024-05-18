#!/usr/bin/env python3
"""A function that concatenates two arrays and return a maatrix"""
def cat_matrices2D(mat1, mat2, axis=0):
    """
    m1: matrix one
    m2: matrix two
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
    
        return mat1 + mat2
    elif axis == 1:
        
        if len(mat1) != len(mat2):
            return None
    
        return [row1 + row2 for row1, row2 in zip(mat1, mat2)]
    else:
        return None
    
if __name__ == "__main__":
    cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D

