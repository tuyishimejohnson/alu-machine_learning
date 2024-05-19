#!/usr/bin/env python3
'''a function that returns the shape of the matrix,using while loop'''


def matrix_shape(matrix):
    """
    matrix: a parameter that holds a matrix
    shape: creating an empty array and use is instance method
    """
    shape = []
    while isinstance(matrix, list):
        shape.append(len(matrix))
        matrix = matrix[0] if matrix else None
    return shape


if __name__ == "__main__":
    matrix_shape_function = __import__('2-size_me_please').matrix_shape
