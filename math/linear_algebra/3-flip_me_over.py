#!/usr/bin/env python3
def matrix_transpose(matrix):

    """
    Returns the transpose of a 2D matrix.
    matrix (list of list of int): The matrix to transpose.
    list of list of int: The transposed matrix.
    """
    return [[matrix[row][col] for row in range(len(matrix))] for col in range(len(matrix[0]))]


if __name__ == "__main__":
	matrix_transpose = __import__('3-flip_me_over').matrix_transpose
