#!/usr/bin/env python3

def matrix_transpose(matrix):
    """ function that returns the transpose of 2d matrix
    """
    transposed = [[matrix[row][col] for row in range(len(matrix))]
                  for col in range(len(matrix[0]))]
    return transposed


if __name__ == "__main__":
    matrix_transpose = __import__('3-flip_me_over').matrix_transpose