#!/usr/bin/env python3

def determinant(matrix):
    """
    Calculate the determinant of a matrix.

    Params:
    matrix: list of lists to calcuate determinant.
    """

    """
    Checking whether a matrix is a list of lists
    """
    if type(matrix) != list:
        raise TypeError("matrix must be a list of lists")
        
    for row in matrix:
        if type(row) != list:
            raise TypeError("matrix must be a list of lists")
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")
    if matrix == [[]]:
        return 1
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    # Recursive case for larger matrices
    d = 0
    for c in range(n):
        submatrix = [row[:c] + row[c + 1:] for row in matrix[1:]]
        sign = (-1) ** c
        d += sign * matrix[0][c] * determinant(submatrix)
    return d
