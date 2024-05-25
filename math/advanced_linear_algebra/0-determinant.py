#!/usr/bin/env python3

def determinant(matrix):
    """
    Calculate the determinant of a matrix.

    Parameters:
    matrix: square matrix whose determinant should be calculated.

    Returns:
    int or float: The determinant of the matrix.

    Raises:
    TypeError: If matrix is not a list of lists.
    ValueError: If matrix is not a square matrix.
    """ 
    """
    Checking if a matrix is a list of lists
    """
    if type(matrix) != list:
        raise TypeError("matrix must be a list of lists")
    """
    Check if each element of the matrix is a list
    """
    for row in matrix:
        if type(row) != list:
            raise TypeError("matrix must be a list of lists") 
    """
    special case of a 0x0 matrix
    """
    if matrix == [[]]:
        return 1
    """
    Check if the matrix is square
    """
    n = len(matrix)
    for row in matrix:
        if len(row) != n:
            raise ValueError("matrix must be a square matrix")
    """
    cases for small matrices
    """
    if n == 1:
        return matrix[0][0]
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    """
    Recursive case for larger matrices
    """
    det = 0
    for c in range(n):
        submatrix = [row[:c] + row[c + 1:] for row in matrix[1:]]
        sign = (-1) ** c
        det += sign * matrix[0][c] * determinant(submatrix)

    return det
