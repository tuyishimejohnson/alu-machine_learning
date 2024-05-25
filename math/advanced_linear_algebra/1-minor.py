#!/usr/bin/env python3


"""
Function to calculate the minor matrix of a given matrix
"""
def minor(matrix):
    """
    Params:
    matrix: list of lists.

    Returns:
    list of lists: The minor matrix of the given matrix.

    Raises:
    TypeError: If the input is not a list of lists.
    ValueError: If the matrix is not square or is empty.
    """
    if type(matrix) != list:
        raise TypeError("matrix must be a list of lists")
    
    for row in matrix:
        if type(row) != list:
            raise TypeError("matrix must be a list of lists")

    num_rows = len(matrix)
    if num_rows == 0 or len(matrix[0]) == 0:
        raise ValueError("matrix must be a non-empty square matrix")
    
    num_cols = len(matrix[0])
    if num_rows != num_cols:
        raise ValueError("matrix must be a non-empty square matrix")
    
    for row in matrix:
        if len(row) != num_cols:
            raise ValueError("matrix must be a non-empty square matrix")
    
    def get_minor(matrix, i, j):
        """ Helper function to get the minor matrix excluding row i and column j """
        return [
            [matrix[row][col] for col in range(num_cols) if col != j]
            for row in range(num_rows) if row != i
        ]
    
    minor_matrix = []
    for i in range(num_rows):
        minor_row = []
        for j in range(num_cols):
            minor_ij = get_minor(matrix, i, j)
            minor_determinant = determinant(minor_ij)
            minor_row.append(minor_determinant)
        minor_matrix.append(minor_row)
    
    return minor_matrix


"""
Calculating the determinant of a given matrix.
"""
def determinant(matrix):
    """
    Parameters:
    matrix (list of lists): A square matrix.

    Returns:
    int/float: The determinant of the matrix.

    Raises:
    ValueError: If the matrix is not square or is empty.
    """
    
    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]

    det = 0
    for c in range(len(matrix)):
        det += ((-1) ** c) * matrix[0][c] * determinant([
            row[:c] + row[c+1:]
            for row in matrix[1:]
        ])
    
    return det
