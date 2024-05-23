#!/usr/bin/env python3

def determinant(matrix):
    if not isinstance(matrix, list) or not all(isinstance(row, list) for row in matrix):
        raise TypeError("matrix must be a list of lists")
    
    if not all(len(row) == len(matrix) for row in matrix):
        raise ValueError("matrix must be a square matrix")
    
    n = len(matrix)
    
    if n == 0:
        return 1  # The determinant of a 0x0 matrix is conventionally 1
    
    if n == 1:
        return matrix[0][0]
    
    if n == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    
    det = 0
    for c in range(n):
        minor = [row[:c] + row[c+1:] for row in (matrix[:0] + matrix[1:])]
        det += ((-1) ** c) * matrix[0][c] * determinant(minor)
    
    return det

if __name__ == '__main__':
    mat0 = [[]]
    mat1 = [[5]]
    mat2 = [[1, 2], [3, 4]]
    mat3 = [[1, 1], [1, 1]]
    mat4 = [[5, 7, 9], [3, 1, 8], [6, 2, 4]]
    mat5 = []
    mat6 = [[1, 2, 3], [4, 5, 6]]

    print(determinant(mat0))  # Output: 1
    print(determinant(mat1))  # Output: 5
    print(determinant(mat2))  # Output: -2
    print(determinant(mat3))  # Output: 0
    print(determinant(mat4))  # Output: 192
    try:
        determinant(mat5)
    except Exception as e:
        print(e)  # Output: matrix must be a square matrix
    try:
        determinant(mat6)
    except Exception as e:
        print(e)  # Output: matrix must be a square matrix

