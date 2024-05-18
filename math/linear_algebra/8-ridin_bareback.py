#!/usr/bin/env python3
"""a function to multiply two matrices and return a new matrix"""
def mat_mul(mat1, mat2):
    """
    condition one: checking whether the length of mat1 equals mat2
    contition two: looping through the matrices to get the length
    """


    if len(mat1[0]) != len(mat2):
        return None
    r = [[0 for _ in range(len(mat2[0]))] for _ in range(len(mat1))]
    for i in range(len(mat1)):
        for j in range(len(mat2[0])):
            for k in range(len(mat2)):
                r[i][j] += mat1[i][k] * mat2[k][j]
    
    return r
if __name__ == "__main__":
    mat_mul = __import__('8-ridin_bareback').mat_mul
