#!/usr/bin/env python3
"""a function to multiply two matrices and return a new matrix"""
def mat_mul(mat1, mat2):
    """
    condition one: checking whether the length of mat1 equals mat2
    contition two: looping through the rows using also zip method
    """
    
    if len(mat1) != len(mat2):
        return None
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None

    
    result = []
    for row1, row2 in zip(mat1, mat2):
        new_row = [
            elem1 * elem2 
            for elem1, elem2 in zip(row1, row2)
        ]
        result.append(new_row)
    
    return result


if __name__ == "__main__":
    mat_mul = __import__('8-ridin_bareback').mat_mul
