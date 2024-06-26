#!/usr/bin/env python3
"""a function to add two arrays and return a matrix"""


def add_matrices2D(mat1, mat2):
    """
    first condition: checking whether the length of mat1 != to length of mat2
    second condition: looping through the rows of the matrices
    """
    if len(mat1) != len(mat2):
        return None
    for row1, row2 in zip(mat1, mat2):
        if len(row1) != len(row2):
            return None
    result = []
    for row1, row2 in zip(mat1, mat2):
        new_row = [
            elem1 + elem2
            for elem1, elem2 in zip(row1, row2)
        ]
        result.append(new_row)
    return result


if __name__ == "__main__":
    add_matrices2D = __import__('5-across_the_planes').add_matrices2D
