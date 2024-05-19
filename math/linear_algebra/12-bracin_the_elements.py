#!/usr/bin/env python3
"""A function that performs element wise add, subtract, multiply and divide"""


def np_elementwise(mat1, mat2):
    """
    add:mat1 + mat2
    add:mat1 - mat2
    add:mat1 * mat2
    add:mat1 / mat2
    """
    return (mat1 + mat2, mat1 - mat2, mat1 * mat2, mat1 / mat2)
