#!/usr/bin/env python3

"""A function that calculates a derivative of polynomial"""


def poly_derivative(poly):

    """
    try: checking the inputs
    check1: checking whether poly is in a list
    check2: checking if coef is a number
    next step: calculating the derivative
    """
    
    try:
        if type(poly) is not list:
            return None
        for coef in poly:
            coef + 0
    except TypeError:
        return None
    if len(poly) == 1:
        return [0]

    derivative = [i * poly[i] for i in range(1, len(poly))]

    
    if all(coef == 0 for coef in derivative):
        return [0]

    return derivative
