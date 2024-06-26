#!/usr/bin/env python3

"""A function that calculates a derivative of polynomial"""


def poly_derivative(poly):

    """
    try: checking the inputs
    check1: checking whether poly is in a list
    check2: checking if coef is a number
    next step: calculating the derivative
    """

    if not isinstance(poly, list) or not poly:
        return None
    for coefficient in poly:
        if not isinstance(coefficient, (int, float)):
            return None
    if len(poly) == 1:
        return [0]
    derivative = [
        coefficient * power
        for power, coefficient in enumerate(poly)
    ][1:]

    return derivative
