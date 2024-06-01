#!/usr/bin/env python3

"""A function that calculates integral of polynomial"""


def poly_integral(poly, C=0):
    """
    c: integration constant
    if poly or c are not valid: return value is None
    """
    if not isinstance(poly, list) or not all(isinstance(coef, (int, float)) for coef in poly):
        return None
    if not isinstance(C, int):
        return None
    integral = [coef / (i + 1) for i, coef in enumerate(poly)]
    integral.insert(0, C)  # Add the integration constant

    return integral
