#!/usr/bin/env python3

"""A function that calculates the summation of i squared. """

def summation_i_squared(n):
    """
    n: stopping condition
    return: integer value of the sum
    if n is not a number: None    
    """

    if type(n) != int:
        return None
    
    return (n*(n + 1)*((2*n) + 1))//6
