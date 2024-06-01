#!/usr/bin/env python3

"""A function that calculates the summation of i squared. """

def calculate_n(n):
    """
    n: stopping condition
    return: integer value of the sum
    if n is not a number: None    
    """

    if(type(n) != int):
        return None
    
    else:
        return (n*(n + 1)*((2*n) + 1)) // 6
