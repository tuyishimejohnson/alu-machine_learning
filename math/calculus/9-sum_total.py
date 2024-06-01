#!/usr/bin/env python3

"""A function that calculates the summation of i squared. """

def summation_i_squared(n):
    """
    n: stopping condition
    return: integer value of the sum
    if n is not a number: None
    try: checking whether n is an integer 
    """
    try:
        integer = int(n)
        if integer <= 0:
            return None
    except(ValueError):

        return None
    return (n*(n + 1)*((2*n) + 1))//6
