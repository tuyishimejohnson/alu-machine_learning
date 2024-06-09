#!/usr/bin/env python3

"""
Defining a class called Poisson.
"""


class Poisson:
    """
    __init__: used to initialize a method for the class instance.
    lambtha <= 0: It must be a positive value.

    """
    def __init__(self, data=None, lambtha=1.):
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = float(sum(data) / len(data))

    """ A method that calculates the value of
    the PMF for a given number of successes.
    """
    def pmf(self, k):
        """
        Calculates the value of the PMF for a given number of successes
        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        factorial = 1
        for i in range(1, k + 1):
            factorial = factorial * i
        result = self.lambtha ** k * 2.7182818285 ** (-self.lambtha)
        return result / factorial
    