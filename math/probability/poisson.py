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
    
    def pmf(self, k):
        exponential = math.exp(-self.lambtha)
        k = int(k)
        if k < 0:
            return 0
        else:
            return (self.lambtha**k)* exponential/ math.factorial(k)
        