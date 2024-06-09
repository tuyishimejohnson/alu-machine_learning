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


    # Calculate factorial of k
    # Calculate e^-lambtha using Taylor series
    # Calculate PMF
    
    def pmf(self, k):
        k = int(k)
        if k < 0:
            return 0
        else:
            
            factorial = 1
            for i in range(1, k + 1):
                factorial *= i

            
            e_neg_lambtha = 1.0
            for i in range(1, 100):
                e_neg_lambtha += (-self.lambtha) ** i / factorial

            
            pmf = (self.lambtha ** k) * e_neg_lambtha / factorial
            return pmf
        