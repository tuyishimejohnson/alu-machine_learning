#!/usr/bin/env python3


"""
Create an Exponential class to represent
exponential distribution.
"""


class Exponential:
    """
    Method for exponential distribution
    if lambtha value is below 0 there is an error
    Change lambtha into a floating number
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
            self.lambtha = 1 / (sum(data) / len(data))

    """
    Calculate the PDF of a given time Period.
    """


    def pdf(self, x):
        """
        x is the time period
        Returns the PDF value for x
        If x is out of range, return 0
        """
        if x < 0:
            return 0
        return self.lambtha * 2.7182818285 ** (-self.lambtha * x)

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given time period
        x = time period
        """
        if x < 0:
            return 0
        return 1 - 2.7182818285 ** (-self.lambtha * x)
