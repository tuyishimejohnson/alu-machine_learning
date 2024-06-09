#!/usr/bin/env python3


"""
A class that represents a normal distribution.
"""


class Normal:

    """
    Calculate the normal distribution
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data) / len(data)
            self.stddev = (sum((x-self.mean)**2 for x in data)/len(data))**0.5

    def z_score(self, x):
        """
        calculates z-score of a given x-value
        x is the value
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculates the z-score of a given x-value
        x is the x-value
        Returns the z-score of x

        Calculates the x-value of a given z-score
        z is the z-score
        Returns the x-value of z
        """
        return self.mean + z * self.stddev

    def pdf(self, x):
        """
        Calculates the value of the PDF for a given x-value
        x is the x-value
        Returns the PDF value for x
        """
        exponent = -((x - self.mean) ** 2) / (2 * (self.stddev ** 2))
        pdf_value = (1 / (self.stddev * ((2 * self.PI) ** 0.5))) * \
            (self.E ** exponent)
        return pdf_value

    def cdf(self, x):
        """
        Calculates the value of the CDF for a given x-value
        x is the x-value
        Returns the CDF value for x
        """
        erf = (x - self.mean) / (self.stddev * (2 ** 0.5))
        cdf = 0.5 * (1 + (self.erf(erf)))
        return cdf
