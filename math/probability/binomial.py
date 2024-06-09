#!/usr/bin/env python3


""" 
A class that represents binomial distribution.
"""
class Binomial:
    def __init__(self, data=None, n=1, p=0.5):
        """ 
        data is a list of the data to be used to estimate the distribution
        n is the number of Bernoulli trials
        p is the probability of a “success”
        Sets the instance attributes n and p
        """
        if data is None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError("n must be a positive value")
            if not isinstance(p, (float, int)) or not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = n
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)
            
            p_initial = 1 - (variance / mean)
            n_initial = round(mean / p_initial)
            
            self.n = n_initial
            self.p = mean / self.n

    def __str__(self):
        return f'n: {self.n} p: {self.p:.3f}'
