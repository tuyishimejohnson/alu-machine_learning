#!/usr/bin/env python3


"""
A class that represents binomial distribution.
"""


class Binomial:
    """
    data is a list of the data to be used to estimate the distribution
    n is the number of Bernoulli trials
    p is the probability of a “success”
    """

    def __init__(self, data=None, n=1, p=0.5):
        if data is not None:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            mean = sum(data) / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            p = 1 - (variance / mean)
            n = round(mean / p)
            p = mean / n

            self.n = n
            self.p = float(p)
        else:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if not (0 < p < 1):
                raise ValueError("p must be greater than 0 and less than 1")

            self.n = int(n)
            self.p = float(p)

    def pmf(self, k):
        """Calculates the value of the PMF for a given number of successes

        k = number of “successes”
        If k is not an integer, convert it to an integer
        If k is out of range, return 0
        Returns the PMF value for k

        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        factorial_k = 1
        for i in range(1, k + 1):
            factorial_k *= i
        factorial_n = 1
        for i in range(1, self.n + 1):
            factorial_n *= i
        factorial_n_k = 1
        for i in range(1, self.n - k + 1):
            factorial_n_k *= i
        coefficient = factorial_n / (factorial_k * factorial_n_k)
        return coefficient * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """Calculates the value of the CDF for a given number of successes

        Calculates the value of the CDF for a given number of “successes”
        k = number of “successes”
        If k is not an integer, convert it to an integer
        If k is out of range, return 0
        Returns the CDF value for k

        """
        if not isinstance(k, int):
            k = int(k)
        if k < 0:
            return 0
        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)
        return cdf
