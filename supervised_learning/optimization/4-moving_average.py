#!/usr/bin/env python3

""" A function that calculates
that normalizes (standardizes) a matrix
"""


def moving_average(data, beta):
    """
    args:
    data:list of data to calculate the moving average of.
    beta:weight used for the moving average.
    returns:
     A list containing the moving averages of data.
    """
    moving_averages = []
    v = 0
    for t in range(len(data)):
        v = beta * v + (1 - beta) * data[t]
        corrected_v = v / (1 - beta ** (t + 1))
        moving_averages.append(corrected_v)
    return moving_averages
