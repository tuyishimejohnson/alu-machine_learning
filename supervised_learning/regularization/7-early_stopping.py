#!/usr/bin/env python3

""" a function that creates a layer of a neural network using dropout
"""


def early_stopping(cost, opt_cost, threshold, patience, count):
    """
    args:
    cost:current validation cost of the neural network
    opt_cost:lowest recorded validation cost of the neural network
    threshold:threshold used for early stopping
    patience:patience count used for early stopping
    count:count of how long the threshold has not been met
    returns:
    a boolean of whether the network should be
    stopped early, followed by the updated count
    """

    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    if count >= patience:
        return True, count
    return False, count
