#!/usr/bin/env python3
"""
A function that calculates the
precision for each class in a confusion matrix
"""

import numpy as np


def precision(confusion):
    """
    confusion: confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    classes: number of classes

    Returns:
    a numpy.ndarray of shape (classes,) containing the precision of each class
    """
    true_positives = np.diag(confusion)

    false_positives = np.sum(confusion, axis=0) - true_positives

    precision = true_positives / (true_positives + false_positives)

    return precision
