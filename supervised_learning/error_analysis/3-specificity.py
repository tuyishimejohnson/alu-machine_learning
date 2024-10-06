#!/usr/bin/env python3
"""
A function that that calculates the specificity for each class in a confusion matrix:
"""

import numpy as np


def specificity(confusion):
    """
    confusion: confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    classes: number of classes

    Returns:
    a numpy.ndarray of shape (classes,) containing the specificity of each class
    """
    true_pos = np.diag(confusion)

    false_neg = np.sum(confusion, axis=1) - true_pos

    false_pos = np.sum(confusion, axis=0) - true_pos

    true_neg = np.sum(confusion) - (true_pos + false_neg + false_pos)

    return true_neg / (true_neg + false_pos)