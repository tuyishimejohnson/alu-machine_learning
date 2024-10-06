#!/usr/bin/env python3
"""
A function that calculates the
sensitivity for each class in a confusion matrix
"""

import numpy as np


def sensitivity(confusion):
    """
    Calculates the sensitivity for each class in a confusion matrix.

    Parameters:
    confusion (np.ndarray): Confusion matrix of shape
    (classes, classes) where row indices represent the
    correct labels and column indices represent the predicted labels.

    Returns:
    np.ndarray: Sensitivity of each class of shape (classes,).
    """
    true_positives = np.diag(confusion)

    false_negatives = np.sum(confusion, axis=1) - true_positives

    sensitivity = true_positives / (true_positives + false_negatives)

    return sensitivity
