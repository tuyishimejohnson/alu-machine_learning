#!/usr/bin/env python3
"""
Create Confusion matrix
"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Function that creates a confusion matrix

    Arguments:
    m: number of data points
    classes: number of classes
    logits:one-hot numpy.ndarray of shape (m, classes)
    containing the predicted labels

    Returns:
    np.ndarray: Confusion matrix of shape
    (classes, classes) with row indices representing the correct
    labels and column indices representing the predicted labels.
    """
    return np.dot(labels.T, logits)