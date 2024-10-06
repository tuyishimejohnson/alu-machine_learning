#!/usr/bin/env python3

import numpy as np

"""
    Creates a confusion matrix.

    params:
    labels (np.ndarray): One-hot numpy.ndarray of shape (m, classes)
    containing the correct labels for each data point.
    logits (np.ndarray): One-hot numpy.ndarray of shape (m, classes)
    containing the predicted labels.
    m: number of data points
    Returns:
    np.ndarray: Confusion matrix of shape (classes, classes) with row
    indices representing the correct labels and column indices
    representing the predicted labels.
    """

def create_confusion_matrix(labels, logits):
    true_labels = np.argmax(labels, axis=1)
    predicted_labels = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion_matrix = np.zeros((classes, classes), dtype=int)

    for true, pred in zip(true_labels, predicted_labels):
        confusion_matrix[true, pred] += 1

    return confusion_matrix.astype(float)