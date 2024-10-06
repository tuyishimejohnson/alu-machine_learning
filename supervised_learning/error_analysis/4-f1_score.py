#!/usr/bin/env python3
"""
A function that that calculates the F1
score of a confusion matrix
"""

sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    confusion: confusion numpy.ndarray of shape (classes, classes)
    where row indices represent the correct labels and column indices
    represent the predicted labels
    classes: number of classes

    Returns:
    a numpy.ndarray of shape (classes,) containing
    the F1 score of each class
    Use: You must use sensitivity = __import__('1-sensitivity').sensitivity
    and precision = __import__('2-precision').precision
    """
    recall = sensitivity(confusion)

    prec = precision(confusion)

    return 2 * (prec * recall) / (prec + recall)
