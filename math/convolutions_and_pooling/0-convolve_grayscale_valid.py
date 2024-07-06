#!/usr/bin/env python3

"""
A function that performs a valid convolution on grayscale images
"""

import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    Returns:
    A numpy.ndarray containing the convolved images
    """
    # We get the dimensions of images and kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # We calculate  the output of dimensions
    output_h = h - kh + 1
    output_w = w - kw + 1

    # We initialize the output array
    output = np.zeros((m, output_h, output_w))

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw]
                                     * kernel, axis=(1, 2))

    return output
