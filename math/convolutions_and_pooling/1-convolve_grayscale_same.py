#!/usr/bin/env python3

import numpy as np

def convolve_grayscale_same(images, kernel):
    """
    Performs a same convolution on grayscale images.

    Args:
    images -- a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
    kernel -- a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution

    Returns:
    A numpy.ndarray containing the convolved images
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate padding
    pad_h = kh // 2
    pad_w = kw // 2
    pad = ((0, 0), (pad_h, pad_h), (pad_w, pad_w))

    # Pad images with zeros
    padded_images = np.pad(images, pad, mode='constant')

    # Initialize output array
    output = np.zeros((m, h, w))

    # Perform convolution
    for i in range(h):
        for j in range(w):
            for img in range(m):
                output[img, i, j] = np.sum(padded_images[img, i:i+kh, j:j+kw] * kernel)

    return output
