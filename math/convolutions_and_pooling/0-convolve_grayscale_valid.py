import numpy as np

def convolve_grayscale_valid(images, kernel):
    """
    Performs a valid convolution on grayscale images.

    Args:
    images -- a numpy.ndarray with shape (m, h, w) containing multiple grayscale images
    kernel -- a numpy.ndarray with shape (kh, kw) containing the kernel for the convolution

    Returns:
    A numpy.ndarray containing the convolved images
    """
    # Get dimensions of images and kernel
    m, h, w = images.shape
    kh, kw = kernel.shape

    # Calculate output dimensions
    output_h = h - kh + 1
    output_w = w - kw + 1

    # Initialize output array
    output = np.zeros((m, output_h, output_w))

    # Perform convolution
    for i in range(output_h):
        for j in range(output_w):
            output[:, i, j] = np.sum(images[:, i:i+kh, j:j+kw] * kernel, axis=(1, 2))
    
    return output
