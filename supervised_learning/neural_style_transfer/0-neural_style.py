#!/usr/bin/env python3

import tensorflow as tf
import numpy as np

class NST:
    # Public class attributes
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    content_layer = 'block5_conv2'

    def __init__(self, style_image, content_image, alpha=1e4, beta=1):
        """
        Initializes an NST instance

        :param style_image: numpy.ndarray of shape (h, w, 3), style reference image
        :param content_image: numpy.ndarray of shape (h, w, 3), content reference image
        :param alpha: weight for content cost
        :param beta: weight for style cost
        """
        if not isinstance(style_image, np.ndarray) or style_image.ndim != 3 or style_image.shape[2] != 3:
            raise TypeError("style_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(content_image, np.ndarray) or content_image.ndim != 3 or content_image.shape[2] != 3:
            raise TypeError("content_image must be a numpy.ndarray with shape (h, w, 3)")

        if not isinstance(alpha, (int, float)) or alpha < 0:
            raise TypeError("alpha must be a non-negative number")

        if not isinstance(beta, (int, float)) or beta < 0:
            raise TypeError("beta must be a non-negative number")

        # Enable eager execution
        tf.config.run_functions_eagerly(True)

        # Set instance attributes
        self.style_image = self.scale_image(style_image)
        self.content_image = self.scale_image(content_image)
        self.alpha = alpha
        self.beta = beta

    @staticmethod
    def scale_image(image):
        """
        Scales an image so that its pixels values are between 0 and 1 and its largest side is 512 pixels

        :param image: numpy.ndarray of shape (h, w, 3), image to be scaled
        :return: scaled image as a tf.Tensor with shape (1, h_new, w_new, 3)
        """
        if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
            raise TypeError("image must be a numpy.ndarray with shape (h, w, 3)")

        # Normalize pixel values to range [0, 1]
        image = image.astype(np.float32) / 255.0

        # Convert to a TensorFlow tensor
        image = tf.convert_to_tensor(image)

        # Add batch dimension
        image = tf.expand_dims(image, axis=0)

        # Get the original dimensions
        h, w, _ = image.shape[1:]

        # Calculate the new dimensions
        if h > w:
            new_h, new_w = 512, int(512 * w / h)
        else:
            new_h, new_w = int(512 * h / w), 512

        # Resize the image using bicubic interpolation
        image = tf.image.resize(image, (new_h, new_w), method='bicubic')

        return image
