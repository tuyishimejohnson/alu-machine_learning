#!/usr/bin/env python3

import numpy as np

def pool(images, kernel_shape, stride, mode='max'):
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride
    
    output_h = (h - kh) // sh + 1
    output_w = (w - kw) // sw + 1
    
    pooled = np.zeros((m, output_h, output_w, c))
    
    for i in range(output_h):
        for j in range(output_w):
            x_start = i * sh
            x_end = x_start + kh
            y_start = j * sw
            y_end = y_start + kw
            
            if mode == 'max':
                pooled[:, i, j, :] = np.max(images[:, x_start:x_end, y_start:y_end, :], 
                                            axis=(1, 2))
            elif mode == 'avg':
                pooled[:, i, j, :] = np.mean(images[:, x_start:x_end, y_start:y_end, :], 
                                             axis=(1, 2))
                
    return pooled
