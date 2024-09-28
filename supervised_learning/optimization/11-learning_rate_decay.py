#!/usr/bin/env python3

""" A function that updates the learning rate using inverse time decay in numpy
"""



def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    args:
    alpha:original learning rate
    decay_rate:weight used to determine the rate at which alpha will decay
    global_step:number of passes of gradient descent that have elapsed
    decay_step:number of passes of gradient descent
      that should occur before alpha is decayed further
    returns:
      the updated value for alpha

    """
    decay_factor = 1 / (1 + decay_rate * (global_step // decay_step))
    updated_alpha = alpha * decay_factor
    return updated_alpha