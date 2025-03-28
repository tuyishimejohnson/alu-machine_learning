#!/usr/bin/env python3

""" A function that creates a learning rate decay
operation in tensorflow using inverse time decay
"""
import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    args:
    alpha:original learning rate
    decay_rate:weight used to determine the rate at which alpha will decay
    global_step:number of passes of gradient descent that have elapsed
    decay_step:number of passes of gradient descent that
    should occur before alpha is decayed further
    the learning rate decay should occur in a stepwise fashion
    returns:
      the updated value for alpha

    """
    learning_rate = tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return learning_rate
