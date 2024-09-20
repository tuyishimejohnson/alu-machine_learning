#!/usr/bin/env python3

import tensorflow as tf


def create_placeholders(nx, classes):
    x = tf.placeholder("float", [None, 3])
    y = x * nx

    with tf.Session() as session:
        x_data = classes
    return session.run(y, feed_dict={x: x_data})


    
