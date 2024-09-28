#!/usr/bin/env python3

""" A function that trains a loaded neural
network model using mini-batch gradient descent
"""

import numpy as np
import tensorflow as tf


def train_mini_batch(X_train, Y_train, X_valid,
                      Y_valid, batch_size=32, epochs=5, 
                      load_path="/tmp/model.ckpt", 
                      save_path="/tmp/model.ckpt"):
    """
    Trains a loaded neural network model using mini-batch gradient descent.
    Returns:
    str: The path where the model was saved.
    """
    shuffle_data = __import__('2-shuffle_data').shuffle_data

    # Load the model
    saver = tf.train.import_meta_graph(load_path + '.meta')
    with tf.Session() as sess:
        saver.restore(sess, load_path)
        
        # Get the necessary tensors and ops from the collection
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        m = X_train.shape[0]
        steps_per_epoch = m // batch_size

        for epoch in range(epochs):
            X_train, Y_train = shuffle_data(X_train, Y_train)
            print(f"After {epoch} epochs:")
            
            # Calculate and print training and validation metrics
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_train, y: Y_train}
            )
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={x: X_valid, y: Y_valid}
            )
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")

            for step in range(steps_per_epoch):
                start = step * batch_size
                end = start + batch_size
                X_batch, Y_batch = X_train[start:end], Y_train[start:end]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if step % 100 == 0 and step != 0:
                    step_cost, step_accuracy = sess.run(
                        [loss, accuracy], feed_dict={x: X_batch, y: Y_batch}
                    )
                    print(f"\tStep {step}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")

            # Handle the last batch if it's smaller than batch_size
            if m % batch_size != 0:
                X_batch, Y_batch = X_train[steps_per_epoch * batch_size:], 
                Y_train[steps_per_epoch * batch_size:]
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

        # Save the model
        save_path = saver.save(sess, save_path)
        return save_path