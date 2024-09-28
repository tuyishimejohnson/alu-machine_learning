#!/usr/bin/env python3

""" A function that creates a batch
normalization layer for a neural network in tensorflow:
"""
import tensorflow as tf
import numpy as np


def create_batch_norm_layer(prev, n, activation):
    """
    Creates a batch normalization layer for a neural network in TensorFlow.

    Parameters:
    prev (tf.Tensor): The activated output of the previous layer.
    n (int): The number of nodes in the layer to be created.
    activation (function): The activation
    function that should be used on the output of the layer.

    Returns:
    tf.Tensor: A tensor of the activated output for the layer.
    """
    dense_layer = tf.layers.Dense(
        units=n,
        kernel_initializer=tf.contrib.layers.variance_scaling_initializer(
            mode="FAN_AVG"
        )
    )(prev)

    mean, variance = tf.nn.moments(dense_layer, axes=[0])

    gamma = tf.Variable(tf.ones([n]), trainable=True, name='gamma')
    beta = tf.Variable(tf.zeros([n]), trainable=True, name='beta')

    epsilon = 1e-8
    batch_norm = tf.nn.batch_normalization(
        dense_layer, mean, variance, beta, gamma, epsilon
    )

    return activation(batch_norm)

def model(Data_train, Data_valid,
          layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8,
          decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    Builds, trains, and saves a neural network model in TensorFlow
    using Adam optimization, mini-batch gradient descent,
    learning rate decay, and batch normalization.

    Parameters:
    Data_train (tuple): Training inputs and labels.
    Data_valid (tuple): Validation inputs and labels.
    layers (list): Number of nodes in each layer of the network.
    activations (list): Activation functions used for each layer of the network.
    alpha (float): Learning rate.
    beta1 (float): Weight for the first moment of Adam Optimization.
    beta2 (float): Weight for the second moment of Adam Optimization.
    epsilon (float): Small number used to avoid division by zero.
    decay_rate (float): Decay rate for inverse time decay of the learning rate.
    batch_size (int): Number of data points in a mini-batch.
    epochs (int): Number of times the training
    should pass through the whole dataset.
    save_path (str): Path where the model should be saved to.

    Returns:
    str: The path where the model was saved.
    """
    X_train, Y_train = Data_train
    X_valid, Y_valid = Data_valid
    m, n = X_train.shape

    # Define placeholders
    X = tf.placeholder(tf.float32, shape=(None, n), name='X')
    Y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]), name='Y')
    global_step = tf.Variable(0, trainable=False, name='global_step')

    # Build the neural network
    A = X
    for i in range(len(layers)):
        A = create_batch_norm_layer(A, layers[i], activations[i])

    # Define the loss function
    loss = tf.losses.softmax_cross_entropy(Y, A)

    # Define the learning rate decay
    learning_rate = tf.train.inverse_time_decay(
        alpha, global_step, decay_steps=1, decay_rate=decay_rate, staircase=True
    )

    # Define the optimizer
    optimizer = tf.train.AdamOptimizer(
        learning_rate=learning_rate, beta1=beta1, beta2=beta2, epsilon=epsilon
    )

    # Define the training operation
    train_op = optimizer.minimize(loss, global_step=global_step)

    # Define the accuracy
    correct_prediction = tf.equal(tf.argmax(A, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    # Initialize variables
    init = tf.global_variables_initializer()

    # Create a saver
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs):
            # Shuffle the training data
            permutation = np.random.permutation(m)
            X_train_shuffled = X_train[permutation]
            Y_train_shuffled = Y_train[permutation]

            # Mini-batch gradient descent
            for i in range(0, m, batch_size):
                end = i + batch_size
                X_batch = X_train_shuffled[i:end]
                Y_batch = Y_train_shuffled[i:end]

                sess.run(train_op, feed_dict={X: X_batch, Y: Y_batch})

                if i % 100 == 0:
                    step_cost, step_accuracy = sess.run(
                        [loss, accuracy], feed_dict={X: X_batch, Y: Y_batch}
                    )
                    print(f"\tStep {i // batch_size}:")
                    print(f"\t\tCost: {step_cost}")
                    print(f"\t\tAccuracy: {step_accuracy}")

            # Calculate the cost and accuracy on the entire training set
            train_cost, train_accuracy = sess.run(
                [loss, accuracy], feed_dict={X: X_train, Y: Y_train}
            )
            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy], feed_dict={X: X_valid, Y: Y_valid}
            )

            print(f"After {epoch + 1} epochs:")
            print(f"\tTraining Cost: {train_cost}")
            print(f"\tTraining Accuracy: {train_accuracy}")
            print(f"\tValidation Cost: {valid_cost}")
            print(f"\tValidation Accuracy: {valid_accuracy}")

        # Save the model
        save_path = saver.save(sess, save_path)

    return save_path
