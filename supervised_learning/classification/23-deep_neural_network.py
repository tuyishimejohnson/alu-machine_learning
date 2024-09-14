#!/usr/bin/env python3
""" Deep Neural Network
"""

import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """ Class that defines a deep neural network performing binary
        classification.
    """

    def __init__(self, nx, layers):
        """ Instantiation function

        Args:
            nx (int): number of input features
            layers (list): representing the number of nodes in each layer of
                           the network
        """
        if not isinstance(nx, int):
            raise TypeError('nx must be an integer')
        if nx < 1:
            raise ValueError('nx must be a positive integer')

        if not isinstance(layers, list):
            raise TypeError('layers must be a list of positive integers')
        if len(layers) < 1:
            raise TypeError('layers must be a list of positive integers')

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
            if not isinstance(layers[i], int) or layers[i] < 1:
                raise TypeError('layers must be a list of positive integers')

            if i == 0:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights['W' + str(i + 1)] = np.random.randn(
                    layers[i], layers[i - 1]) * np.sqrt(2 / layers[i - 1])

            self.__weights['b' + str(i + 1)] = np.zeros((layers[i], 1))

    @property
    def L(self):
        """ Return layers in the neural network"""
        return self.__L

    @property
    def cache(self):
        """ Return dictionary with intermediate values of the network"""
        return self.__cache

    @property
    def weights(self):
        """Return weights and bias dictionary"""
        return self.__weights

    def forward_prop(self, X):
        """
        Calculates the forward propagation of the neural network.
        """
        self.__cache['A0'] = X
        for i in range(self.__L):
            Wi = self.__weights['W' + str(i + 1)]
            bi = self.__weights['b' + str(i + 1)]
            Ai = self.__cache['A' + str(i)]
            Zi = np.dot(Wi, Ai) + bi
            self.__cache['A' + str(i + 1)] = 1 / (1 + np.exp(-Zi))
        return self.__cache['A' + str(self.__L)], self.__cache

    def cost(self, Y, A):
        """
        calculates the cost of the model using logistic regression.
        """
        m = Y.shape[1]
        cost = -np.mean(Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        return cost

    def evaluate(self, X, Y):
        """
        Evaluates the neural network’s predictions.
        """
        A, _ = self.forward_prop(X)
        cost = self.cost(Y, A)
        prediction = np.where(A >= 0.5, 1, 0)
        return prediction, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """
        Calculates one pass of gradient descent on the neural network.
        updates:
        __weights (dict): The weights and biases of the network.
        """
        m = Y.shape[1]
        L = self.__L
        A_prev = cache['A' + str(L - 1)]
        A = cache['A' + str(L)]
        dZ = A - Y

        for i in reversed(range(L)):
            A_prev = cache['A' + str(i)]
            W = self.__weights['W' + str(i + 1)]
            b = self.__weights['b' + str(i + 1)]

            dW = np.dot(dZ, A_prev.T) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            if i > 0:
                dZ = np.dot(W.T, dZ) * (A_prev * (1 - A_prev))

            self.__weights['W' + str(i + 1)] -= alpha * dW
            self.__weights['b' + str(i + 1)] -= alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """
        Trains the deep neural network.
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        costs = []

        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, cache, alpha)
            if verbose and i % step == 0:
                cost = self.cost(Y, A)
                print(f"Cost after {i} iterations: {cost}")
                costs.append(cost)

        if graph:
            plt.plot(range(0, iterations, step), costs, 'b-')
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)
