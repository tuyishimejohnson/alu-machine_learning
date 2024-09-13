#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

Neuron = __import__('0-neuron').Neuron

class Neuron: 
    def __init__(self, nx):
        self.nx = nx

        if not isinstance(self.nx, int):
            raise TypeError("nx must be an integer")
        elif self.nx < 1:
            raise ValueError("nx must be a positive integer")


lib_train = np.load('../data/Binary_Train.npz')
X_3D, Y = lib_train['X'], lib_train['Y']
X = X_3D.reshape((X_3D.shape[0], -1)).T

np.random.seed(0)
neuron = Neuron(X.shape[0])
print(neuron.W)
print(neuron.W.shape)
print(neuron.b)
print(neuron.A)
neuron.A = 10
print(neuron.A)

