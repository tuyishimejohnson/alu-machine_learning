#!/usr/bin/env python3

""" A function that creates a forward propagation graph
for the neural network
"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    args:
    x: the placeholder for input data
    layer_sizes: list of nodes in each layer
    activation: list activation function for a layer to use

    returns:
    prediction of the network in tensor form
    """
    layer =x 
    for size, activation in zip(layer_sizes, activations):
        layer = create_layer(layer, size, activation)
    return layer
