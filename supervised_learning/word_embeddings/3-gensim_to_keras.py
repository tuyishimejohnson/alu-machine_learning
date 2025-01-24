#!/usr/bin/env python3

'''
Converts a Gensim Word2Vec model to a Keras Embedding layer.
'''

from tensorflow.keras.layers import Embedding
import numpy as np


def gensim_to_keras(model):
    '''
    Args:
        model: A trained Gensim Word2Vec model.
    Returns:
        A trainable Keras Embedding layer
        initialized with the word vectors.
    '''
    
    
    # Extract the vocabulary size and vector dimensions
    vocab_size = len(model.wv.index_to_key)
    vector_dim = model.wv.vector_size

    # Create a weights matrix
    weights_matrix = np.zeros((vocab_size, vector_dim))
    for i, word in enumerate(model.wv.index_to_key):
        weights_matrix[i] = model.wv[word]

    # Create a Keras Embedding layer
    embedding_layer = Embedding(
        input_dim=vocab_size,
        output_dim=vector_dim,
        weights=[weights_matrix],
        trainable=True
    )

    return embedding_layer
