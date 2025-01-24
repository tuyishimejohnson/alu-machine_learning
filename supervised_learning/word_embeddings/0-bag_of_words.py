#!/usr/bin/env python3


'''
A function that creates a TF-IDF embedding.
'''

import numpy as np
from collections import Counter
from math import log

def tf_idf(sentences, vocab=None):
    """
    Computes TF-IDF embeddings for a list of sentences
    Returns:
    A numpy.ndarray of shape (s, f) containing the embeddings
    A list of features (vocabulary) used for embeddings
    """
    tokenized_sentences = [sentence.split() for sentence in sentences]

    if vocab is None:
        vocab = sorted(set(word for sentence in tokenized_sentences for word in sentence))
    else:
        vocab = sorted(vocab)
    
    # Initialize variables
    s = len(sentences)
    f = len(vocab)
    vocab_index = {word: idx for idx, word in enumerate(vocab)}
    
    # Compute term frequency (TF)
    tf = np.zeros((s, f), dtype=float)
    for i, sentence in enumerate(tokenized_sentences):
        word_counts = Counter(sentence)
        total_words = len(sentence)
        for word, count in word_counts.items():
            if word in vocab_index:
                tf[i, vocab_index[word]] = count / total_words
    
    # Compute inverse document frequency (IDF)
    idf = np.zeros(f, dtype=float)
    for word, idx in vocab_index.items():
        doc_count = sum(1 for sentence in tokenized_sentences if word in sentence)
        idf[idx] = log(s / (1 + doc_count))
    
    # Compute TF-IDF
    tf_idf_matrix = tf * idf
    
    return tf_idf_matrix, vocab


