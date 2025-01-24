#!/usr/bin/env python3
"""
that creates a TF-IDF embedding
"""

from sklearn.feature_extraction.text import TfidfVectorizer


def tf_idf(sentences, vocab=None):
    """
    args:
    a list of sentences to analyze
    vocabulary: a list of vocabulary to analyze
    Returns:
    embeddings and features
    """

    # Initialize TF-IDF vectorizer with optional vocabulary
    vectorizer = TfidfVectorizer(vocabulary=vocab)

    # Fit and transform sentences
    tfidf_matrix = vectorizer.fit_transform(sentences)

    # Extract feature names (vocabulary) if not provided
    features = vectorizer.get_feature_names()

    # Convert TF-IDF matrix to an array (embeddings)
    embeddings = tfidf_matrix.toarray()

    return embeddings, features