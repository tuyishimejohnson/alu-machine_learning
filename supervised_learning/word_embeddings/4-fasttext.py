#!/usr/bin/env python3
"""
FastText Model Creation
"""

from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5, window=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    Creates and trains a Gensim FastText model.
    """
    # Set the training algorithm to CBOW or Skip-gram
    sg = 0 if cbow else 1

    # Initialize and train the FastText model
    model = FastText(
        sentences=sentences,
        vector_size=size,
        window=window,
        min_count=min_count,
        sg=sg,
        negative=negative,
        seed=seed,
        epochs=iterations,
        workers=workers
    )

    return model