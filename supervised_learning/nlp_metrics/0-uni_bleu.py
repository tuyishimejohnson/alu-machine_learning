#!/usr/bin/env python3

"""
A function that calculates the unigram BLEU score for a sentence
"""

from collections import Counter
import math


def uni_bleu(references, sentence):
    """
    Calculates the unigram BLEU score for a sentence.
    """

    # Step 1: Count the number of words in the candidate sentence
    sentence_count = Counter(sentence)

    # Step 2: Count the maximum possible occurrences of each word
    max_ref_count = Counter()
    for ref in references:
        ref_count = Counter(ref)
        for word in ref_count:
            max_ref_count[word] = max(max_ref_count[word], ref_count[word])

    # Step 3: Clip the candidate counts by the maximum reference counts
    clipped_count = {
        word: min(sentence_count[word], max_ref_count[word])
        for word in sentence_count
        }

    # Calculate precision
    precision = sum(clipped_count.values()) / len(sentence)

    # Step 4: Calculate brevity penalty
    ref_lengths = [len(ref) for ref in references]
    closest_ref_len = min(
        ref_lengths,
        key=lambda ref_len: (abs(ref_len - len(sentence)), ref_len)
        )

    if len(sentence) > closest_ref_len:
        brevity_penalty = 1
    else:
        brevity_penalty = math.exp(1 - closest_ref_len / len(sentence))

    # Step 5: Calculate BLEU score (precision * brevity penalty)
    bleu_score = brevity_penalty * precision

    return bleu_score
