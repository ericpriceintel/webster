
from collections import defaultdict
from itertools import combinations_with_replacement

import numpy as np
from scipy import sparse
from scipy.sparse import linalg


DIMS = 10


def probabilities(keywords, docs):
    """Calculate the PPMI between the keywords of the given documents.

    each entry is the number of times we've the two words together.
    how do we generate this efficiently?
    matrix =

    mapping of keyword to the number of times we've seen that keyword
    counts = {keyword: += 1}
    """
    counts = defaultdict(int)
    associations = defaultdict(int)

    for doc in docs:
        for i, keyword in enumerate(keywords):
            if keyword in doc:
                counts[keyword] += 1

        for word1, word2 in combinations_with_replacement(keywords, 2):
            if word1 in doc and word2 in doc:
                associations[(word1, word2)] += 1

    return counts, associations


def build_matrix(lookup, keywords, counts, assocs):
    """Build the PPMI matrix."""

    matrix = sparse.dok_matrix((len(keywords), len(keywords)), dtype=np.float64)
    keyword_count = len(keywords)

    for word1, word2 in combinations_with_replacement(keywords, 2):
        word1_idx = lookup[word1]
        word2_idx = lookup[word2]

        num = assocs[(word1, word2)] * keyword_count
        denom = counts[word1] * counts[word2]

        pmi = np.log(num / denom)
        ppmi = max(0, pmi)

        matrix[word1_idx, word2_idx] = ppmi
        matrix[word2_idx, word1_idx] = ppmi

    return matrix.tocsc()


def build_word_vec(lookup, word):

    vec = np.zeros(len(lookup))
    word_idx = lookup[word]

    vec[word_idx] = 1
    return vec


def match(word_vec, u, s, v):
    """Find the closest vectors."""

    vec_in_basis = np.dot(np.dot(s, u.transpose()), word_vec)

    similarities = []
    for i, col in enumerate(v.transpose()):
        likeness = (
            np.dot(vec_in_basis, col) /
            np.linalg.norm(vec_in_basis) * np.linalg.norm(col))
        similarities.append(likeness)

    return similarities


def display_likeness(lookup, similarities):
    reverse_lookup = {idx: word for word, idx in lookup.items()}
    ordered = [
        (reverse_lookup[i], likeness) for i, likeness in enumerate(similarities)]

    for word, likeness in sorted(ordered, key=lambda x: x[1], reverse=True):
        print(word, likeness)


if __name__ == '__main__':
    with open('documents.txt') as f:
        tweets = map(lambda s: s.strip().split(), f.readlines())
        docs = [set(map(lambda s: s.lower(), tweet)) for tweet in tweets]

    keywords = set.union(*docs)

    counts, associations = probabilities(keywords, docs)

    lookup = {word: i for i, word in enumerate(keywords)}
    matrix = build_matrix(lookup, keywords, counts, associations)

    # np.savetxt('test.csv', matrix.todense(), delimiter=',')

    word_vec = build_word_vec(lookup, 'apple')
    u, s, v = linalg.svds(matrix, k=DIMS)

    similarities = match(word_vec, u, np.diag(s), v)

    display_likeness(lookup, similarities)
