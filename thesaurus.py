
import sys

import numpy as np
from scipy import io
from scipy.sparse import linalg


DIMS = 100
MAX_RESULTS = 100


def load(matrix_dir, keyword_dir):
    """Load the pre-computed PPMI matrix and list of keywords from files."""

    matrix = io.mmread(matrix_dir)

    with open(keyword_dir) as f:
        keywords = f.readline().split(',')
        lookup = {keyword: i for i, keyword in enumerate(keywords)}

    return matrix, lookup


def build_word_vec(lookup, word, u, s):
    vec = np.zeros(len(lookup))
    word_idx = lookup[word]

    vec[word_idx] = 1
    return np.dot(np.dot(s, u.transpose()), vec)


def calc_similarities(word_vec, v):
    """Find the closest vectors."""

    similarities = []
    for i, col in enumerate(v.transpose()):
        likeness = np.dot(word_vec, col)
        similarities.append(likeness)

    return np.array(similarities)


def display_likeness(search_term, lookup, similarities):
    idx_lookup = {idx: word for word, idx in lookup.items()}

    ordered = [
        (idx_lookup[i], likeness) for i, likeness in enumerate(similarities)
        if idx_lookup[i] != search_term]
    ordered = sorted(ordered, key=lambda x: x[1], reverse=True)[:MAX_RESULTS]

    max_similarity = ordered[0][1]

    print('The words most similar to "%s" in Donald Trump\'s tweets are:' % search_term)
    for word, likeness in ordered:
        print(word, likeness / max_similarity)


if __name__ == '__main__':

    matrix_dir, keyword_dir, search_term = sys.argv[1], sys.argv[2], sys.argv[3]

    matrix, lookup = load(matrix_dir, keyword_dir)

    u, s, v = linalg.svds(matrix, k=DIMS)
    s = np.diag(np.sqrt(s))

    word_vec = build_word_vec(lookup, search_term, u, s)
    similarities = calc_similarities(word_vec, v)

    display_likeness(search_term, lookup, similarities)
