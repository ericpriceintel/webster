
from collections import defaultdict
from itertools import combinations_with_replacement
import string
import sys

import numpy as np
from scipy import sparse
from scipy.sparse import linalg


DIMS = 100
EXCLUDE_DIR = 'exclude.txt'


def read_tweets(filename):

    with open(filename) as f:
        tweets = map(lambda s: s.strip().split(), f.readlines())
        table = str.maketrans('', '', string.punctuation + string.digits)
        docs = [
            set(map(lambda s: s.lower().translate(table), tweet))
            for tweet in tweets]

    return docs


def read_exclude():
    with open(EXCLUDE_DIR) as f:
        exclude = set([word.strip() for word in f])

    return exclude


def probabilities(keywords, docs):
    """Find counts and associations between the keywords of the given documents.

    counts maps a word -> number of times it appears
    associations maps (word1, word2) -> number of times they appear together
    """
    counts = defaultdict(int)
    associations = defaultdict(int)
    ordered_keywords = list(keywords)

    for doc in docs:
        for i, keyword in enumerate(keywords):
            if keyword in doc:
                counts[keyword] += 1

        for i, word1 in enumerate(ordered_keywords):
            if word1 not in doc:
                continue

            for word2 in ordered_keywords[i:]:
                if word2 in doc:
                    associations[(word1, word2)] += 1

    return counts, associations


def build_matrix(lookup, keywords, counts, assocs):
    """Build the PPMI matrix."""

    keyword_count = len(keywords)
    matrix = sparse.lil_matrix((keyword_count, keyword_count), dtype=np.float64)

    for word1, word2 in combinations_with_replacement(keywords, 2):
        word1_idx = lookup[word1]
        word2_idx = lookup[word2]

        num = assocs[(word1, word2)] * keyword_count
        denom = counts[word1] * counts[word2]

        pmi = np.log(num / denom) if num > 0 else 0
        ppmi = max(0, pmi)

        matrix[word1_idx, word2_idx] = ppmi
        matrix[word2_idx, word1_idx] = ppmi

    return matrix


def build_word_vec(lookup, word):
    vec = np.zeros(len(lookup))
    word_idx = lookup[word]

    vec[word_idx] = 1
    return vec


def calc_similarities(word_vec, u, s, v):
    """Find the closest vectors."""

    vec_in_basis = np.dot(np.dot(s, u.transpose()), word_vec)

    similarities = []
    for i, col in enumerate(v.transpose()):
        likeness = np.dot(vec_in_basis, col)
        similarities.append(likeness)

    return np.array(similarities) / max(similarities)


def display_likeness(lookup, similarities):
    idx_lookup = {idx: word for word, idx in lookup.items()}
    ordered = [
        (idx_lookup[i], likeness) for i, likeness in enumerate(similarities)]

    for word, likeness in sorted(ordered, key=lambda x: x[1], reverse=True):
        print(word, likeness)


if __name__ == '__main__':

    filename, search_term = sys.argv[1], sys.argv[2]

    docs = read_tweets(filename)
    exclude = read_exclude()
    keywords = {
        w for w in set.union(*docs) ^ exclude
        if len(w) > 2 and not w.startswith('http')}

    print("There are %s distinct keywords in this dataset." % len(keywords))

    counts, associations = probabilities(keywords, docs)

    lookup = {word: i for i, word in enumerate(keywords)}

    matrix = build_matrix(lookup, keywords, counts, associations)

    u, s, v = linalg.svds(matrix, k=DIMS)

    # np.savetxt('test.csv', matrix.todense(), delimiter=',')

    word_vec = build_word_vec(lookup, search_term)
    similarities = calc_similarities(word_vec, u, np.diag(s), v)

    display_likeness(lookup, similarities)
