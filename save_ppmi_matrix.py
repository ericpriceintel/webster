
from collections import defaultdict
from itertools import combinations_with_replacement
import string
import sys

import numpy as np
from scipy import sparse, io


EXCLUDE_DIR = 'documents/exclude.txt'
MATRIX_DIR = 'matrices/%s_matrix.mtx'
KEYWORD_DIR = 'matrices/%s_word_list.txt'


def read_tweets(text_dir):

    with open(text_dir) as f:
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


def save(lookup, matrix, prefix):
    """Save the matrix and word ordering for later use.

    Forgoes the need to compute the PPMI matrix on each run.
    """
    reverse_lookup = {i: word for word, i in lookup.items()}
    word_list = ','.join([reverse_lookup[i] for i in range(len(keywords))])

    io.mmwrite(MATRIX_DIR % prefix, matrix)
    with open(KEYWORD_DIR % prefix, 'w') as f:
        f.write(word_list)


if __name__ == '__main__':

    text_dir, prefix = sys.argv[1], sys.argv[2]

    docs = read_tweets(text_dir)
    exclude = read_exclude()
    keywords = {
        w for w in set.union(*docs) - exclude
        if len(w) > 2 and not w.startswith('http')}

    print("There are %s distinct keywords in this dataset." % len(keywords))

    counts, associations = probabilities(keywords, docs)

    ordered = [(word, count) for word, count in counts.items()]
    ordered = sorted(ordered, key=lambda x: x[1], reverse=True)

    lookup = {word: i for i, word in enumerate(keywords)}
    matrix = build_matrix(lookup, keywords, counts, associations)

    save(lookup, matrix, prefix)
