## @realdonaldtrump's Tweets Thesaurus.


### Overview:
This is a "thesaurus" of @realdonaldtrump's tweets. It accepts a word as
input and finds the keywords in @realdonaldtrump's tweets that are most closely
related to the given word.


### How this works:
I began by pulling the last 3,200 of his tweets (maximum amount allowed by
Twitter), and used the data to construct a PPMI matrix of all of the keywords,
where each column represents a unique keyword in the tweets.
From there, I reduced the dimensionality of this matrix via SVD.

Using this reduced matrix, I construct a vector for a given search term and then
compute the dot product between each word vector and each column in the matrix.
The columns that have largest dot product value with the given search term
will represent the words that are most similar to the given search term.


### How I tested this:
While this is hard to test via unit or integration tests, I perform a similar
analysis using a thesaurus. With the thesaurus as the document corpus, I found
that the top results were close synonyms with the given search term.

I also ran this analysis for 20+ keywords in @realdonaldtrump's tweets, and also
found the top results to be very similar to how Trump speaks.


### How to run this for yourself:
To install the required packages for this to work, run:
    <br>
    `pip install -r requirements.txt`

To generate the underlying matrix for a different text corpus, run:
    <br>
    `python save_ppmi_matrix.py [path_to_text_corpus] [file_prefix]`

This will generate two files in the `matrices/` directory, both prefixed with
your given `file_prefix`. One will be the raw PPMI matrix, with the extension
`.mtx` and the other will be an ordered list of keywords, with the extension
`.txt`.

To use the actual thesaurus, run:
    <br>
    `python thesaurus.py [path_to_matrix] [path_to_word_list] [search_term]`


### The research behind the methodology:
Most of the math behind this method is derived from:
    <br>
    - http://eng.datafox.com/general/2015/04/17/keyword-similarities/
    <br>
    - http://papers.nips.cc/paper/5477-neural-word-embedding-as-implicit-matrix-factorization.pdf
