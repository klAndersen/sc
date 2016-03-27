# http://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage

from sklearn.feature_extraction.text import CountVectorizer


# functions made based on examples:

def print_feature_names(matrix=CountVectorizer):
    """
    Prints the feature names that exists within the passed matrix

    Arguments:
        matrix (CountVectorizer): Matrix with content to print

    Returns:
        str: String containing the feature names

    """
    print "Frequency of given term (column) at given position in the given document (row):"
    term_str = ""
    for term in matrix.get_feature_names():
        term_str += term + "; "
    return term_str


def print_feature_index(matrix=CountVectorizer, term=str):
    """
    Prints the feature names that exists within the passed matrix

    Arguments:
        matrix (CountVectorizer): Matrix with content to print
        term (str): The terms which feature index we are looking for

    Returns:
        CountVectorizer.vocabulary_.get(): Array/matrix containing the feature index

    """
    print "Feature index of the given term(s): "
    print "The term(s) '" + term + "' have index: "
    return matrix.vocabulary_.get(term)


print " ===== Common-Vectorizer-Usage ===== "
print

vectorizer = CountVectorizer(min_df=1)
print vectorizer
print

'''
# tabbed for index view
corpus = [
    # 1      # 2    # 3     # 4         # 5
    'This    is     the     first       document.',                     # 0
    # 1      # 2    # 3     # 4         # 5             # 6
    'This    is     the     second      second          document.',     # 1
    # 1      # 2    # 3     # 4         # 5
    'And     the    third   one.',                                      # 2
    # 1      # 2    # 3     # 4         # 5
    'Is      this   the     first       document?',                     # 3
]
'''

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

print "prints (doc=row, index=column, frequency), \n" \
      "(index=column is sorted alphabetically (a-z))"
print "(doc=row, index=column, frequency)"
X = vectorizer.fit_transform(corpus)
print X
print

print "Feature names: "
print vectorizer.get_feature_names()
print

analyze = vectorizer.build_analyzer()
print analyze("This is a text document to analyze.") == (['this', 'is', 'text', 'document', 'to', 'analyze'])
print

# term 1 - 8; ordered alphabetically
print vectorizer.get_feature_names() == (['and', 'document', 'first', 'is', 'one', 'second', 'the', 'third', 'this'])
print

'''
prints out the following array
array([[0, 1, 1, 1, 0, 0, 1, 0, 1],  # corpus[0]
       [0, 1, 0, 1, 0, 2, 1, 0, 1],  # corpus[1]
       [1, 0, 0, 0, 1, 0, 1, 1, 0],  # corpus[2]
       [0, 1, 1, 1, 0, 0, 1, 0, 1]]...)  # corpus[3]

(where row=corpus document, column=term (sorted alphabetically), value=frequency):
'''

print print_feature_names(vectorizer)
print X.toarray()
print

# maps the feature name to the column index; e.g. document: 1, this: 8, and: 0
print print_feature_index(vectorizer, 'document')
print

# prints out a row of '0's because none of these words exists in the current matrix/array
# "words not seen in the training corpus will be completely ignored in future calls to the transform"
print vectorizer.transform(['Something completely new.']).toarray()
print

# splits the words, so that each word is read 1-gram, and then 2-gram
# e.g. "Bi-grams" (one word) is read as two separate words when read as 1-gram,
# but when read as 2-gram it becomes "Bi grams" (no '-' because symbols are ignored)
# this can be used to maintain the ordering of the words
bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
analyze = bigram_vectorizer.build_analyzer()
# prints True because both versions are identical to the ngram_range in the analyzer
print analyze('Bi-grams are cool!') == (['bi', 'grams', 'are', 'cool', 'bi grams', 'grams are', 'are cool'])
print

# using the corpus from earlier, we get a larger matrix/array (4 x 21)
X_2 = bigram_vectorizer.fit_transform(corpus).toarray()

print print_feature_names(bigram_vectorizer)
print X_2
print

'''
X_2[:, feature_index] prints [0 0 0 1], where

- the row value is the frequency of the terms
- the column value is the corpus/document in which the terms reside in

Thus, it goes to the column with index=feature_index and then returns
that column as a row:

                d1      d2      d3      d4
frequency:   [   0      0       0       1   ]

'''

feature_index = print_feature_index(bigram_vectorizer, 'is this')
# print feature_index  # prints 7
print X_2[:, feature_index]  # prints [0 0 0 1]
print
