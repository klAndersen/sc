# http://scikit-learn.org/stable/modules/feature_extraction.html#tfidf-term-weighting
# Tf means term-frequency while tf-idf means term-frequency times inverse document-frequency

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

print " ===== Tfidf-Term-Weighting ===== "
print

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

transformer = TfidfTransformer()
print transformer
print

# here, first term is present in all documents (100% --- first column).
# as for the rest, they are below 50% and are therefore more representative
counts = [[3, 0, 1],
          [2, 0, 0],
          [3, 0, 0],
          [4, 0, 0],
          [3, 2, 0],
          [3, 0, 2]]

# tfidf only considers those where value > 0; 0 = non-existent
# therefore some values are'nt listed, e.g. (0, 1), (1, 1), (1, 2), etc.
tfidf = transformer.fit_transform(counts)
print tfidf  # .__repr__()  # prints object as shown in example
print

# prints the array containing the L2 normalized vectors
# see also:
# http://blog.christianperone.com/2011/10/machine-learning-text-feature-extraction-tf-idf-part-ii/
print tfidf.toarray()
print

# Each row is normalized to have unit euclidean norm.
# The weights of each feature computed by the fit method call are stored in a model attribute:
print transformer.idf_
print

# TfidfVectorizer combines all the options of CountVectorizer and TfidfTransformer in a single model
# see also (2014):
# http://stackoverflow.com/questions/22489264/is-a-countvectorizer-the-same-as-tfidfvectorizer-with-use-idf-false
vectorizer = TfidfVectorizer(min_df=1)
print vectorizer.fit_transform(corpus)
print

'''
Is a countvectorizer the same as tfidfvectorizer with use_idf=false? (2014)

http://stackoverflow.com/questions/22489264/is-a-countvectorizer-the-same-as-tfidfvectorizer-with-use-idf-false

This is done so that dot-products on the rows are cosine similarities.
Also TfidfVectorizer can use logarithmically discounted frequencies when given the option sublinear_tf=True.
To make TfidfVectorizer behave as CountVectorizer, give it the constructor options use_idf=False, normalize=None.
--> norm=None

TfidfVectorizer normalizes its results, i.e. each vector in its output has norm 1:
'''
print CountVectorizer().fit_transform(["foo bar baz", "foo bar quux"]).A

print TfidfVectorizer(use_idf=False).fit_transform(["foo bar baz", "foo bar quux"]).A
