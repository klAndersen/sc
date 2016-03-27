# http://scikit-learn.org/stable/modules/feature_extraction.html#limitations-of-the-bag-of-words-representation
from sklearn.feature_extraction.text import CountVectorizer

print " ===== Limits-BOW-representations ===== "
print

ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2), min_df=1)
counts = ngram_vectorizer.fit_transform(['words', 'wprds'])
print ngram_vectorizer.get_feature_names() == ([' w', 'ds', 'or', 'pr', 'rd', 's ', 'wo', 'wp'])
print

print counts.toarray().astype(int)
print

ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5), min_df=1)
print ngram_vectorizer.fit_transform(['jumpy fox'])
print

print ngram_vectorizer.get_feature_names() == ([' fox ', ' jump', 'jumpy', 'umpy '])
print

ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5), min_df=1)
print ngram_vectorizer.fit_transform(['jumpy fox'])
print


print ngram_vectorizer.get_feature_names() == (['jumpy', 'mpy f', 'py fo', 'umpy ', 'y fox'])
print
