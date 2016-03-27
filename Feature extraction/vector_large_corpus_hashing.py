# http://scikit-learn.org/stable/modules/feature_extraction.html#vectorizing-a-large-text-corpus-with-the-hashing-trick

from sklearn.feature_extraction.text import HashingVectorizer

print " ===== Vector-large-corpus-hashing ===== "
print

corpus = [
    'This is the first document.',
    'This is the second second document.',
    'And the third one.',
    'Is this the first document?',
]

hv = HashingVectorizer(n_features=10)
print hv.transform(corpus)
print

hv = HashingVectorizer()
print hv.transform(corpus)
print
