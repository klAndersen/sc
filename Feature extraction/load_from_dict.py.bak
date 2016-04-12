# http://scikit-learn.org/stable/modules/feature_extraction.html#loading-features-from-dicts

from sklearn.feature_extraction import DictVectorizer

# "city" is a categorical attribute while "temperature" is a traditional numerical feature
measurements = [
    {'city': 'Dubai', 'temperature': 33.},
    {'city': 'London', 'temperature': 12.},
    {'city': 'San Fransisco', 'temperature': 18.},
]

vec = DictVectorizer()

'''
prints
[[  1.   0.   0.  33.]
 [  0.   1.   0.  12.]
 [  0.   0.   1.  18.]]

because each city occurs only once for each temperature value
'''
print vec.fit_transform(measurements).toarray()
print

# prints: ['city=Dubai', 'city=London', 'city=San Fransisco', 'temperature']
print vec.get_feature_names()
print

# For example, suppose that we have a first algorithm that extracts Part of Speech (PoS) tags
# that we want to use as complementary tags for training a sequence classifier (e.g. a chunker).
# The following dict could be such a window of features extracted around the word 'sat' in the
# sentence 'The cat sat on the mat.':
pos_window = [
    {
        'word-2': 'the',
        'pos-2': 'DT',
        'word-1': 'cat',
        'pos-1': 'NN',
        'word+1': 'on',
        'pos+1': 'PP',
    },
    # in a real application one would extract many such dictionaries
]

# This description can be vectorized into a sparse two-dimensional matrix suitable for feeding
# into a classifier (maybe after being piped into a text.TfidfTransformer for normalization):
vec = DictVectorizer()
pos_vectorized = vec.fit_transform(pos_window)

# prints (row, column)
print pos_vectorized
print

# prints: [[ 1.  1.  1.  1.  1.  1.]]
print pos_vectorized.toarray()
print

# prints: ['pos+1=PP', 'pos-1=NN', 'pos-2=DT', 'word+1=on', 'word-1=cat', 'word-2=the']
print vec.get_feature_names()
print
