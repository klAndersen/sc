# Taken from: http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html

from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest, chi2

'''
prints:

[[ 2.  0.  1.]
 [ 0.  1.  3.]]

because it sorts alphabetically, where row=value, column=key;

 ['bar',    'baz',  'foo']

 [ 2.       0.      1.]  # baz has no value
 [ 0.       1.      3.]  # bar has no value

'''
v = DictVectorizer(sparse=False)
D = [{'foo': 1, 'bar': 2}, {'foo': 3, 'baz': 1}]
X = v.fit_transform(D)

print X
print

print (v.inverse_transform(X) == [{'bar': 2.0, 'foo': 1.0}, {'baz': 1.0, 'foo': 3.0}])
print

# prints [[ 0.  0.  4.]] because only foo exists in 'v'
print (v.transform({'foo': 4, 'unseen_feature': 3}))
print

# http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.DictVectorizer.html#sklearn.feature_extraction.DictVectorizer.restrict

support = SelectKBest(chi2, k=2).fit(X, [0, 1])

print v.get_feature_names()
print

print v.restrict(support.get_support())
print

print v.get_feature_names()
print
