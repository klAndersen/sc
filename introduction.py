"""
An introduction to machine learning with scikit-learn
http://scikit-learn.org/stable/tutorial/basic/tutorial.html
"""

import pickle
import numpy as np
from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets
from sklearn.externals import joblib
from sklearn import random_projection

'''
Loading an example dataset
http://scikit-learn.org/stable/tutorial/basic/tutorial.html#loading-an-example-dataset
'''

# load datasets
iris = datasets.load_iris()
digits = datasets.load_digits()

# A dataset is a dictionary-like object that holds all the data and some metadata about the data.
# This data is stored in the .data member, which is a n_samples, n_features array.  In the case of
# supervised problem, one or more response variables are stored in the .target member.

# digits.data gives access to the features that can be used to classify the digits samples:
print("digits.data: ")
print(digits.data)
print('\n')
# digits.target gives the ground truth for the digit dataset, that is the number corresponding to each
# digit image that we are trying to learn:
print("digits.target: ")
print(digits.target)
print('\n')
# The data is always a 2D array, shape (n_samples, n_features), although the original data may have had a
# different shape. In the case of the digits, each original sample is an image of shape (8, 8) and can be
# accessed using:
print("digits.images[0]: ")
print(digits.images[0])
print('\n')

'''
Learning and predicting
http://scikit-learn.org/stable/tutorial/basic/tutorial.html#learning-and-predicting
'''

# clf = classifier; estimator, SVC = Support Vector Classification
# here gamma is set manually, but it can be estimated by using
# grid search (http://scikit-learn.org/stable/modules/grid_search.html#grid-search)
# or cross-validation (http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation)
clf = svm.SVC(gamma=0.001, C=100.)

# fit the data by selecting all but the last one (achieved by using [:-1])
clf.fit(digits.data[:-1], digits.target[:-1])
print("=== clf = svm.SVC(gamma=0.001, C=100.); [:-1] ===")
print(clf)
print('\n')

#
print("Prediction: ")
print(clf.predict(digits.data[-1:]))
print('\n')

'''
Model persistence
http://scikit-learn.org/stable/tutorial/basic/tutorial.html#model-persistence

See also:
http://scikit-learn.org/stable/modules/model_persistence.html#model-persistence
'''

clf = svm.SVC()
X, y = iris.data, iris.target
print("clf.fit(X, y): ")
print(clf.fit(X, y))
print('\n')

# https://docs.python.org/2/library/pickle.html
s = pickle.dumps(clf)
clf2 = pickle.loads(s)
clf2.predict(X[0:1])

print("y[0]: ")
print(y[0])
print('\n')

'''
joblib.dump returns a list of filenames. Each individual numpy array contained in the clf object is serialized as a
separate file on the filesystem. All files are required in the same folder when reloading the model with joblib.load.
'''
# dump data to file by using joblib
joblib.dump(clf, './introduction_files/filename.pkl')
# load dumped data from file
clf = joblib.load('./introduction_files/filename.pkl')

'''
Conventions
http://scikit-learn.org/stable/tutorial/basic/tutorial.html#conventions
'''

# --- Type casting ----
rng = np.random.RandomState(0)
X = rng.rand(10, 2000)
# Set data type to float32 (unless specified, input is cast to float64)
X = np.array(X, dtype='float32')
# print the data type of X
print("Data type of X: ")
print(X.dtype)
print('\n')

# here, X is cast to float64 by the fit_transform
transformer = random_projection.GaussianRandomProjection()
X_new = transformer.fit_transform(X)
# print the data type of X
print("New data type of X: ")
print(X_new.dtype)

# --- Regression ----
iris = datasets.load_iris()
clf = SVC()
print("clf.fit(iris.data, iris.target): ")
print(clf.fit(iris.data, iris.target))
print('\n')

# returns an integer array, since iris.target (an integer array) was used in fit
print("Prediction array with integer based on clf: ")
print(list(clf.predict(iris.data[:3])))
print('\n')

# Classification fit; iris.target_names are string values
print("clf.fit(iris.data, iris.target_names[iris.target]): ")
print(clf.fit(iris.data, iris.target_names[iris.target]))
print('\n')

# returns a string array, since iris.target_names was for fitting
print("Prediction array with strings based on clf: ")
print(list(clf.predict(iris.data[:3])))
print('\n')

# --- Refitting and updating parameters ---
rng = np.random.RandomState(0)
X = rng.rand(100, 10)
y = rng.binomial(1, 0.5, 100)
X_test = rng.rand(5, 10)

# Construct the estimator, then change kernel from rbf to linear
# Note! Calling fit() more than once will overwrite previous values
clf = SVC()
print("clf.set_params(kernel='linear').fit(X, y): ")
print(clf.set_params(kernel='linear').fit(X, y))
print('\n')

# first prediction
print("clf.predict(X_test): ")
print(clf.predict(X_test))
print('\n')

# change the kernel back to rbf to refit estimator
print("clf.set_params(kernel='rbf').fit(X, y): ")
print(clf.set_params(kernel='rbf').fit(X, y))
print('\n')

# make second prediction
print("clf.predict(X_test): ")
print(clf.predict(X_test))
print('\n')
