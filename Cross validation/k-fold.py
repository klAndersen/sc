"""

Based on:
http://scikit-learn.org/stable/modules/cross_validation.html#k-fold


"""

import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import LabelKFold
from sklearn.cross_validation import StratifiedKFold

# Example of 2-fold cross-validation on a dataset with 4 samples:
print("KFold:")
kf = KFold(4, n_folds=2)
for train, test in kf:
    print("%s %s" % (train, test))

# create training/test sets using numpy indexing:
X = np.array([[0., 0.], [1., 1.], [-1., -1.], [2., 2.]])
y = np.array([0, 1, 0, 1])
X_train, X_test, y_train, y_test = X[train], X[test], y[train], y[test]

'''
StratifiedKFold is a variation of k-fold which returns stratified folds: each set contains approximately
the same percentage of samples of each target class as the complete set.
'''
# Example of stratified 3-fold cross-validation on a dataset with 10 samples from two slightly unbalanced classes:
print("StratifiedKFold:")
labels = [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
skf = StratifiedKFold(labels, 3)
for train, test in skf:
    print("%s %s" % (train, test))

# LabelKFold
'''
LabelKFold is a variation of k-fold which ensures that the same label is not in both testing and training sets.
This is necessary for example if you obtained data from different subjects and you want to avoid over-fitting
(i.e., learning person specific features) by testing and training on different subjects.
'''
print("LabelKFold:")
labels = [1, 1, 1, 2, 2, 2, 3, 3, 3, 3]

lkf = LabelKFold(labels, n_folds=3)
for train, test in lkf:
    print("%s %s" % (train, test))
