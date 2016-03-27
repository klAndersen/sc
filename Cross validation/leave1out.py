"""

Based on:
http://scikit-learn.org/stable/modules/cross_validation.html#leave-one-out-loo


"""

from sklearn import cross_validation
from sklearn.cross_validation import LeavePOut
from sklearn.cross_validation import LeaveOneOut
from sklearn.cross_validation import LeavePLabelOut
from sklearn.cross_validation import LeaveOneLabelOut
from sklearn.cross_validation import LabelShuffleSplit

'''
LeaveOneOut (or LOO) is a simple cross-validation. Each learning set is created by taking all the samples except one,
the test set being the sample left out. Thus, for n samples, we have n different training sets and n different tests
set. This cross-validation procedure does not waste much data as only one sample is removed from the training set:
'''
print("LeaveOneOut")
loo = LeaveOneOut(4)
for train, test in loo:
    print("%s %s" % (train, test))


'''
LeavePOut is very similar to LeaveOneOut as it creates all the possible training/test sets by removing p samples from
the complete set. For n samples, this produces {n \choose p} train-test pairs. Unlike LeaveOneOut and KFold, the test
sets will overlap for p > 1. Example of Leave-2-Out on a dataset with 4 samples:
'''
print("LeavePOut")
lpo = LeavePOut(4, p=2)
for train, test in lpo:
    print("%s %s" % (train, test))


'''
LeaveOneLabelOut (LOLO) is a cross-validation scheme which holds out the samples according to a third-party provided
array of integer labels. This label information can be used to encode arbitrary domain specific pre-defined
cross-validation folds. Each training set is thus constituted by all the samples except the ones related to a specific
label. For example, in the cases of multiple experiments, LOLO can be used to create a cross-validation based on the
different experiments: we create a training set using the samples of all the experiments except one:
'''
print("LeaveOneLabelOut")
labels = [1, 1, 2, 2]
lolo = LeaveOneLabelOut(labels)
for train, test in lolo:
    print("%s %s" % (train, test))

# LeavePLabelOut
'''
LeavePLabelOut is similar as Leave-One-Label-Out, but removes samples related to P labels for each training/test set.
Example of Leave-2-Label Out:
'''
print("LeaveOneOut")
labels = [1, 1, 2, 2, 3, 3]
lplo = LeavePLabelOut(labels, p=2)
for train, test in lplo:
    print("%s %s" % (train, test))

ss = cross_validation.ShuffleSplit(5, n_iter=3, test_size=0.25, random_state=0)
for train_index, test_index in ss:
    print("%s %s" % (train_index, test_index))


'''
The ShuffleSplit iterator will generate a user defined number of independent train / test dataset splits.
Samples are first shuffled and then split into a pair of train and test sets. It is possible to control the
randomness for reproducibility of the results by explicitly seeding the random_state pseudo random number generator.
'''
print("LabelShuffleSplit")
labels = [1, 1, 2, 2, 3, 3, 4, 4]
slo = LabelShuffleSplit(labels, n_iter=4, test_size=0.5, random_state=0)
for train, test in slo:
    print("%s %s" % (train, test))
