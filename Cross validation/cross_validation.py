"""

Based on:
http://scikit-learn.org/stable/modules/cross_validation.html#cross-validation

Learning the parameters of a prediction function and testing it on the same data is a methodological mistake:
a model that would just repeat the labels of the samples that it has just seen would have a perfect score but
would fail to predict anything useful on yet-unseen data. This situation is called overfitting. To avoid it, it
is common practice when performing a (supervised) machine learning experiment to hold out part of the available
data as a test set X_test, y_test.

When evaluating different settings ("hyperparameters") for estimators, such as the C setting that must be manually
set for an SVM, there is still a risk of overfitting on the test set because the parameters can be tweaked until
the estimator performs optimally. This way, knowledge about the test set can ""leak into the model and evaluation
metrics no longer report on generalization performance. To solve this problem, yet another part of the dataset can be
held out as a so-called "validation set": training proceeds on the training set, after which evaluation is done on the
 validation set, and when the experiment seems to be successful, final evaluation can be done on the test set.

However, by partitioning the available data into three sets, we drastically reduce the number of samples which can
be used for learning the model, and the results can depend on a particular random choice for the pair of (train,
validation) sets.

A solution to this problem is a procedure called cross-validation (CV for short). A test set should still be held
out for final evaluation, but the validation set is no longer needed when doing CV. In the basic approach, called
k-fold CV, the training set is split into k smaller sets (other approaches are described below, but generally follow
the same principles). The following procedure is followed for each of the k "folds":

- A model is trained using k-1 of the folds as training data;
- the resulting model is validated on the remaining part of the data
  (i.e., it is used as a test set to compute a performance measure such as accuracy).

The performance measure reported by k-fold cross-validation is then the average of the values computed in the loop.
This approach can be computationally expensive, but does not waste too much data (as it is the case when fixing an
arbitrary test set), which is a major advantage in problem such as inverse inference where the number of samples is
very small.

"""

from sklearn import svm
from sklearn import metrics
from sklearn import datasets
from sklearn import preprocessing
from sklearn import cross_validation
from sklearn.pipeline import make_pipeline

iris = datasets.load_iris()
print("IRIS")
print(iris.data.shape, iris.target.shape)

# load training data, but hold out 40% for training (test_size=0.4)
X_train, X_test, y_train, y_test = cross_validation.train_test_split(iris.data, iris.target,
                                                                     test_size=0.4, random_state=0)
print('\n')
print("X/Y Train")
print(X_train.shape, y_train.shape)

print('\n')
print("X/Y Test")
print(X_test.shape, y_test.shape)


clf = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
print('\n')
print("clf.score")
print(clf.score(X_test, y_test))

clf = svm.SVC(kernel='linear', C=1)
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)
print('\n')
print("cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5)")
print(scores)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

# metrics
scores = cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_weighted')
print('\n')
print("cross_validation.cross_val_score(clf, iris.data, iris.target, cv=5, scoring='f1_weighted')")
print(scores)

n_samples = iris.data.shape[0]
cv = cross_validation.ShuffleSplit(n_samples, n_iter=3, test_size=0.3, random_state=0)

cross_validation.cross_val_score(clf, iris.data, iris.target, cv=cv)

# preprocessing
X_train, X_test, y_train, y_test = cross_validation.train_test_split(
    iris.data, iris.target, test_size=0.4, random_state=0)
scaler = preprocessing.StandardScaler().fit(X_train)
X_train_transformed = scaler.transform(X_train)
clf = svm.SVC(C=1).fit(X_train_transformed, y_train)
X_test_transformed = scaler.transform(X_test)
print('\n')
print("Preprocessing - clf.score")
print(clf.score(X_test_transformed, y_test))

# make_pipeline
clf = make_pipeline(preprocessing.StandardScaler(), svm.SVC(C=1))
cross_validation.cross_val_score(clf, iris.data, iris.target, cv=cv)

predicted = cross_validation.cross_val_predict(clf, iris.data, iris.target, cv=10)
metrics.accuracy_score(iris.target, predicted)

