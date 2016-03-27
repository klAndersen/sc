"""
Working With Text Data
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html
"""

import numpy as np
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.datasets import load_files
from sklearn.grid_search import GridSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
# from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer


'''
Loading the 20 newsgroups dataset
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#loading-the-20-newsgroups-dataset
'''

# Selected categories for this tutorial
categories = ['alt.atheism',
              'soc.religion.christian',
              'comp.graphics',
              'sci.med']

# twenty_train = fetch_20newsgroups(subset='train', categories=categories, shuffle=True, random_state=42)
# Fails with error: "No handlers could be found for logger "sklearn.datasets.twenty_newsgroups"

# path to location with training data
container_path_training = "./data/twenty_newsgroups/20news-bydate-train/"
container_path_testing = "./data/twenty_newsgroups/20news-bydate-test/"
# load the matching categories
# too avoid UnicodeDecodeError, set encoding to latin1;
# see http://stackoverflow.com/questions/28888336/error-with-the-loading-files-in-scikit
twenty_train = load_files(container_path_training, description=None, categories=categories, load_content=True,
                          shuffle=True, encoding='latin1', decode_error='strict', random_state=42)

# print the list of the requested category names
print(twenty_train.target_names)
# print filenames

print("len(twenty_train.data): " + len(twenty_train.data).__str__(),
      "len(twenty_train.filenames): " + len(twenty_train.filenames).__str__())
print('\n')

# print the first lines of the first loaded file
print("\n".join(twenty_train.data[0].split("\n")[:3]))
print('\n')
print(twenty_train.target_names[twenty_train.target[0]])
print('\n')

'''
For speed and space efficiency reasons scikit-learn loads the target attribute as an array of integers
that corresponds to the index of the category name in the target_names list. The category integer id of
each sample is stored in the target attribute:
'''

print(twenty_train.target[:10])
print('\n')

# get back the category names
for t in twenty_train.target[:10]:
    print(twenty_train.target_names[t])


'''
Extracting features from text files
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#extracting-features-from-text-files
'''

# build a dictionary of features and transform documents to feature vectors
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(twenty_train.data)
print(X_train_counts.shape)

# count N-grams of words or consequective characters.
# Once fitted, the vectorizer builds a dictionary of feature indices
print(count_vect.vocabulary_.get(u'algorithm'))

# compute tf and tf-idf ("Term Frequency times Inverse Document Frequency")
# computing tf
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
print(X_train_tf.shape)

# computing tf-idf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
print(X_train_tfidf.shape)


'''
Training a classifier
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#training-a-classifier
'''

# train a classifier by using naive Bayes with a multinomial variant
clf = MultinomialNB().fit(X_train_tfidf, twenty_train.target)
# new documents to predict
docs_new = ['God is love', 'OpenGL on the GPU is fast']

# using transform instead of fit_transform, since the transformers have already been fit to the training set
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

# print prediction
predicted = clf.predict(X_new_tfidf)
for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, twenty_train.target_names[category]))


'''
Building a pipeline
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#building-a-pipeline
'''

# pipeline class simplying the process ( vectorizer => transformer => classifier)
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
                     ])
# train the model
text_clf = text_clf.fit(twenty_train.data, twenty_train.target)


'''
Evaluation of the performance on the test set
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#evaluation-of-the-performance-on-the-test-set
'''

# commented out due to previously mentioned error
# twenty_test = fetch_20newsgroups(subset='test', categories=categories, shuffle=True, random_state=42)

# load the test data
twenty_test = load_files(container_path_testing, description=None, categories=categories, load_content=True,
                         shuffle=True, encoding='latin1', decode_error='strict', random_state=42)
docs_test = twenty_test.data
predicted = text_clf.predict(docs_test)
# predict model accuracy: ~0.8348 (83.4%)
print(np.mean(predicted == twenty_test.target))

# do the same, just this time using SVM
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
                     ])
_ = text_clf.fit(twenty_train.data, twenty_train.target)
predicted = text_clf.predict(docs_test)
# predict model accuracy: ~0.908 (90.8%)
print(np.mean(predicted == twenty_test.target))

# classification report; precision, recall, f1-score, support and (avg / total)
print(metrics.classification_report(twenty_test.target, predicted, target_names=twenty_test.target_names))

# confusion matrix;
print(metrics.confusion_matrix(twenty_test.target, predicted))


'''
Parameter tuning using grid search
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#parameter-tuning-using-grid-search

Instead of tweaking the parameters of the various components of the chain, it is possible to run an exhaustive search
of the best parameters on a grid of possible values. We try out all classifiers on either words or bigrams, with or
without idf, and with a penalty parameter of either 0.01 or 0.001 for the linear SVM:
'''

parameters = {'vect__ngram_range': [(1, 1), (1, 2)],
              'tfidf__use_idf': (True, False),
              'clf__alpha': (1e-2, 1e-3),
              }

# If we have multiple CPU cores at our disposal, we can tell the grid searcher to try these eight parameter
# combinations in parallel with the n_jobs parameter. If we give this parameter a value of -1, grid search
# will detect how many cores are installed and uses them all:
gs_clf = GridSearchCV(text_clf, parameters, n_jobs=-1)

# Execute search on a smaller subset
# calling fit on a GridSearchCV object is a classifier that we can use to predict
gs_clf = gs_clf.fit(twenty_train.data[:400], twenty_train.target[:400])
print(twenty_train.target_names[gs_clf.predict(['God is love'])])

# can get optimal parameters by inspecting the object's grid_scores_ attribute, which
# is a list of parameters/score pairs. To get the best scoring attributes, we can do:
best_parameters, score, _ = max(gs_clf.grid_scores_, key=lambda x: x[1])
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

print(score)


'''
Exercise 1: Language identification
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#exercise-1-language-identification

Write a text classification pipeline using a custom preprocessor and CharNGramAnalyzer using data from Wikipedia
articles as training set. Evaluate the performance on some held out test set.

ipython command line:
%run workspace/exercise_01_language_train_model.py data/languages/paragraphs/

'''

'''
Exercise 2: Sentiment Analysis on movie reviews
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#exercise-2-sentiment-analysis-on-movie-reviews

Write a text classification pipeline to classify movie reviews as either positive or negative.
Find a good set of parameters using grid search.
Evaluate the performance on a held out test set.

ipython command line:
%run workspace/exercise_02_sentiment.py data/movie_reviews/txt_sentoken/

'''

'''
Exercise 3: CLI text classification utility
http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#exercise-3-cli-text-classification-utility

Using the results of the previous exercises and the cPickle module of the standard library, write a command line
utility that detects the language of some text provided on stdin and estimate the polarity (positive or negative)
if the text is written in English.

Bonus point if the utility is able to give a confidence level for its predictions.

'''
