# http://scikit-learn.org/stable/modules/feature_extraction.html#customizing-the-vectorizer-classes

from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer

print " ===== Customizing-the-Vectorizer-classes ===== "
print


def my_tokenizer(s):
    return s.split()

vectorizer = CountVectorizer(tokenizer=my_tokenizer)
print vectorizer.build_analyzer()(u"Some... punctuation!") == (['some...', 'punctuation!'])


class LemmaTokenizer(object):

    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]

vect = CountVectorizer(tokenizer=LemmaTokenizer())

print vect
