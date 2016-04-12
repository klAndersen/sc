# http://scikit-learn.org/stable/modules/feature_extraction.html#limitations-of-the-bag-of-words-representation
from sklearn.feature_extraction.text import CountVectorizer

print " ===== Limits-BOW-representations ===== "
print

# Option 'char_wb' creates character n-grams only from text inside word boundaries (padded with space on each side)
# basically, it only looks at the the word and not the sentence
ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(2, 2), min_df=1)
counts = ngram_vectorizer.fit_transform(['words', 'wprds'])

# prints true because each of these features exists in ngram_vectorizer
# note that there is a space in ' w' and 's ' (space padding)
print ngram_vectorizer.get_feature_names() == ([' w', 'ds', 'or', 'pr', 'rd', 's ', 'wo', 'wp'])
print

'''
prints the following array based on the content in the two terms 'words' and 'wprds'
(replaced SPACE with _ for visualisation on ' w' and 's ':

    _w      ds      or      pr      rd      s_    wo    wp        # column: features

[[  1       1       1       0       1       1     1     0   ]     # words
 [  1       1       0       1       1       1     0     1   ]]    # wprds

'''
print counts.toarray().astype(int)
print


ngram_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(5, 5), min_df=1)
print ngram_vectorizer.fit_transform(['jumpy fox'])  # .__repr__()
print

# note that also here space is added in all features except 'jumpy'
# I'm guessing it splits up all words that are longer then ngram_range, by adding space?
print ngram_vectorizer.get_feature_names() == ([' fox ', ' jump', 'jumpy', 'umpy '])
print


# The 'char' analyzer creates n-grams that span across words
ngram_vectorizer = CountVectorizer(analyzer='char', ngram_range=(5, 5), min_df=1)
print ngram_vectorizer.fit_transform(['jumpy fox'])
print

# char analyser includes the spaces
'''
based on code found in ```sklearn.feature_extraction.text.VectorizerMixin._char_ngrams```


    def _char_ngrams(self, text_document):
        """Tokenize text_document into a sequence of character n-grams"""
        # normalize white spaces
        text_document = self._white_spaces.sub(" ", text_document)

        text_len = len(text_document)
        ngrams = []
        min_n, max_n = self.ngram_range
        for n in xrange(min_n, min(max_n + 1, text_len + 1)):
            for i in xrange(text_len - n + 1):
                ngrams.append(text_document[i: i + n])
        return ngrams


it seems that it takes the first word(s) and retrieves all characters until it has passed the limit
    then it moves to the next char and retrieves until it has passed the limit
    repeat until no more words
'''
print ngram_vectorizer.get_feature_names() == (['jumpy', 'mpy f', 'py fo', 'umpy ', 'y fox'])
print
