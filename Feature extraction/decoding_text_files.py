# http://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction

import chardet
from sklearn.feature_extraction.text import CountVectorizer

# printout for space before start
print

# http://scikit-learn.org/stable/modules/feature_extraction.html#decoding-text-files

print " ===== Decoding-text-files ===== "
print

text1 = b"Sei mir gegr\xc3\xbc\xc3\x9ft mein Sauerkraut"

text2 = b"holdselig sind deine Ger\xfcche"

text3 = b"\xff\xfeA\x00u\x00f\x00 " \
        b"\x00F\x00l\x00\xfc\x00g\x00e\x00l\x00n\x00 " \
        b"\x00d\x00e\x00s\x00 " \
        b"\x00G\x00e\x00s\x00a\x00n\x00g\x00e\x00s\x00,\x00 " \
        b"\x00H\x00e\x00r\x00z\x00l\x00i\x00e\x00b\x00c\x00h\x00e\x00n\x00,\x00 " \
        b"\x00t\x00r\x00a\x00g\x00 " \
        b"\x00i\x00c\x00h\x00 " \
        b"\x00d\x00i\x00c\x00h\x00 " \
        b"\x00f\x00o\x00r\x00t\x00"

decoded = [x.decode(chardet.detect(x)['encoding'])
           for x in (text1, text2, text3)]

v = CountVectorizer().fit(decoded).vocabulary_

for term in v:
    print(v)
