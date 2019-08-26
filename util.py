
'''
   utility functions for processing terms

    shared by both indexing and query processing
'''

import nltk

def reading_stop_words():
    f = open("stopwords", 'r')
    # stopwords file is uploaded
    stopword = []
    for words in f:
        # words are split in document, stored in a list
        wrd = words.split()
        for e in wrd:
            # each word is appended
            stopword.append(e)

    f.close()
    return stopword


def isStopWord(word):
    ''' using the NLTK functions, return true/false'''

    # ToDo
    stopword = reading_stop_words()
    # print(stopword)
    if word in stopword:
        # if word is a stopword, it drops that word in reduced list
        return False
    else:
        return True





def stemming(word):
    ''' return the stem, using a NLTK stemmer. check the project description for installing and using it'''

    # ToDo
    stemmer = nltk.PorterStemmer()
    st = stemmer.stem(word)
    # stemmer stem all words to a root word
    return st


if __name__ == '__main__':
    word = "goes"

    print(stemming(word))

