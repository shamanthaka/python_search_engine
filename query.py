'''
query processing

'''

import nltk
from nltk.tokenize import RegexpTokenizer
from norvig_spell import correction
import util
from numpy import dot
from numpy.linalg import norm
from operator import itemgetter
from cranqry import loadCranQry
from index import *
from time import sleep
import sys
import re
from cran import *

class QueryProcessor:

    def __init__(self, query, index, invertedIndObj):
        ''' index is the inverted index; collection is the document collection'''
        self.raw_query = query
        self.index = index
        self.iiObj = invertedIndObj

    def __init__(self, query, index):
        ''' index is the inverted index; collection is the document collection'''
        self.raw_query = query
        self.index = index
        self.query = []

    def preprocessing(self, qid):
        ''' apply the same preprocessing steps used by indexing,
            also use the provided spelling corrector. Note that
            spelling corrector should be applied before stopword
            removal and stemming (why?)'''

        qbody = self.raw_query
        qbody = qbody.get(qid)  # self.convertFromMap(qbody)   #self.docs
        print("Below is the query: ")
        print(qbody.text)
        #in query.text, there are no 005, etc due to this exception will
        # be thrown.
        try:
            qbody = re.sub("[^a-z0-9]+", " ", str(qbody.text))
        except Exception:
            print("Query ID which is not having text: ", qid)
            raise

        tokens = nltk.tokenize.word_tokenize(qbody)

        corrected_tokens = [correction(word) for word in tokens] #spell check
        converted_tokens = [word.lower() for word in corrected_tokens]
        #below query will not have stop words
        clean_query = []

        for word in converted_tokens:  #removing stop words
            if util.isStopWord(word):
                clean_query.append(util.stemming(word))
        if len(clean_query) > 0:
            self.query.append(clean_query)

        print("Query after spell check and  removing the stop words: ", self.query)


    def booleanQuery(self):
        ''' boolean query processing; note that a query like "A B C" is transformed to "A AND B AND C" for retrieving posting lists and merge them'''
        # ToDo: return a list of docIDs
        inverted_index = InvertedIndex()
        #loaded_index = inverted_index.load(str(self.index))
        #above one is not working, but when index file is hardcoded which is working fine
        loaded_index = inverted_index.load('index_file.pickle')

        setted = set()

        for term in self.query[0]:
            try:
                if not setted:
                    setted = set((list(loaded_index[term].keys())))
                else:
                    setted = setted.intersection(list(loaded_index[term].keys()))
                #print(setted)
            except Exception:
                pass

        print("BooleanQuery: Document id's which is having the query terms", setted)


    def doc_freq(self, term):
        if term in self.index.keys():
            return len(self.index[term].keys())
        else:
            return 0

    def vectorQuery(self, k):
        terms = self.index
        # the terms in dictionary for comparision

        results = []
        for word in self.query[0]:
            try:
                files = [filename for filename in list(terms[word].keys())]
                results = self.rankResults(files, word)

            except Exception:
                pass

        if len(results) > 2:
            topk(k, results)
        else:
            print("Vector Query : Document id's which is having the query terms ", results)

    #new methods starts

    def total_number_of_docs(self):
        my_list = list(self.index.values())
        my_set = set()
        for v in range(0, my_list+1):
            my_set.add(my_list[v].key())
        return len(my_set)

    def term_frequency(self, term):
        if term in self.index.keys():
            return len(self.index[term])
        else:
            return 0

    def document_frequency(self, term):  # returns df of a term
        if term in self.index[term].keys():
            return len(self.index[term].keys())
        else:
            return 0

    def inverse_document_frequency(self, term):  # returns df of a term
        if term in self.index[term].keys():
            return self.total_number_of_docs()/self.document_frequency(term)
        else:
            return 0

    def generateScores(self, term, document):
        return self.term_frequency(term) * self.inverse_document_frequency(term)

    '''finding the score of each word in corpus'''
    def make_vectors(self, documents):
        inverted_index = InvertedIndex()
        loaded_index = inverted_index.load('index_file.pickle')
        vecs = {}
        for doc in documents:
            doc_vec = [0] * len(loaded_index.keys())  # array object with zero filled will be created.
            for ind, term in enumerate(loaded_index.keys()):  # enumerating each term in indexed documents
                try:
                    doc_vec[ind] = self.generateScores(term, doc)  # storing each term score => tf*idf
                except Exception:
                    pass
            vecs[doc] = doc_vec
        return vecs

    def dotProduct(self, doc1, doc2):
        if len(doc1) != len(doc2):
            return 0
        return sum([x*y for x, y in zip(doc1, doc2)])


    '''gives the documenst(s) for matched term queries.'''
    def rankResults(self, resultDocs, word):
        vectors = self.make_vectors(resultDocs)
        queryVec = self.query_vec(word)
        results = [[self.dotProduct(vectors[result], queryVec), result] for result in resultDocs]
        results.sort(key=lambda x: x[0])
        results = [x[1] for x in results]
        return results

    def query_vec(self, query):
        inverted_index = InvertedIndex()
        loaded_index = inverted_index.load('index_file.pickle')
        query1s = query.split()
        queryVec = [0] * len(query1s)  # array object with zero filled will be created.
        index = 0
        finalScore = []
        for ind, word in enumerate(query1s):
            queryVec[index] = self.query_freq(word, query)
            index += 1
        try:
            # getting the idf for each term in corpus
            queryidf = [self.index.idf[word] for word in self.index.getUniques()]
            # finding the magnitude of query terms,those are in
            # indexed documents.
            magnitude = pow(sum(map(lambda x: x ** 2, queryVec)), .5)
            # finding query terms in indexed document terms
            freq = self.term_freq(loaded_index.keys(), query)
            tf = [x / magnitude for x in freq]
            # score
            finalScore = [tf[i] * queryidf[i] for i in range(len(loaded_index.keys()))]
        except Exception:
            pass
        return finalScore

    '''in given search query, finding the frequency of given words by comparing with indexed terms'''
    def query_freq(self, term, query):
        count = 0
        for word in query.split():
            if word == term:
                count += 1
        return count

    '''finding the all indexed terms frequency with the query terms, if term is having
    frequency means query terms exists in document(s).'''
    def term_freq(self, terms, query):
        temp = [0] * len(terms)  # array object with zero filled will be created.
        for i, term in enumerate(terms):
            temp[i] = self.query_freq(term, query)
        return temp

    # new methods ends


def topk(k,itemsDic): #Only top K=3 are retrieved for vector model
    #items = sorted(itemsDic.items())
    items = sorted(itemsDic)
    for i in range(k):
        print("document id's",items[i]) #top

def numofterms(term, toks): #num of tokens are counted
    return toks.count(term.lower())

def tf(term, toks): #number of terms per total tokens
    return numofterms(term, toks) / float(len(toks))


def cos(query,doc): #cosine similarity is determined by dot product of query and document
    return dot(query,doc)/(norm(query)*norm(doc))

def getDoc(qrys):
    myDoc = []
    for doc in qrys:
        myDoc.append(qrys[doc].text)
    return myDoc

def query(processing_algorithm, query, index, queryId): #args for command line
    ''' the main query processing program, using QueryProcessor'''

    # ToDo: the commandline usage: "echo query_string | python query.py index_file.pickle processing_algorithm"
    # processing_algorithm: 0 for booleanQuery and 1 for vectorQuery
    # for booleanQuery, the program will print the total number of documents and the list of docuement IDs
    # for vectorQuery, the program will output the top 3 most similar documents
    qp = QueryProcessor(query, index)
    qp.preprocessing(queryId)
    if (processing_algorithm == 0):
        qp.booleanQuery()
    else:
        qp.vectorQuery(3)

if __name__ == '__main__':

    #index_file = str(sys.argv[1]) #index_file.pickle
    #algo = int(sys.argv[2]) # 0
    #query_text = str(sys.argv[3]) #query.text
    #queryId = str(sys.argv[4]) # '009'

    index_file = "index_file.pickle"
    algo = 0
    query_text = "query.text"
    queryId = '009'
    qrys = loadCranQry(query_text)
    invertedInd = InvertedIndex()
    #loading the indexed doucment file

    index = invertedInd.load(index_file)

    #no need to use below one
    #coll = getDoc(qrys)

    #query(alogo, qrys, index, queryId)

    qr = QueryProcessor(qrys, index)
    qr.preprocessing(queryId)
    #There are two types of queries
    # 1. is booleanQuery, 2. vectoryQuery

    # if algo = 0, which is booleanQuery
    # if algo other than 0 which is vectoryQuery

    #algo = 1
    if algo == 0:
        qr.booleanQuery()
    else:
        qr.vectorQuery(3)


# code should be ran in the following way
# python query.py index_file.pickle 0 query.text 009
#python query.py index_file.pickle 1 query.text 009


