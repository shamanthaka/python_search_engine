
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

    def __init__(self, query, index, collection):
        ''' index is the inverted index; collection is the document collection'''
        self.raw_query = query
        self.index = index
        self.docs = collection

    def __init__(self, query, index):
        ''' index is the inverted index; collection is the document collection'''
        self.raw_query = query
        self.index = index
        self.query = []



    def convertFromMap(self, mv):
        mapped = []  # vecRanking[doc+1]=cos(tfs,df)
        iter_ = list(mv.values())
        for i in iter_:
            mapped.append(i.text)
        return mapped

    def preprocessing(self,qid):
        ''' apply the same preprocessing steps used by indexing,
            also use the provided spelling corrector. Note that
            spelling corrector should be applied before stopword
            removal and stemming (why?)'''

        #ToDo: return a list of terms

        qbody = self.raw_query
        cqObj = CranFile('query.text')
        qbody = qbody.get(qid)#self.convertFromMap(qbody)   #self.docs
        #print(qbody)
        try:
            qbody = re.sub("[^a-z0-9]+", " ", str(qbody.text))
        except Exception:
            print("Query ID which is not having text: ", qid)
            raise

        reduced = nltk.tokenize.word_tokenize(qbody)

        '''for words in reduced:
            # reduced terms are passed through stopwords and stemming in util
            if util.isStopWord(words):
                self.query.append(util.stemming(words).lower())
        # normalized terms are stored in reducedList Dictionary
        print("1...", self.query)'''

        correctedwords = [correction(word) for word in reduced]
        lowercasewords = [word.lower() for word in correctedwords]
        notstopwords = []
        for word in lowercasewords:
            if util.isStopWord(word):
                notstopwords.append(util.stemming(word))
        if len(notstopwords) > 0:
            self.query.append(notstopwords)

        print("1...", self.query)


    def booleanQuery(self):
        ''' boolean query processing; note that a query like "A B C" is transformed to "A AND B AND C" for retrieving posting lists and merge them'''
        #ToDo: return a list of docIDs
        #print("1...", self.query)

        iiObj = InvertedIndex()
        loadedindex = iiObj.load('index_file.pickle')
        qv = set(self.query[0])
        retrievedQueries = [] #set(loadedindex[qv].keys()) #set(loadedindex[self.query[0]].posting.keys())
        setted = set()
        #print(retrievedQueries)
        for term in self.query[0]:
            try:
                if not setted:
                    setted = set((list(loadedindex[term].keys())))
                else:
                    setted = setted.intersection(list(loadedindex[term].keys()))
                #print(setted)
            except Exception:
                pass
            #retrievedQueries = retrievedQueries.intersection(loadedindex[term].keys())
        print("Document id's which is having the query terms", setted)



    def vectorQuery(self, k):
        ''' vector query processing, using the cosine similarity. '''
        #ToDo: return top k pairs of (docID, similarity), ranked by their cosine similarity with the query in the descending order
        # You can use term frequency or TFIDF to construct the vectors

        tfs = []
        # term frequncies for every term
        #df = []
        # document frequency list to store it for each term
        vecRanking = {}
        # a dictionary to store the cosine similarities
        terms = self.index.keys()
        # the terms in dictionary for comparision
        terms = set(terms).union(set(self.query[0])) #set(terms).union(self.query)
        for term in terms:
            # for every term, tf is stored in ranking dictionary by appending
            tfs.append(tf(term, self.query))
        # print sum(tfs)
        #for doc in range(self.index.numOfTerms):
        for doc in range(len(self.index)):
            # for every term, df is stored to calculate the idf for ranking
            #df = []
            i = 0
            for term in terms:
                # tfidf of each term is stored in the dictionary
                #i = self.index.tf(term, str(doc + 1))
                i = tf(term, str(doc + 1))
                x = self.doc_freq(term)
                vecRanking[doc + 1] = cos(tfs, x) #cos(tfs, df)
            # ranking is stored in dictionary
            topk(k, vecRanking)



    def doc_freq(self, term):
        if term in self.index.keys():
            return len(self.index[term].keys())
        else:
            return 0


def mytest():
    ''' test your code thoroughly. put the testing cases here'''
    print('Pass')

def tfOfQuery(term,query): #num of times a word appearing is query is returned
    i=0
    for word in query:
        if word==term:
            i+=1
    return i

def numofterms(term, toks): #num of tokens are counted
    return toks.count(term.lower())

def tf(term, toks): #number of terms per total tokens
    return numofterms(term, toks) / float(len(toks))

def df(term, tokslist): #returns df of a term
    x = 0
    for toks in tokslist:
        if numofterms(term, toks) > 0:
            x += 1
    return x

def idf(term, tokslist): #idf is calculated by n/df
    return len(tokslist) / float(df(term, tokslist))

def tfidf(term, toks, tokslist): # tfidf is calculated  for cosine similarity
    return tf(term, toks) * idf(term, tokslist)

def cos(query,doc): #cosine similarity is determined by dot product of query and document
    return dot(query,doc)/(norm(query)*norm(doc))

def topk(k,itemsDic): #Only top K=3 are retrieved for vector model
    items=sorted(itemsDic.items())
    for i in range(k):
        print(items[i])#op

def query(processing_algorithm,query,index): #args for command line
    ''' the main query processing program, using QueryProcessor'''

    # ToDo: the commandline usage: "echo query_string | python query.py index_file.pickle processing_algorithm"
    # processing_algorithm: 0 for booleanQuery and 1 for vectorQuery
    # for booleanQuery, the program will print the total number of documents and the list of docuement IDs
    # for vectorQuery, the program will output the top 3 most similar documents
    qp = QueryProcessor(query, index)
    qp.preprocessing()
    if (processing_algorithm == 0):
        qp.booleanQuery()
    else:
        qp.vectorQuery(3)

def getDoc(qrys):
    myDoc = []
    for doc in qrys:
        myDoc.append(qrys[doc].text)
    return myDoc

if __name__ == '__main__':
    qrys = loadCranQry('query.text') #loadCranQry('query.text')
    # query.text is retrieved from loadCranQry
    invertedInd = InvertedIndex()
    index = invertedInd.load("index_file.pickle")  # sys.argv[1]
    # arg 1 in command line is pickle file
    # qr = QueryProcessor(query, index)
    coll = getDoc(qrys)
    #qr = QueryProcessor(qrys, index, coll)
    qr = QueryProcessor(qrys, index)
    qr.preprocessing('009')

    alg = '0'  # sys.argv[2]
    # arg 2 is 0 for bool, 1 for vector
    if (alg == '0'):
        qr.booleanQuery()
    else:
        qr.vectorQuery(3)
    # Shamanthaka added below code
    qid = 1  # sys.argv[4]
    #query = qrys[qid].text
    # query=sys.argv[3]
    # arg 3 for getting query.text
    # qid=sys.argv[4]
    # arg 4 for query id
