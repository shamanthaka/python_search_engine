'''
Index structure:
    The Index class contains a list of IndexItems, stored in a dictionary type for easier access
    each IndexItem contains the term and a set of PostingItems
    each PostingItem contains a document ID and a list of positions that the term occurs
'''

import util
from cran import *
import doc
import sys     #File IO
import re      #Regex
import nltk
import collections
from nltk.tokenize import RegexpTokenizer
import codecs
import string
import pickle


class Posting:
    def __init__(self, docID):
        self.docID = docID
        self.positions = []

    def append(self, pos):
        self.positions.append(pos)

    def sort(self):
        ''' sort positions'''
        self.positions.sort()

    def merge(self, positions):
        self.positions.extend(positions)

    def term_freq(self):
        ''' return the term frequency in the document'''
        #ToDo
        tf = len(self.positions)
        return tf


class IndexItem:
    def __init__(self, term):
        self.term = term
        self.posting = {} #postings are stored in a python dict for easier index building
        self.sorted_postings= [] # may sort them by docID for easier query processing

    def add(self, docid, pos):
        ''' add a posting'''
        #if not self.posting.has_key(docid):  #for python 2
        if docid not in self.posting:
            self.posting[docid] = Posting(docid)
        self.posting[docid].append(pos)

    def sort(self):
        ''' sort by document ID for more efficient merging. For each document also sort the positions'''
        # ToDo
        for docid in self.posting.keys():
            self.posting[docid].sort()
            self.sorted_postings.extend(self.posting[docid].positions)


class InvertedIndex:

    def __init__(self):
        self.items = {}  #list of IndexItems
        self.nDocs = 0   #the number of indexed documents
        self.endmost = {}

    #tokenization for document(s) as tokens
    def tokenization(self, body):
        body = re.sub("[^a-z0-9]+", " ", body)
        tokens = nltk.tokenize.word_tokenize(body)
        return tokens

    #tokenization for document(s) as tokens
    def tokenization2(self, body):
        tokenizer = RegexpTokenizer(r'\w+')
        tokens = tokenizer.tokenize(body)
        return tokens


    # indexing a Document object
    ''' indexing a docuemnt, using the simple SPIMI algorithm, 
    but no need to store blocks due to the small collection
    we are handling. Using save/load the whole index instead'''
    def indexDoc(self, doc):
        # ToDo: indexing only title and body; use some functions defined in util.py
        # (1) convert to lower cases,
        # (2) remove stopwords,
        # (3) stemming

        doc_body = doc.body
        doc_docid = doc.docID
        tokens = self.tokenization(doc_body)
        reduced_list = []
        #storing stopwords in array
        stop_words = util.reading_stop_words()

        for word in tokens:
            if word not in stop_words:
                reduced_list.append(util.stemming(word))

        #print(reduced_list)
        pos_dic = self.make_word_postition_dictionary(reduced_list)
        self.make_word_document_dictionaray(pos_dic, doc_docid)


    #words of dictionaries
    #{term1:[pos1, pos3..], term2:[..], ..} dictionary is created
    def make_word_postition_dictionary(self, reducedList):
        pos_dict = {}  # Dictionary is created to store (terms, positions) in documents
        for position, word in enumerate(reducedList):
            if word in pos_dict.keys():
                pos_dict[word].append(position)
            else:
                pos_dict[word] = [position]

        #print(pos_dict)
        return pos_dict

    def make_word_document_dictionaray(self, pos_dic, doc_docid):
        # to assign the positions of a word to its respective docID
        for word in pos_dic:
            docdic = {}
            if word in self.items.keys():
                # if word is already in the item.keys(), then just add the docID
                pos = pos_dic[word]
                self.items[word].add(int(doc_docid), pos)
                for i in range(0, int(doc_docid) + 1, 1): #1401
                    # for every doc, add the docId and positions to the endmost dictionary
                    if i in self.items[word].posting.keys():
                        docdic[i] = self.items[word].posting[i].positions
                    self.endmost[word] = docdic
            else:
                # if word isn't present in items.keys(), create an index item, then add docId to positions
                obj_Index_item = IndexItem(word)
                pos = pos_dic[word]
                obj_Index_item.add(int(doc_docid), pos)
                self.items[word] = obj_Index_item
                for i in range(0, int(doc_docid) + 1, 1):  #1401
                    if i in self.items[word].posting.keys():
                        docdic[i] = self.items[word].posting[i].positions
                    self.endmost[word] = docdic

        print(self.endmost)
        print(len(self.endmost))

    #def sort(self):
      #  ''' sort all posting lists by docID'''
        #ToDo

    def find(self, term):
        return self.items[term]

    #def idf(self, term):
     #   ''' compute the inverted document frequency for a given term'''
       #ToDo: return the IDF of the term

    def save(self, filename):
        ''' save to disk'''
        # ToDo: using your preferred method to serialize/deserialize the index
        save_pickle = open(filename, "wb")
        pickle.dump(self.endmost, save_pickle)
        save_pickle.close()

        #print(self.endmost)


    def load(self, filename):
        ''' load from disk'''
        # ToDo
        load_pickle = open(filename, "rb")
        self.endmost = pickle.load(load_pickle)
        return self.endmost

    def sort(self):
        ''' sort all posting lists by docID'''
        dict_ = collections.OrderedDict(sorted(dict()))
        return dict_

    def find(self, term):
        return self.items[term]

    def numofterms(term, toks):
        return toks.count(term.lower())

    def tf(term, toks):
        return InvertedIndex.numofterms(term, toks) / float(len(toks))

    def df(term, tokslist):
        x = 0
        for toks in tokslist:
            if InvertedIndex.numofterms(term, toks) > 0:
                x += 1
        return x

    def idf(term, tokslist):
        return len(tokslist) / float(InvertedIndex.df(term, tokslist))

    def tfidf(term, toks, tokslist):
        return InvertedIndex.tf(term, toks) * InvertedIndex.idf(term, tokslist)



#test method which calls the inverted index class

def indexingCranfield(filee, filename):
    # ToDo: indexing the Cranfield dataset and save the index to a file
    # command line usage: "python index.py cran.all index_file.pickle"
    # the index is saved to index_file.pickle
    invertedIndex = InvertedIndex()
    # object for inverted index is created
    cf = CranFile(filee)
    # cran.all is uploaded from an cran.py object

    for doc in cf.docs:
        invertedIndex.indexDoc(doc)
        invertedIndex.sort()
        invertedIndex.save(filename)

    print("Done.")


#main method
if __name__ == '__main__':
    # corpusFile = str(sys.argv[1])  #"cran.all"
    # pickleFile = str(sys.argv[2])  #"index_file.pickle"
    corpusFile = "cran.all"
    pickleFile = "index_file.pickle"
    print("Indexing started. ")
    indexingCranfield(corpusFile, pickleFile)
    print("Index completed. ")


#To run the program from command line
# python index.py cran.all index_file.pickle

