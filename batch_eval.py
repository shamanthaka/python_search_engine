'''
a program for evaluating the quality of search algorithms using the vector model

it runs over all queries in query.text and get the top 10 results,
and then qrels.text is used to compute the NDCG metric

usage:
    python batch_eval.py index_file query.text qrels.text

    output is the average NDCG over all the queries

'''
import random
from functools import reduce
import scipy
import sys
from query import query
from cranqry import loadCranQry
from index import InvertedIndex
from cran import CranFile
from metrics import ndcg_score
import numpy as np


def eval(indexfilename, queryfilename, queryrefilename, numberofrandomqueries):

    # ToDo
    actual = []
   #
    if numberofrandomqueries > 225:
        raise Exception('please enter query count less than or equal to 225')
    qrys = loadCranQry("query.text")
    validqueries = []
    querycounter = 0
    for q in qrys:
        validqueries.append(int(q))

    loadiindex = InvertedIndex()
    loadiindex = loadiindex.load("index_file.pickle")
    #    print("index loaded")
    cf = CranFile('cran.all')
    #QueryProcessor.numberofresult =10
    #qp = QueryProcessor(qrys,loadiindex,cf.docs,10)
    queryRelevence = dict()
    for line in open(queryrefilename):

        fields = line.split(" ")
        fields[0] = '%0*d' % (3, int(fields[0]))
        if fields[0] in queryRelevence:
            # and let's extract the data:
            queryRelevence[fields[0]].append(fields[1])
        else:
            # create a new array in this slot
            queryRelevence[fields[0]] = [fields[1]]
    replacecounter = 0
    queryRelevenceUpdated = {}
    for k in queryRelevence:

        queryRelevenceUpdated['%0*d' % (3, int(validqueries[replacecounter]))] = queryRelevence.get(k)
        replacecounter = replacecounter + 1

  #  relevent = list(queryRelevence.keys())
   # relevent = list(map(int, relevent))
    #samplespace = np.intersect1d(relevent, validqueries)
    list_of_random_items = random.sample(validqueries, numberofrandomqueries)
    tempcounter2 = 0
    booleanndcg = []
    vectorndcg = []

    while tempcounter2 < numberofrandomqueries:

        list_of_random_items[tempcounter2] = '%0*d' % (3, int(list_of_random_items[tempcounter2]))
        print('query for which ndcg is calculated '+ str(list_of_random_items[tempcounter2]))
        y = str(list_of_random_items[tempcounter2])
        vectorresult = query(indexfilename, '1', queryfilename,  str(list_of_random_items[tempcounter2]), 10)
 #       vectorresult = ['573', '51', '944', '878', '12', '486', '875', '879', '746', '665']
 #       print(vectorresult)
        tempcounter = 0
        for z in vectorresult:

            if z in queryRelevenceUpdated[str(list_of_random_items[tempcounter2])]:
                vectorresult[tempcounter] = 1
            else:
                vectorresult[tempcounter] = 0

            tempcounter = tempcounter + 1
        #print(vectorresult)
        idealvectorresult = vectorresult.copy()
        idealvectorresult.sort(reverse=True)
        #print(idealvectorresult)
        if sum(idealvectorresult) == 0:
            ndcgscore = 0
        else:
            ndcgscore = ndcg_score(idealvectorresult,vectorresult)
       # print(ndcgscore)
        vectorndcg.append(ndcgscore)
        tempcounter3 = 0

        booleanqueryresult = query(indexfilename, '0', queryfilename,  str(list_of_random_items[tempcounter2]), 10)
        #booleanqueryresult = ['462','462','462','462','462','462','462','462','462']
        booleanquery = booleanqueryresult.copy()
        for g in booleanquery:

            if g in queryRelevenceUpdated[str(list_of_random_items[tempcounter2])]:
                booleanquery[tempcounter3] = 1
            else:
                booleanquery[tempcounter3] = 0

            tempcounter3 = tempcounter3 + 1
        #print(booleanquery)
        tempcounter4 = len(booleanquery)
        while tempcounter4 < 10:
            booleanquery.append(0)
            tempcounter4 = tempcounter4 + 1
        idealbooleanresult = []
        for i in range(0,10):
            if i < len(queryRelevenceUpdated[str(list_of_random_items[tempcounter2])]):
                idealbooleanresult.append(1)
            else:
                idealbooleanresult.append(0)

        idealbooleanresult.sort(reverse=True)
        if sum(booleanquery) == 0:
            ndcgscoreboolean = 0
        else:
            ndcgscoreboolean = ndcg_score(booleanquery,idealbooleanresult)
        booleanndcg.append(ndcgscoreboolean)
        tempcounter2 = tempcounter2 + 1
    print('P value for all the queries processed is:')
    print(scipy.stats.wilcoxon(vectorndcg, booleanndcg, zero_method='wilcox', correction=False))
    print('Done')

if __name__ == '__main__':
    #eval(str(sys.argv[1]), str(sys.argv[2]),str(sys.argv[3]),int(sys.argv[4]))
    # python batch_eval.py index_file query.text qrels.text 225

    eval("index_file.pickle", "query.text", "qrels.text", 225)
