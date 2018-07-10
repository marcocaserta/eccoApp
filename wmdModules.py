"""
/***************************************************************************
 *   copyright (C) 2018 by Marco Caserta                                   *
 *   marco dot caserta at ie dot edu                                       *
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 2 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 *   This program is distributed in the hope that it will be useful,       *
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of        *
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         *
 *   GNU General Public License for more details.                          *
 *                                                                         *
 *   You should have received a copy of the GNU General Public License     *
 *   along with this program; if not, write to the                         *
 *   Free Software Foundation, Inc.,                                       *
 *   59 Temple Place - Suite 330, Boston, MA  02111-1307, USA.             *
 ***************************************************************************/

 Algorithm for the Computation of Distances between documents.

 Author: Marco Caserta (marco dot caserta at ie dot edu)
 Started : 02.02.2018
 Updated : 27.02.2018 -> introduction of a cycle
           09.03.2018 -> shortlist creation using Doc2Vec (per year)
 Ended   :

 Command line options (see parseCommandLine):
 -s name of file containing the target sentence, i.e., the query

NOTE: This code is based on the tutorial from:

https://github.com/RaRe-Technologies/gensim/blob/develop/docs/notebooks/doc2vec-lee.ipynb

Branch ECCO:

The idea is to work on the documents of the ECCO dataset at a sentence level.
Given a "reference sentence," we want to find the list of sentences closer to
the target in a given pool of sentences. Note that the dataset must be
organized at a sentence level.

As of today, this can be achieved in a number of ways. We explore:
- doc2vec, which provides an embeddding for each document
- wmd, which transforms a sentence into another solving a transportation pbr

We observed that the doc2vec query is pretty fast, with the advantage that the
actual doc2vec model can be precomputed. Once the doc2vec model is available,
computing the distance between a query sentence and any of the documents in the
corpus is quite fast. In contrast, wmd is computationally intensive. Therefore,
the idea is to use both of them in sequence:
1. use doc2vec on the premium set of sentences for a given time period retrieve
the N most similar sentences (N can be quite large here). Let us call this list
the "shortlist".
2. use the N sentences retrieved in the previous step (i.e., the shortlist) as
input to the wmd algorithm. Since N << number of premium sentences << original
number of sentences (i.e., before applying the premium list filter), wmd is
able to provide an answer in an amount of time which depends on N.

Why not just using doc2vec? It seems the similarity measure produced by wmd
outperforms the one given by doc2vec. In addition, it allows to use any
embedding (not just the one provided by word2vec) to assign a numerical vector
to each word of the document.

The high-level structure of the algorithm is thus as follows:
for each year in the period:
    read premium lists for that year and the corresponding doc2vec model
    get the N/(nr.years) best sentences w.r.t. the query sentence
    store the shortlist into the docs and corpus structure
At this point, docs and corpus contain the shortlists built over all the years
of the period under analysis. Pass docs to wmd and get the top list (i.e., the
very limited, human readable, set of sentences that are the closest to the
query.)

"""
import sys
from gensim.models import Word2Vec
from os import path
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
from timeit import default_timer as timer
import csv
import cplex
from math import inf as infinity
from scipy.spatial.distance import cdist
import bisect
from multiprocessing import Pool, cpu_count
from itertools import islice, repeat
from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords

stop_words = stopwords.words("english")

prefix                 = path.expanduser("/home/mcaserta/research/nlp/data")
ecco_models_folder     = "ecco_models/"
vocab_folder           = "google_vocab/"
modelnameW2V           = "word2vec.model.96-00.all"
premiumDocsXRowBase    = "/home/mcaserta/research/nlp/ecco_code/preproc/premiumDocsXRow.csv"
premiumCorpusXRowBase  = "/home/mcaserta/research/nlp/ecco_code/preproc/premiumCorpusXRow.csv"

solutionDf         = "results/solution.csv" #  storing the current best query result
queryDf            = "results/query.csv" #  storing the current best query result

class Sentence:
    def __init__(self):
        self.tokens = []
        self.z      = -1
        self.id     = -1
#        self.year   = -1 # not used, verify
class Top():
    """
    Data structure used to store the top N sentences matching a given target
    sentence.
    We store both the full sentence (untokenized) and the tokenized and
    preprocessed sentence. We also store the previous and next sentences.
    """
    def __init__(self, nTop):
        self.score     = [-1]*nTop
        self.idnr      = [""]*nTop
        self.year      = [""]*nTop
        self.tokenSent = [[]]*nTop
        self.sent      = [""]*nTop
        self.prevSent  = [""]*nTop
        self.nextSent  = [""]*nTop
        self.idx       = [-1]*nTop
        self.best      = infinity
        self.star      = " "
        self.plus      = " "
            
    def getSortedList(self):
        return [self.score[i] for i in self.idx]


def docPreprocessing(doc, modelWord2Vec):

    doc = word_tokenize(doc)
    doc = [w.lower() for w in doc if w.lower() not in stop_words] # remove stopwords
    passing = 1
    for w in doc:
        if w not in modelWord2Vec.wv.vocab:
            print("[ERROR] Word ", w, " of target sentence is not in vocabulary.")
            passing = 0
        if w.isalpha() == False:
            print("[ERROR] Word ", w, " is not in the alphabet.")
            passing = 0

        if passing == 0:
            return -1

    doc = [w for w in doc if w.isalpha() and w in modelWord2Vec.wv.vocab] # remove numbers and pkt
    #  doc = [w for w in doc if w.isalpha() and w in vocab_dict] # remove numbers and pkt

    return doc

def setupTarget(target, model):
    # setup target (invariant over cycle)
    nWords = len(target)
    demand = {key:0 for key in target}
    for w in target:
        demand[w] += 1
    nD     = len(demand)
    dem    = [val/nWords for val in demand.values()]
    D      = model.wv[demand.keys()]
    
    return nD, dem, D

def solveTransport2(matrixC, cap, dem, nS, nD):
    """
    Solve transportation problem as an LP.
    This is my implementation of the WMD.
    """
    
    cpx   = cplex.Cplex()
    x_ilo = []
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    for i in range(nS):
        x_ilo.append([])
        for j in range(nD):
            x_ilo[i].append(cpx.variables.get_num())
            #  varName = "x." + str(i) + "." + str(j)
            cpx.variables.add(obj   = [float(matrixC[i][j])],
                              lb    = [0.0])
                              #  names = [varName])
    # capacity constraint
    for i in range(nS):
        index = [x_ilo[i][j] for j in range(nD)]
        #value = [1.0]*nD
        capacity_constraint = cplex.SparsePair(ind=index, val=[1.0]*nD)
        #capacity_constraint = cplex.SparsePair(ind=index, val=np.ones(nD))
        cpx.linear_constraints.add(lin_expr = [capacity_constraint],
                                   senses   = ["L"],
                                   rhs      = [cap[i]])

    # demand constraints
    for j in range(nD):
        index = [x_ilo[i][j] for i in range(nS)]
        #value = [1.0]*nS
        demand_constraint = cplex.SparsePair(ind=index, val=[1.0]*nS)
        cpx.linear_constraints.add(lin_expr = [demand_constraint],
                                   senses   = ["G"],
                                   rhs      = [dem[j]])
    cpx.parameters.simplex.display.set(0)
    cpx.solve()

    return cpx.solution.get_objective_value()


def wmdTransportNoLB(model, source, D, nD, dem):
        nWords   = len(source)
        capacity = {key:0 for key in source}
        #  print("SOURCE = ", source)
        for w in source:
            capacity[w] += 1

        nS       = len(capacity)
        cap      = [val/nWords for val in capacity.values()]
        try:
            S        = model.wv[capacity.keys()]
        except:
            return -1
        dd       = cdist(S,D)

        # solve transportation problem
        z = solveTransport2(dd, cap, dem, nS, nD)

        return z
    
def wmdTransport(model, sent, D, nD, dem, lastScore):
        source = sent.tokens
        nWords   = len(source)
        capacity = {key:0 for key in source}
        #print("SOURCE = ", source)
        for w in source:
            capacity[w] += 1

        nS       = len(capacity)
        cap      = [val/nWords for val in capacity.values()]
        try:
            S        = model.wv[capacity.keys()]
        except: #this might occur when the word is not in the dictionary
            return -1
        dd       = cdist(S,D)
        
        # compute lower bounds for fathoming
        lb = np.dot(dd.min(axis=1), cap)
        if lb >= lastScore:
            #sent.z=-1
            return sent
        
        lb = np.dot(dd.min(axis=0), dem)
        if lb > lastScore:
            #sent.z=-1
            return sent

        # if not pruned, solve transportation problem
        sent.z = solveTransport2(dd, cap, dem, nS, nD)
        
        return sent


def loadW2V():
    fullpath = path.join(prefix, ecco_models_folder)
    fullname = fullpath + modelnameW2V
    mW2V = Word2Vec.load(fullname)
    mW2V.init_sims(replace=True)
    return mW2V

def populateFirstnTopForbidden(model, tops, year, nPopulate, readerDocs, D, nD, dem, nTop, forbiddenWords):
    '''
    This is NOT a simple copy of the \c populateFirstnTop module. The problem here
    is that, since some sentences will be skipped, the id of the sentences requires
    special care.

    The counter \c absCounter is define to account for the skipped sentences.
    '''
    # populate empty list
    i = 0
    absCounter = -1
    for k, source in islice(enumerate(readerDocs), 0, nPopulate):
        absCounter += 1
        # discard the sentence if it contains a word in the forbidden list
        found = False
        for w in source:
            if w in forbiddenWords:
                found = True
                break
        if found:
            continue
        z = wmdTransportNoLB(mW2V, source, D, nD, dem)
        tops.score[i]     = z
        tops.tokenSent[i] = source        
        tops.idx[i]       = i
        tops.idnr[i]      = absCounter 
        tops.year[i]      = year
        i += 1

        if i==nTop:
            break

    # sort index w.r.t. score
    tops.idx = [x for _,x in sorted(zip(tops.score,tops.idx))]
    return tops, absCounter

def populateFirstnTop(model, tops, year, nPopulate, readerDocs, D, nD, dem):
    # populate empty list
    i = 0
    for source in islice(readerDocs, 0, nPopulate):
        #print("sentence for process ", multiprocessing.current_process().name, " = ", source)
        z = wmdTransportNoLB(mW2V, source, D, nD, dem)
        tops.score[i]     = z
        tops.tokenSent[i] = source        
        tops.idx[i]       = i
        tops.idnr[i]      = i
        tops.year[i]      = year
        i += 1
    # sort index w.r.t. score
    tops.idx = [x for _,x in sorted(zip(tops.score,tops.idx))]
    return tops
        
def sortedInsertion(tops, sent, year, nTop):
    last = tops.idx[-1]
    ss = [tops.score[i] for i in tops.idx]
    ss = tops.getSortedList()
    pos  = bisect.bisect(ss, sent.z)
    # add here information of the newly inserted item
    # (all the other fields do not need to be changed
    #  just move the idx value)
    tops.score[last]      = sent.z
    tops.idnr[last]       = sent.id
    tops.tokenSent[last]  = sent.tokens
    tops.year[last]       = year
    for i in range(nTop-2, pos-1, -1):
        tops.idx[i+1]       = tops.idx[i]
    tops.idx[pos] = last
    #print("Is sorted ? ", tops.getSortedList() == sorted(tops.score))
 
    return tops

def updateBest(tops, year, totSents, totPruned, i, totRead):
    if tops.score[tops.idx[0]] < tops.best:
        tops.best = tops.score[tops.idx[0]]
        tops.star = "*"
    else:
        tops.star = " "
    print("[{0:5.0f} secs. -{1}- {2:7d}/{3}] z* = {4:5.3f}\t [Fathomed : {5:7d} ({6:5.3f})] {7}{8}".format(timer()-start, year, i, totSents, tops.best, totPruned, totPruned/totRead, tops.plus, tops.star))
    tops.plus = " "
    return tops

def printTops(tops):
    for i,id in enumerate(tops.idx):
        print("[{0:4d}--{1}.{2:8d}] {3:5.3f} :: {4}".format(i, tops.year[id], tops.idnr[id], tops.score[id], tops.tokenSent[id]))

def wmdParallel(nCores, batch, printStep, years, tops, nTop, totPruned,
totRead, totSentences, targetTokenized, fs, stepFs):

    nD, dem, D = setupTarget(targetTokenized, mW2V)

    p     = Pool(nCores)
    for year in years:
        premiumDocsXRow   = premiumDocsXRowBase + "." + year
        premiumCorpusXRow = premiumCorpusXRowBase + "." + year
        totSents = totSentences[year]
        #  totSents = 1000
        print("Loading sentences for year {0} [{1:7d} sentences]".format(year,totSents))

        with open(premiumDocsXRow, "r") as fDocs:
            readerDocs   = csv.reader(fDocs)
            nPopulate = 0
            if year == years[0]:
                nPopulate = min(nTop, totSents)
                tops = populateFirstnTop(mW2V, tops, year, nPopulate, readerDocs, D, nD, dem)
                tops = updateBest(tops, year, totSents, totPruned, nTop, nTop)

            batches = batch*nCores
            sources = []
            #  initIndex = nPopulate # to recover the index of each sentence

            for i,source in islice(enumerate(readerDocs), 0, totSents):
                sent = Sentence()
                sent.tokens = source
                sent.id = nPopulate+i
                sources.append(sent)

                if i % printStep == 0:
                    totRead += printStep
                    tops = updateBest(tops, year, totSents, totPruned, i, totRead)
                    fs.value += stepFs*printStep

                if (i+1) % batches == 0 or i==totSents-1:
                    # divide and send
                    nSent = len(sources)
                    sets = np.array_split(np.arange(0,nSent,1), nCores)
                    slicedSources = [[ sources[s] for s in myset] for myset in sets]
                    results = p.starmap(getWMD,zip(slicedSources, repeat(tops.score[tops.idx[-1]]), repeat(D), repeat(nD), repeat(dem)) )

                    for cc in range(nCores):
                        for sent in results[cc]:
                            if sent.z == -1:
                                totPruned += 1
                            else:
                                if sent.z < tops.score[tops.idx[-1]]:
                                    tops.plus = "+"
                                    sortedInsertion(tops, sent, year, nTop)
                    sources = []
                    #  initIndex += nSent

            totRead += printStep
            tops = updateBest(tops, year, totSents, totPruned, i+1, totRead)
 
    p.close()       
        
    return tops, totPruned, totRead

def wmdParallelForbidden(nCores, batch, printStep, years, tops, nTop,
totPruned, totRead, totSentences, targetTokenized, forbiddenWords, fs, stepFs):
    '''
    Just like in the case of the populateFirstnTopForbidden module, here we also
    have a problem with resetting the sentence id. Note that the parameter
    \c nPopulate is re-defined inside the \c populateFirstnTopForbidden module,
    since we do not really know how many rows of the file are to be rad to reach
    the desired value of nTop (remember that some sentences might be skipped due to
    the forbidden words in them.)
    '''

    nD, dem, D = setupTarget(targetTokenized, mW2V)

    p     = Pool(nCores)
    for year in years:
        premiumDocsXRow   = premiumDocsXRowBase + "." + year
        premiumCorpusXRow = premiumCorpusXRowBase + "." + year
        totSents = totSentences[year]
        #  totSents = 10000
        print("Loading sentences for year {0} [{1:7d} sentences]".format(year,totSents))

        with open(premiumDocsXRow, "r") as fDocs:
            readerDocs   = csv.reader(fDocs)
            nPopulate = 0
            if year == years[0]:
                nPopulate = min(nTop,totSents)
                tops, nPopulate = populateFirstnTopForbidden(mW2V, tops, year,
                totSents, readerDocs, D, nD, dem, nTop, forbiddenWords)
                tops = updateBest(tops, year, totSents, totPruned, nTop, nTop)
                nPopulate += 1 # sentences in nTop go from 0 to nTop-1. Increase by 1

            batches = batch*nCores
            sources = []
            #  initIndex = nPopulate # to recover the index of each sentence
            

            for i,source in islice(enumerate(readerDocs), 0, totSents):

                # discard the sentence if it contains a word in the forbidden list
                found = False
                for w in source:
                    if w in forbiddenWords:
                        found = True
                        break
                if found:
                    continue

                sent = Sentence()
                sent.tokens = source
                # shift counter to account for first nTop sentences !!!
                sent.id = nPopulate + i
                sources.append(sent)

                if i % printStep == 0:
                    totRead += printStep
                    tops = updateBest(tops, year, totSents, totPruned, i, totRead)
                    fs.value += stepFs*printStep

                if (i+1) % batches == 0 or i==totSents-1:
                    # divide and send
                    nSent = len(sources)
                    sets = np.array_split(np.arange(0,nSent,1), nCores)
                    slicedSources = [[ sources[s] for s in myset] for myset in sets]
                    results = p.starmap(getWMD,zip(slicedSources, repeat(tops.score[tops.idx[-1]]), repeat(D), repeat(nD), repeat(dem)) )

                    for cc in range(nCores):
                        for sent in results[cc]:
                            if sent.z == -1:
                                totPruned += 1
                            else:
                                if sent.z < tops.score[tops.idx[-1]]:
                                    tops.plus = "+"
                                    sortedInsertion(tops, sent, year, nTop)
                    sources = []
                    #  initIndex += nSent

            totRead += printStep
            tops = updateBest(tops, year, totSents, totPruned, i+1, totRead)
 
    p.close()       
        
    return tops, totPruned, totRead

def getWMD(sources, lastScore, D, nD, dem):
    return [wmdTransport(mW2V, sent, D, nD, dem, lastScore) for sent in sources]

def retrieveSentences(topsOut, nTop):
    # store it the "right way", no more pointers
    df = pd.DataFrame({
        'year'  : [topsOut.year[i] for i in topsOut.idx],
        'idnr'  : [topsOut.idnr[i] for i in topsOut.idx],
        'score' : [topsOut.score[i] for i in topsOut.idx],
        'tokenSent' : [topsOut.tokenSent[i] for i in topsOut.idx],
        'sent'      : ['']*nTop,
        'prevSent'  : ['']*nTop,
        'nextSent'  : ['']*nTop        
    })
    
    df = df.sort_values(['year','idnr'])
    #print("SORTED DF = ", df)
    groups = df.groupby('year')
    previousIndex = -1
    # check missing: if sentence is first or last of the file, i.e., idnr=0 or nSent
    for year, grouped_df in groups:
        premiumCorpusXRow = premiumCorpusXRowBase + "." + year
        with open(premiumCorpusXRow, "r") as fCorpus:
            readerCorpus   = csv.reader(fCorpus)

            currentPos = 1 # position of the header in file 
            # REM: after reading line e.g., 7, the header is positioned on line 8
            for index,el in grouped_df.iterrows():
                
                position    = el.idnr-currentPos
                currentPos  = el.idnr + 3 # each time, we move three steps
                if position == -2:  # e.g., 10 and 11
                    df.prevSent[index] = df.sent[previousIndex]
                    df.sent[index]     = df.nextSent[previousIndex]
                    df.nextSent[index] = fCorpus.readline()
                elif position == -1: # e.g., 10 and 12
                    df.prevSent[index] = df.nextSent[previousIndex]
                    df.sent[index]     = fCorpus.readline()
                    df.nextSent[index] = fCorpus.readline()                    
                else: # e.g., 10 and 13
                    for line in islice(fCorpus, position, position+1, 1):
                        df.prevSent[index] = line
                    df.sent[index]     = fCorpus.readline()
                    df.nextSent[index] = fCorpus.readline()
                previousIndex = index 
    return df

def eccoWMD(years, query, nTop, fs, forbidden):


    years = [str(t) for t in years]
    
    batch     = 25
    printStep = 200000
    nCores    = cpu_count()

    global start
    global tops
    global mW2V

    print("Loading Word2Vec Model", modelnameW2V) 
    mW2V = loadW2V()
    print("Done.")

    targetTokenized = docPreprocessing(query, mW2V)

    if targetTokenized == -1:
        return -1
    else:
        dfTarget = pd.DataFrame({
            "query": [query], 
            "token": [targetTokenized]
        })
        dfTarget.to_csv(queryDf)

    tops     = Top(nTop)
    bestDist = infinity




    #years        = ["1796", "1797", "1798", "1799", "1800"]
    #  years        = ["1797", "1800"]
    totSentences = {"1796":3280661, "1797":2945687,"1798":2857190, "1799":2622691, "1800":3098020 }
    grandTotal = sum([totSentences[t] for t in years])
    fs.value = fs.min
    stepFs = (fs.max-fs.min)/grandTotal

    totPruned = 0
    totRead   = 0

    start = timer()
    if len(forbidden.value) > 0:
        forbiddenWords = forbidden.value.strip("'").split(",")
        if len(forbiddenWords) > 0:
            # remove empty spaces, if any
            forbiddenWords = [w.strip() for w in forbiddenWords]
        print("List of forbidden words :: ", forbiddenWords)
        tops, totPruned, totRead = wmdParallelForbidden(nCores, batch, \
        printStep, years, tops, nTop, totPruned, totRead, totSentences,\
        targetTokenized, forbiddenWords, fs, stepFs)
    else:
        tops, totPruned, totRead = wmdParallel(nCores, batch, printStep,\
        years, tops, nTop, totPruned, totRead, totSentences, targetTokenized,\
        fs, stepFs)

    print("Total Sentences Analyzed = ", grandTotal, "over a period of ", len(years), "years in ", np.round(timer()-start,2), " seconds.")

    print("Retrieving sentences from disk...")
    df = retrieveSentences(tops, nTop)
    df.to_csv(solutionDf)
    print("Solution written on disk file", solutionDf,". Run analysis.") 

    return 0



