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

from multiprocessing import Pool
import json
import sys, getopt
import cplex
import os
from os import path, listdir
import csv
import itertools
import fnmatch
import numpy as np

from scipy.spatial.distance import cdist
from scipy._lib.six import xrange
from scipy import stats
from numpy.linalg import norm

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from timeit import default_timer as timer
import math
from sklearn.utils.extmath import row_norms, safe_sparse_dot
from sklearn.utils import check_array


from gensim.models import doc2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
from gensim.corpora.dictionary import Dictionary

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics.pairwise import euclidean_distances
from sklearn import manifold
import plotly
import plotly.plotly as py
import plotly.graph_objs as go

from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords
stop_words = stopwords.words("english")

prefix             = path.expanduser("/home/mcaserta/research/nlp/data/")
ecco_folders       = ["ecco/ecco_portions/normed/96-00/"]
vocab_folder       = "google_vocab/"
ecco_models_folder = "ecco_models/"

querySol           = "topList.txt" #  storing the current best query result
modelnameW2V       = "word2vec.model.96-00.all" #  the Word2Vec model used in WMD
modelnameD2V       = "doc2vec.model" #  the Word2Vec model used in WMD
premiumList        = "premium/ecco-donut_freqs.txt"
premiumDocs        = "/home/mcaserta/research/nlp/ecco_code/preproc/premiumDocs.csv"
premiumCorpus      = "/home/mcaserta/research/nlp/ecco_code/preproc/premiumCorpus.csv"
premiumDocsXRowBase    = "/home/mcaserta/research/nlp/ecco_code/preproc/premiumDocsXRow.csv"
premiumCorpusXRowBase  = "/home/mcaserta/research/nlp/ecco_code/preproc/premiumCorpusXRow.csv"
#  ecco_folders    = ["/home/marco/gdrive/research/nlp/data/temp/"]

nCores       =  8
totFiles     = -1

#  period = ["1796"]
period = ["1796", "1797", "1798", "1799", "1800"]

class Top:
    """
    Data structure used to store the top N sentences matching a given target
    sentence.
    We store both the full sentence (untokenized) and the tokenized and
    preprocessed sentence. We also store the previous and next sentences.
    """
    def __init__(self, nTop):
        self.score     = np.array([math.inf]*nTop)
        self.idnr      = [""]*nTop
        self.tokenSent = [[]]*nTop
        self.sent      = [""]*nTop
        self.previous  = [""]*nTop
        self.next      = [""]*nTop


# Parse command line
def parseCommandLine(argv):
    global withCplex
    global inputfile
    global targetFile 
    targetFile = ""
    withCplex = -1
    
    try:
        opts, args = getopt.getopt(argv, "ht:i:s:", ["help","type=", "ifile=",
        "sentence="])
    except getopt.GetoptError:
        print ("Usage : python doc2Vec.py -t <type> -s <sentence> ")

        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print ("Usage : python doc2Vec.py -t <type> -i <inputfile> ")
            sys.exit()
        elif opt in ("-i", "--ifile"):
            inputfile = arg
        elif opt in ("-t", "--type"):
            withCplex= arg
        elif opt in ("-s", "--sentence"):
            targetFile = arg

    if withCplex == -1:
        print("Error : Distance type not defined. Select one using -t : ")
        print("\t 1 : Cosine similarity (document-wise)")
        print("\t 2 : Word Mover's Distance (word-wise)")
        sys.exit(2)
    if targetFile == "":
        print("Error: Target sentence file not defined. Select one using -s\
        <namefile>")
        exit(2)

def readTargetSentence(targetFile):
    """
    Read the target sentence from disk.
    Note that we also read how many sentences should be included in the query
    and how many sentences per year should be filtered by 
    """
    ff = open(targetFile, "r")
    reader   = csv.reader(ff)
    target   = next(reader)[0]
    nTop     = int(next(reader)[0])
    nPerYear = int(next(reader)[0])

    ff.close()
    return target, nTop, nPerYear

def readEccoParallel(namefile):
    """
    Maybe for future uses. I am not able to make it work.
    """

    print("Reading ", namefile)
    docs = []
    sentences = []
    countFiles = 0
    ff = open(namefile)
    sents = sent_tokenize(ff.read())
    for sent in sents:
        words = [w.lower() for w in word_tokenize(sent)]
        words = [w for w in words if w.isalpha() and w not in stop_words]
        if len(words) > 2:
            docs.append(words)
            sentences.append([sent])
    print("Returning :", docs, " AND ", sentences)
    return zip(*docs, *sentences)

def printSummary(docs):

    nDocs   = len(docs)
    lengths = np.array([len(sent) for sent in docs])
    qq      = stats.mstats.mquantiles(lengths, prob = [0.0, 0.25, 0.50, 0.75, 1.10])

    print("\n\n====================================================")
    print("marco caserta (c) 2018 - WMD ")
    print("====================================================")
    print("* Nr. Sentences        : {0:>25d} *".format(nDocs))
    print("* Avg. Length          : {0:>25.2f} *".format(lengths.mean()))
    print("*  Min                 : {0:>25.2f} *".format(qq[0]))
    print("*  Q1                  : {0:>25.2f} *".format(qq[1]))
    print("*  Q2                  : {0:>25.2f} *".format(qq[2]))
    print("*  Q3                  : {0:>25.2f} *".format(qq[3]))
    print("*  Max                 : {0:>25.2f} *".format(qq[4]))

    print("\n")
    print("* Distance Type        : {0:>25s} *".format("WMD"))
    print("====================================================")
    print("\n")


    return

    fig, axes = plt.subplots(nrows=1, ncols=1)


    df = pd.DataFrame({
        "words": lengths})

    step_size = round((qq[4]-qq[0])/15)
    bins = np.arange(start=qq[0], stop=qq[4]+1, step= step_size)
    sns.distplot(df.words, hist_kws=dict(edgecolor="gray", linewidth=2),
    hist=True, kde=False, rug=False, bins=bins, ax=axes)
    axes.set(xticks=bins)
    axes.set(xlabel="Nr. Words")
    axes.set_title(r"[$\mu = $" +
    str("{0:5.2f}".format(lengths.mean())) + " $\sigma = $" +
    str("{0:5.2f}".format(lengths.std())) + "]")

    sns.plt.suptitle("Distribution of Words Length")
    #  sns.plt.show()
    sns.plt.savefig("distributionPlots.png")
    print("Distribution Plots saved on disk ('distributionPlots.png')")

    return


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


def transform4Doc2Vec(docs):
    """
    Doc2Vec requires documents to be tokenized and, in addition, each doc
    should have a unique TAG.

    Note that, if the corpus has been preprocessed, each document is already
    tokenized. Therefore, some of the steps below can be skipped.
    """

    # transform documents to be used by doc2Vec
    documents = []
    analyzedDocument = namedtuple('AnalyzedDocument', 'words tags')
    for i, doc in enumerate(docs):
        # use first line if documents are not tokenized, otherwise next line
        #  words = text.lower().split()
        tags = [i]
        documents.append(analyzedDocument(doc, tags))

    return documents


def trainingModel4wmd(corpus):
    """
    Training a model to be used in WMD.

    Settings:
    - hs = 0: Do not use hiercarchical softmax, but negative sampling
    - negative > 0 : number of negative words to be used in negative sampling
    - sorted_vocab : the most frequent words come first (we can trim it)

    """
    model = Word2Vec(workers = nCores, size = 100, window = 300,
    min_count = 2, hs=0, negative= 15, iter = 100, sorted_vocab=1)
    # build vocabulary based on the corpus
    model.build_vocab(corpus)

    model.train(corpus, total_examples=model.corpus_count,
    epochs=model.iter) 

    # use the following if we want to normalize the vectors
    # in this case, all vectors are converted to unit-length
    model.init_sims(replace=True)

    return model


def getMostSimilarWMD(model, corpus, target, nTop):
    """
    Using the word2vec "model", find in the corpus the nTop most similar
    documents to target.

    NOTE: WmdSimilarity() provide the "negative" of wmdistance(), i.e.:
    sim(d1,d2) = 1/(1+wmdistance(d1,d2)).

    See: https://markroxor.github.io/gensim/static/notebooks/WMD_tutorial.html
    """

    instance = WmdSimilarity(corpus, model, num_best=nTop)
    instance.num_best = nTop
    sims = instance[target]
    #  print('Query:', target)
    #  print("="*80)
    #  for i in range(nTop):
    #      print( 'sim = %.4f' % sims[i][1])
    #      print(corpus[sims[i][0]])
    #      print("="*80)

    return sims

def transportationProblem(model, target, source):
    """
    Define the Transportation model used for the Word Movers Distance (WMD). We
    also setup a graph, to visualize the weight associated to each pair of
    words in a query.
    """

    from gensim.corpora import Dictionary

    print("== == == OFFICIAL WMD = ", model.wmdistance(source, target))

    dctTarget = Dictionary([target])
    dctd1     = Dictionary([source])
    capacity  = dctd1.doc2bow(source)
    demand    = dctTarget.doc2bow(target)
    nS = len(capacity)
    nD = len(demand)
    cap = np.array([capacity[i][1] for i in range(nS)])
    cap = cap/sum(cap)
    dem = np.array([demand[i][1] for i in range(nD)])
    dem = dem/sum(dem)
    print("S and D = ", capacity, " " , demand)
    for i in range(nS):
        print("v2 = ", dctd1[i], " = ", capacity[i][1])
    print("*"*50)
    for j in range(nD):
        print("v2 = ", dctTarget[j], " = ", demand[j][1])
#
    print("Cap vector ", cap)
    print("Dem vector ", dem)

    # distances
    S         = [model.wv[dctd1[i]] for i in range(nS)]
    D         = [model.wv[dctTarget[i]] for i in range(nD)]
    dd        = pairwise_distances(S, D, metric = "euclidean")
    # compute lower bound
    print(dd)
    lb1 = 0.0
    for i in range(nS):
        lb1 += min([dd[i][j] for j in range(nD)])*cap[i]
    print("lb1 = ", lb1)
    lb2 = 0.0
    for j in range(nD):
        lb2 += min([dd[i][j] for i in range(nS)])*dem[j]
    print("lb2 = ", lb2)


    [z, xSol] = solveTransport(dd, cap, dem)

    indexMax  = [np.argmax(xSol[i]) for i in range(nS)]
    thickness = [xSol[i][indexMax[i]] for i in range(nS)]
    indexMax  = [indexMax[i] + nS for i in range(nS)]

        
    X  = []
    tt = []
    for i in range(nS):
        X.append(model.wv[dctd1[i]])
        tt.append("doc")
    for j in range(nD):
        X.append(model.wv[dctTarget[j]])
        tt.append("target")

    seed = np.random.RandomState(seed=27)
    mds  = manifold.MDS(n_components = 2, metric = True, max_iter = 3000, eps =
    1e-9, random_state = seed, dissimilarity = "euclidean", n_jobs = nCores)
    sol  = mds.fit(X).embedding_


    colors      = ["blue", "red", "green", "magenta"]
    typesLabels = ["doc", "target"]
    nPoints     = len(X)
    data        = []
    words       = [dctd1[i] for i in range(nS)]
    ids         = [i for i in range(nPoints) if tt[i] == "doc"]
    trace = go.Scatter(
                       x            = sol[ids,0],
                       y            = sol[ids,1],
                       showlegend   = False,
                       hoverinfo    = "skip",
                       text         = words,
                       textposition = "bottom",
                       mode         = "text+markers",
                       marker       = dict(color = "red",size = cap*100))
    data.append(trace)

    words = [dctTarget[i] for i in range(nD)]
    ids   = range(nS, nS+nD)
    trace = go.Scatter(
                       x            = sol[ids,0],
                       y            = sol[ids,1],
                       showlegend   = False,
                       text         = words,
                       textposition = "bottom",
                       mode         = "text+markers",
                       marker       = dict(color      = "blue",size = dem*100))
    data.append(trace)

    annotations = []
    for i in range(nS):
        annot = dict(
                     x         = sol[i,0],
                     y         = sol[i,1],
                     xref      = "x",
                     yref      = "y",
                     text      = round(thickness[i],2),
                     ax        = -50,
                     ay        = -50,
                     #  ax     = (sol[indexMax[i],0]-sol[i,0])*500,
                     #  ay     = (sol[indexMax[i],1]-sol[i,1])*500,
                     showarrow = False)
        annotations.append(annot)

        trace = go.Scatter(
                           x               = [sol[i,0], sol[indexMax[i],0]],
                           y               = [sol[i,1], sol[indexMax[i],1]],
                           showlegend      = False,
                           opacity         = 0.25,
                           mode            = "lines",
                           text            = round(thickness[i],2),
                           #  textposition = "top center",
                           line            = dict(color = "red",width = thickness[i]*20))
        data.append(trace)


    layout = go.Layout(
                        title       = "MDS of Two Sentences",
                        annotations = annotations,
                        xaxis       = dict(zeroline  = False),
                        yaxis       = dict(zeroline  = False))

    fig         = go.Figure(data = data, layout = layout)

    plotly.offline.plot(fig)


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
            varName = "x." + str(i) + "." + str(j)
            cpx.variables.add(obj   = [float(matrixC[i][j])],
                              lb    = [0.0],
                              names = [varName])
    # capacity constraint
    for i in range(nS):
        index = [x_ilo[i][j] for j in range(nD)]
        value = [1.0]*nD
        capacity_constraint = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [capacity_constraint],
                                   senses   = ["L"],
                                   rhs      = [cap[i]])

    # demand constraints
    #  for j in dctTarget:
    for j in range(nD):
        index = [x_ilo[i][j] for i in range(nS)]
        value = [1.0]*nS
        demand_constraint = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [demand_constraint],
                                   senses   = ["G"],
                                   rhs      = [dem[j]])
    cpx.parameters.simplex.display.set(0)
    cpx.solve()

    z = cpx.solution.get_objective_value()

    return z

def solveTransport(matrixC, cap, dem):
    """
    Solve transportation problem as an LP.
    This is my implementation of the WMD.
    """
    
    nS    = len(cap)
    nD    = len(dem)
    cpx   = cplex.Cplex()
    x_ilo = []
    cpx.objective.set_sense(cpx.objective.sense.minimize)
    for i in range(nS):
        x_ilo.append([])
        for j in range(nD):
            x_ilo[i].append(cpx.variables.get_num())
            varName = "x." + str(i) + "." + str(j)
            cpx.variables.add(obj   = [float(matrixC[i][j])],
                              lb    = [0.0],
                              names = [varName])
    # capacity constraint
    for i in range(nS):
        index = [x_ilo[i][j] for j in range(nD)]
        value = [1.0]*nD
        capacity_constraint = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [capacity_constraint],
                                   senses   = ["L"],
                                   rhs      = [cap[i]])

    # demand constraints
    #  for j in dctTarget:
    for j in range(nD):
        index = [x_ilo[i][j] for i in range(nS)]
        value = [1.0]*nS
        demand_constraint = cplex.SparsePair(ind=index, val=value)
        cpx.linear_constraints.add(lin_expr = [demand_constraint],
                                   senses   = ["G"],
                                   rhs      = [dem[j]])
    cpx.parameters.simplex.display.set(0)
    cpx.solve()

    z = cpx.solution.get_objective_value()

    #  print("z* = ", z)
    #  x_sol = []
    #  for i in range(nS):
    #      x_sol.append(cpx.solution.get_values(x_ilo[i]))
    #      print([round(x_sol[i][j],2) for j in range(nD)])

    #  return [z, x_sol]
    return [z, []]

def loadWord2VecModel():
    """
    Read or create a Word2Vec model (depending on whether the one defined in
    the header of this file actually exists.).
    """

    fullpath = path.join(prefix,ecco_models_folder)
    fullname = fullpath + modelnameW2V
    if os.path.exists(fullname):
        print("    \t Word2Vec model read from disk ({0})".format(modelnameW2V))
        modelWord2Vec = Word2Vec.load(fullname)
        modelWord2Vec.init_sims(replace=True)
    else:
        print("    \t ERROR ::: Word2Vec model not available!")

    return modelWord2Vec


def compileWMDSimilarityList(modelWord2Vec, target, docs, p):
    """
    Returns a list of the top N most similar sentences w.r.t. the target
    sentence, using the Word Mover's Distance method.
    """


    nDocs = len(docs)
    # create list of tasks for parallel processing
    tasks = []
    for i in range(nDocs):
        tasks.append([target,docs[i]])

    results = p.starmap(modelWord2Vec.wmdistance, tasks)

    #  print ("## Computing distances using WMD (sequential version)\
    #  ...".format(nCores))
    #  start = timer()
    #  sims = getMostSimilarWMD(modelWord2Vec, docs, target, 5)

    # NOTE: The corpus now needs to be re-processed, since when we use the
    # transportation model below, we need to get the word2vec representation of
    # every word. If, in training, min_count > 1, some of the words in the
    # preprocessed corpus might not be in the vocabulary produced by word2vec,
    # even though they were in the vocabulary produced by google.


    # source : the origin, the document we want to transform into the target
    #  source = docs[sims[2][0]]
    #  transportationProblem(modelWord2Vec, target, source)
    #  print("... done with distance computation in {0:5.2f} seconds.\n".format(timer()-start))

    return results

def updateList2(tops, distance, doc, corp, prev, pos, idd):

    dist = np.append(tops.score, distance)
    idx = dist.argsort()

    topAux = Top(nTop)
    for i in range(nTop):
        topAux.score[i] = dist[idx[i]]
        if idx[i] == nTop: #  this is the new sentence
            topAux.previous[i]  = prev
            topAux.next[i]      = pos
            topAux.tokenSent[i] = doc
            topAux.sent[i]      = corp
            topAux.idnr[i]      = idd
        else:
            topAux.tokenSent[i] = tops.tokenSent[idx[i]]
            topAux.sent[i]      = tops.sent[idx[i]]
            topAux.previous[i]  = tops.previous[idx[i]]
            topAux.next[i]      = tops.next[idx[i]]
            topAux.idnr[i]      = tops.idnr[idx[i]]

    return topAux



def updateList(tops, distances, docs, corpus, previous, post, period):
    """
    Update list of top N best results.
    This can be improved.
    """

    if min(distances) > tops.score[nTop-1]:
        return tops

    nDocs = len(docs)
    for i in range(nTop):
        distances.append(tops.score[i])

    dist = np.array(distances)
    idx = dist.argsort()
    topAux = Top(nTop)
    for i in range(nTop):
        topAux.score[i] = dist[idx[i]]
        if idx[i] < nDocs:
            topAux.previous[i]  = previous[idx[i]]
            topAux.next[i]      = post[idx[i]]
            topAux.tokenSent[i] = docs[idx[i]]
            topAux.sent[i]      = corpus[idx[i]]
        else:
            topAux.tokenSent[i] = tops.tokenSent[idx[i]-nDocs]
            topAux.sent[i]      = tops.sent[idx[i]-nDocs]
            topAux.previous[i]  = tops.previous[idx[i]-nDocs]
            topAux.next[i]      = tops.next[idx[i]-nDocs]

    # write current best query to disk
    printQueryResults(topAux, nTop, period, toDisk = 1)

    return topAux

def printQueryResults(tops, nTop, period, toDisk = 1):
    """
    Print, either to screen or to file, the list of sentences closest to the
    query. This is called every time a new best solution is found.
    """


    # dump it to file or print it to screen
    if toDisk == 1:
        sys.stdout = open(querySol, "w")

    print("** ** ** Target sentence : ", target)

    print("\n\n LIST OF SENTENCES IN CONTEXT\n")
        
    for i in range(nTop):
        print("="*80)
        ss = ", ".join( repr(e) for e in tops.previous[i] )
        print(ss)
        print("-"*80)
        ss = ", ".join( repr(e) for e in tops.sent[i])
        print("({0:3d}) :: {1} ".format(i+1, ss))
        print("-"*80)
        ss = ", ".join( repr(e) for e in tops.next[i] )
        print(ss)
        print("="*80)
        print("\n\n")

    dd = {key:0 for i in range(nTop) for key in tops.tokenSent[i] }
    for i in range(nTop):
        for w in tops.tokenSent[i]:
            dd[w] += 1

    for key in sorted(dd, key=dd.get, reverse=True):
        print("{0:20s} = {1:4d}".format(key, dd[key]))

    years = [tops.idnr[i].split(".")[0] for i in range(nTop)]
    dyear = {key:0 for i in range(nTop) for key in years}
    for i in range(nTop):
        dyear[years[i]] += 1

    nT   = len(period)
    minT = period[0]
    maxT = period[nT-1]
    print("\nDistribution of Top Sentences over Time Period {0} - {1}".format(minT, maxT))
    for key in dyear:
        print("{0:8s} = {1:4d}".format(key, dyear[key]))
    print("\n")


    for i in range(nTop):
        print("="*80)
        print("Keywords({0}) = {1:5.3f} :: {2}".format(i+1, tops.score[i],
        tops.tokenSent[i]))
        ss = ", ".join( repr(e) for e in tops.sent[i])
        print("{0:15s} - {1}".format(tops.idnr[i], ss))
        print("="*80)

    #  for key,val in sorted(dd.iteritems(), key=lambda (k,v): (v,k)):
    #      print("{0:20s} = {1:4d}".format(key, val))


    sys.stdout = sys.__stdout__

    df = pd.DataFrame({
        "id"       : tops.idnr,
        "sentence" : tops.sent,
        "previous" : tops.previous,
        "next"     : tops.next,
        "year"     : years,
        "score"    : tops.score,
        "tokenized": tops.tokenSent
    })
    df.to_csv("sol.csv")


def readPremiumLists():
    """
    The premium sentences are those sentences obtained after preprocessing,
    i.e., docs are already tokenized (while corpus retains the original
    sentence), isalpha(), stopw_words, etc. have been applied, and, in
    addition, words that are not in the vocabulary and in the premium list have
    been removed (see createDoc2VecModel.py for more details on how these lists
    have been produced.)
    """

    print(" .. Reading premium sentences from ", premiumDocsXRow)
    fDocs = open(premiumDocsXRow, "r")
    totSents = sum(1 for _ in fDocs)
    print(" .. Tot sentences = ", totSents)

    fCorpus = open (premiumCorpusXRow, "r")
    readerCorpus = csv.reader(fCorpus)
    fDocs = open(premiumDocsXRow, "r")
    readerDocs   = csv.reader(fDocs)
   
    corpus = []
    docs   = []
    for row in itertools.islice(readerCorpus, 0, totSents):
        #  print(readerCorpus.line_num, "Row corpus = ", row)
        corpus.append(row)

    for row in itertools.islice(readerDocs, 0, totSents):
        #  print(readerDocs.line_num, "Row docs = ", row)
        docs.append(row)


    return docs, corpus, totSents

def readAllPremium(period):

    docsAll   = []
    corpusAll = []
    totAll    = []
    idsAll    = []

    for year in period:
        premiumDocsXRow   = premiumDocsXRowBase + "." + year
        premiumCorpusXRow = premiumCorpusXRowBase + "." + year

        start = timer()
        fDocs = open(premiumDocsXRow, "r")
        totSents = sum(1 for _ in fDocs)
        print("    \t - Reading premium sentences from {0}. (Tot. Sentences = {1:10d})".format(premiumDocsXRow, totSents))

        fCorpus = open (premiumCorpusXRow, "r")
        readerCorpus = csv.reader(fCorpus)
        fDocs = open(premiumDocsXRow, "r")
        readerDocs   = csv.reader(fDocs)
       
        corpus = []
        docs   = []
        ids    = []
        for row in itertools.islice(readerCorpus, 0, totSents):
            corpus.append(row)

        for row in itertools.islice(readerDocs, 0, totSents):
            #  print(readerDocs.line_num, "Row docs = ", row)
            docs.append(row)
            idx = year + "." + str(readerDocs.line_num)
            ids.append(idx)

        docsAll.append(docs)
        corpusAll.append(corpus)
        totAll.append(totSents)
        idsAll.append(ids)
    
        print("    \t   Done in {0:5.2f} seconds.\n".format(timer() - start))

    return docsAll, corpusAll, totAll, idsAll



def shortlistDoc2Vec(model, year, targetTokenized, docs, corpus, totSents, ids, nPerYear):
    """
    Based on the doc2vec.model.year model, the top N sentences for the current
    year are retrieved and stored in a temporary data structure. These
    sentences are eventually appended to docs and corpus, this entering in the
    shortlist to be used by wmd.
    """

    docsNew   = []
    corpusNew = []
    previous  = []
    post      = []
    idsNew    = []

    namefile = modelnameD2V + "." + year
    fullpath = path.join(prefix,ecco_models_folder)
    print("\t Loading ", namefile)
    fullname = fullpath + namefile
    modelDoc2Vec  = doc2vec.Doc2Vec.load(fullname)
    inferred_vector = modelDoc2Vec.infer_vector(targetTokenized)
    nSelected = min(nPerYear, totSents)
    sims = modelDoc2Vec.docvecs.most_similar([inferred_vector], topn=nSelected)

    #  sims = zip(list(range(len(docs))), list(range(len(docs))))
    for i,j in sims:
        #********************************************************************
        # NOTE: Remove this when models are properly built !!!!!! <----------
        #  docTemp = [w for w in docs[i] if w in model.wv.vocab]
        #  docsNew.append(docTemp)
        docsNew.append(docs[i])
        corpusNew.append(corpus[i])
        idsNew.append(ids[i])
        if i > 0:
            previous.append(corpus[i-1])
        else:
            previous.append("**")
        if i < totSents-1:
            post.append(corpus[i+1])
        else:
            post.append("**")

    modelDoc2Vec = None

    return docsNew, corpusNew, previous, post, idsNew

def mycdist(XA, XB):
    XA = np.asarray(XA, order="c")
    XB = np.asarray(XB, order="c")
    sA = XA.shape
    sB = XB.shape
    mA = sA[0]
    mB = sB[0]
    n  = sA[1]
    dm = np.zeros((mA,mB), dtype=np.double)
    XA = np.ascontiguousarray(XA, dtype=np.double)
    XB = np.ascontiguousarray(XB, dtype=np.double)
    for i in xrange(0,mA):
        for j in xrange(0,mB):
            dm[i,j]=norm(XA[i,:]-XB[j,:])

    return dm
    

def pairwise_dists(X, Y, YY):
    """
    Fast implementation of pairwise distances. We use the shortcut:
    dist(X,Y ) = sqrt( dot(X,X) - 2*dot(X,Y) + dot(Y,Y)
    which has the problem that does not return perfectly symmetric results (but
    this is not a problem in this context.)
    """
    X = check_array(X)
    XX = row_norms(X, squared=False)[:, np.newaxis]

    distances = safe_sparse_dot(X, Y.T, dense_output = True)
    distances *= -2
    distances += XX
    distances += YY
    np.maximum(distances, 0, out=distances)

    return np.sqrt(distances, out=distances)

def wmdTransport(model, target, docs, corpus, previous, post, ids):

    
    topsAux   = Top(nTop)
    nDocs     = len(docs)
    results   = [-1.0]*nDocs
    bestUB    = math.inf
    totPruned = 0
    bestTime  = 0.0
    star      = ""
    start     = timer()

    # setup target (invariant over cycle)
    nWords = len(target)
    demand = {key:0 for key in target}
    for w in target:
        demand[w] += 1
    nD     = len(demand)
    dem    = [val/nWords for val in demand.values()]
    D      = model.wv[demand.keys()]

    for index in range(nDocs):
        
        if index  % 50000 == 0:
            print("[{0:9d}/{1:9d}] score = {2:5.3f}\t time = {3:5.2f}\t [Fathomed {4:9d} ({5:4.2f})] {6}".format(index, nDocs, bestUB, timer()-start, totPruned, totPruned/nDocs, star)) 
            star = ""

        source   = docs[index]
        nWords   = len(source)
        capacity = {key:0 for key in source}
        #  print("SOURCE = ", source)
        for w in source:
            capacity[w] += 1

        nS       = len(capacity)
        cap      = [val/nWords for val in capacity.values()]
        S        = model.wv[capacity.keys()]
        dd       = cdist(S,D)
        #  dd2=mycdist(S2,D)

        lb = 0.0
        for i in range(nS):
            lb += min([dd[i][j] for j in range(nD)])*cap[i]
        if lb >= topsAux.score[nTop-1]:
            totPruned += 1
            continue
        lb = 0.0
        for j in range(nD):
            lb += min([dd[i][j] for i in range(nS)])*dem[j]
        if lb > topsAux.score[nTop-1]:
            totPruned += 1
            continue

        # if not pruned, solve transportation problem
        results[index] = solveTransport2(dd, cap, dem, nS, nD)
        if results[index] < topsAux.score[nTop-1]:
            doc  = source
            corp = corpus[index]
            prev = previous[index]
            pos  = post[index]
            idd  = ids[index]
            topsAux = updateList2(topsAux, results[index], doc, corp, prev,
            pos, idd)

        #  zWMD = model.wmdistance(source, target)
            if results[index] < bestUB:
                star     = " *"
                bestUB   = results[index]
                bestTime = timer()-start
                # write current best query to disk
                printQueryResults(topsAux, nTop, period, toDisk = 1)

    print("Pruned = {0}/{1}".format(totPruned, nDocs))

    return topsAux

    

def main(argv):
    '''
    The Word Mover Distance (WMD) implementation for the ECCO dataset.
    '''

    global target
    global nTop
    global premiumDocsXRow
    global premiumCorpusXRow


    parseCommandLine(argv)


    print("\n")
    print("*"*80)
    print("\t Preprocessing :: ")


    print("[P1]\t Setting up Word2Vec model ...")
    start = timer()
    modelWord2Vec = loadWord2VecModel()
    print("\t .... Done in {0:5.2f} seconds.\n".format(timer() - start))

    # read all preprocessed sentences and keep them in memory
    print("[P2]\t Setting up Word2Vec model ...")
    start = timer()
    docsD2V, corpusD2V, totSents, idsD2V = readAllPremium(period)
    print("\n\t Preprocessing done in {0:5.2f} seconds.".format(timer() - start))
    print("*"*80)

    while True:
        startCycle = timer()

        print(" ------------------------------")
        print("|  DEFINE THE QUERY HERE - NEW |")
        print(" ------------------------------")
        quitting = False
        answ1 = ""
        while answ1 not in ["y","n"]:
            answ1 = input("Do you want to process a new query? [y/n] ")
            if answ1 == "n":
                answ2 = ""
                while answ2 not in ["y","n"]:
                    answ2 = input("Do you want to EXIT? [y/n] ")
                    if answ2 == "y":
                        print("Quitting the application...")
                        quitting = True
                        break
                    elif answ2 == "n":
                        answ1=""
            elif answ1 == "y":
                answ2 = ""
                while answ2 != "y":
                    query = input("Type here the query : ")
                    print("The query is '{0}'".format(query))
                    answ2 = input("Is that correct? [y/n] ")

                while True:
                    nrTop = 1000
                    answ4 = input("Nr. of top sentences in final answer [Press enter for default=1000, or insert any value] = ")
                    if answ4 != "":
                        nrTop = int(answ4)
                    if nrTop >= 1:
                        break

                while True:
                    nrSentences = 500000
                    answ3 = input("Nr. of sentences per year? [Press enter for default=500k, or insert any value] = ")
                    if answ3 != "":
                        nrSentences = int(answ3)
                    if nrSentences >= 1:
                        break

                print("Writing the 'target.txt' file to disk ...")
                fTarget = open("target.txt", "w")
                fTarget.write(query+"\n")
                fTarget.write(str(nrTop)+"\n")
                fTarget.write(str(nrSentences)+"\n")
                fTarget.close()

                answ3 = ""
                while answ3 not in ["y", "q"]:
                    answ3 = input("Restart? Type 'y' to restart, 'q' to quit: ")
                if answ3 == "q":
                    print("Quitting the application...")
                    quitting = True
                    break

        if quitting:
            break

        docs       = [] #  the documents in format required by doc2vec
        corpus     = [] #  the documents after preprocessing (not fit for doc2vec)
        distMatrix = [] #  the distance matrix used by hierarchical clustering
        previous   = [] #  previous sentence in the shortlist
        post       = [] #  next sentence in the shortlist
        ids        = [] #  sentence id (year.row_number) in preproc file
        targetTokenized = -1

        while targetTokenized == -1:
            target, nTop, nPerYear = readTargetSentence(targetFile)
            targetTokenized = docPreprocessing(target, modelWord2Vec)

            if targetTokenized == -1:
                input("Fix the target file and press any key to continue\
                ('target.txt')")
            else:
                print("\n\n")
                print("*"*80)
                print("* Query : ", target)
                print("*"*80)
                print("\n\n")

        tops     = Top(nTop)
        bestDist = math.inf

        print("TArget tokenized is ", targetTokenized, " vs ", target)
        dfTarget = pd.DataFrame({
            "query": [target], 
            "token": [targetTokenized]
        })
        dfTarget.to_csv("query.csv")

        #  p = Pool(nCores) #  used for parallel processing of WMD
        start = timer()


        counter = -1
        for year in period:
            counter += 1
            print("- Processing sentences of year ", year)
            startYear = timer()

            # reactivate this part to easy up use of memory
            #  premiumDocsXRow   = premiumDocsXRowBase + "." + year
            #  premiumCorpusXRow = premiumCorpusXRowBase + "." + year
            #
            #  print("="*80)
            #  print("* Year ", year)
            #  print("="*80)
            #  docsD2V, corpusD2V, totSents = readPremiumLists()

            docsAux, corpusAux, previousAux, postAux, idsAux =\
            shortlistDoc2Vec(modelWord2Vec, year,
            targetTokenized, docsD2V[counter], corpusD2V[counter],
            totSents[counter], idsD2V[counter], nPerYear)
            #  shortlistDoc2Vec(modelWord2Vec, year,
            #  targetTokenized, docsD2V, corpusD2V, totSents, nPerYear)
            for i in range(len(docsAux)):
                docs.append(docsAux[i])
                corpus.append(corpusAux[i])
                previous.append(previousAux[i])
                post.append(postAux[i])
                ids.append(idsAux[i])

            print("\t Done in {0:5.2f} seconds. Up to year {1}, nr. sentences in corpus = {2}".format(timer()- startYear, year, len(docs)))


        print("... Doc2Vec filtering done in {0:5.2f} seconds.\n".format(timer() - start))

        printSummary(docs)

        start    = timer()
        if withCplex == "1":
            tops = wmdTransport(modelWord2Vec, targetTokenized, docs, corpus,
            previous, post, ids)
        else:
            distances = compileWMDSimilarityList(modelWord2Vec,
            targetTokenized, docs, p)
            tops = updateList(tops, distances, docs, corpus, previous, post,
            period)
            if tops.score[0] < bestDist:
                bestDist = tops.score[0]
                print("Best Match = {0:5.2f} :: {1}".format(tops.score[0], tops.tokenSent[0]))


        printQueryResults(tops, nTop, period, toDisk = 1)

        print("... WMD done in {0:5.2f} seconds.\n".format(timer() - start))

        print("\nIMPORTANT: Download the 'topList.txt' to get the solution.\n")



        #  print("Do you want to produce a mapping? Choose sentence [1-",nTop,"]\
        #  (any other number to exit)")
        #  k = 0
        #  #  k = int(input())
        #  if k > 0 and k <= nTop:
        #      source = tops.tokenSent[k-1]
        #      transportationProblem(modelWord2Vec, targetTokenized, source)


if __name__ == '__main__':
    main(sys.argv[1:])
