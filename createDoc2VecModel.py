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

 Author: Marco Caserta (marco dot caserta at ie dot edu)
 Started : 07.03.2018
 Updated : 
 Ended   :

"""
import sys
import logging
import csv
import json
import itertools
from collections import namedtuple
import os
from os import path, listdir
import fnmatch
from timeit import default_timer as timer

from gensim.models import doc2vec
from gensim.models import Word2Vec
from gensim.models import KeyedVectors
from gensim.similarities import WmdSimilarity
from gensim.corpora.dictionary import Dictionary

from nltk import word_tokenize
from nltk import sent_tokenize
from nltk.corpus import stopwords

stop_words        = stopwords.words("english")
prefix            = path.expanduser("~/research/nlp/data/")
ecco_folders      = ["ecco/ecco_portions/normed/96-00/"]
#  ecco_folders    = ["/home/marco/gdrive/research/nlp/data/temp/"]
vocab_folder      = "google_vocab/"
ecco_models_folder = "ecco_models/"
modelnameD2V       = "doc2vec.model" #  the Word2Vec model used in WMD

premiumList       = "premium/ecco-donut_freqs.txt"

preprocListDocs   = "preproc/preprocListDocs.csv"
preprocListCorpus = "preproc/preprocListCorpus.csv"
premiumDocs       = "preproc/premiumDocs.csv"
premiumCorpus     = "preproc/premiumCorpus.csv"
mappingYears      = "ecco_models/year_map.tsv"

period = ["1796", "1797", "1798", "1799", "1800"]
targetFile = "target.txt"


nCores            = 8

def vocabularyBuilding(prefix):
    '''
    Build embeddings and vocabulary based on Google News set (1.5Gb and 3M
    words and phrases based on 100B words from Google News)
    '''

    global vocab_dict

    # create dictionary first, using google list
    fullpath = path.join(prefix, vocab_folder)
    #  fullname = fullpath + "embed.dat"
    namevocab = fullpath + "embed500.vocab"
    with open(namevocab) as f:
        vocab_list = map(str.strip, f.readlines())
    vocab_dict = {w: k for k, w in enumerate(vocab_list)}
    print("Vocabulary read from disk ... ")


    #  print("Reading model from disk ")
    #  start = timer()
    #  #  gPath = fullpath + "GoogleNews-vectors-negative300.bin.gz"
    #  #  model = KeyedVectors.load_word2vec_format(gPath, binary=True,
    #  #  limit=500000)
    #  filename = fullpath + "embed500.model"
    #  #  model = KeyedVectors.load(filename, mmap="r")
    #  model = KeyedVectors.load(filename)
    #  model.init_sims(replace=True)
    #  print("Done in ", timer()-start, " time")

    #  return model
def readPremiumList():
    global premium
    
    premium = []
    with open(path.join(prefix,premiumList), "r") as f:
        for line in f:
            w = line.split()[0]
            if len(w) > 2 and w not in stop_words and w in vocab_dict:
                premium.append(w)
    print("len premium = ", len(premium))
    
    return premium

def readEcco(file2year, year):


    fDocs = open(premiumDocsXRow, "w")
    writerDocs = csv.writer(fDocs)

    fCorpus = open(premiumCorpusXRow, "w")
    writerCorpus = csv.writer(fCorpus)

    docs      = []
    sentences = []
    tot       = 0
    totFiles  = 0
    discarded = 0
    for folder in ecco_folders:
        fullpath = path.join(prefix, folder)
        #  totFiles += len(fnmatch.filter(os.listdir(fullpath), '*.txt'))
        totFiles = sum([file2year[i]==year for i in file2year.keys()])
        countFiles = 0
        for f in listdir(path.join(prefix, folder)):
            key = f.split(".")[0]
            #  print("Is ", key, " in dict? ", key in file2year)
            #  print("Year is ", file2year[key])
            #  print("Shall we take it ? ", file2year[key]==year)
            if file2year[key] != year:
                continue
            countFiles  += 1
            fullname     = fullpath + f
            ff           = open(fullname)
            sents        = sent_tokenize(ff.read())
            for sent in sents:
                #  print("-"*80)
                #  print("Initial sentence = ", sent)
                tot += 1
                words   = [w.lower() for w in word_tokenize(sent) if w not in
                stop_words]
                initLen = len(words)
                #  print(initLen, " = ", words)
                words    = [w for w in words if w.isalpha()]
                lenAlpha = len(words)
                #  print(lenAlpha, "After alpha = ", words)
                # keep if at least 50% words have all characters from alphabet
                if lenAlpha >= 0.5*initLen:
                    words = [w for w in words if w in vocab_dict and w in premium]
                    #  print("next ", words)
                    ll = len(words)
                    if ll > 2 and ll >= int(0.75*lenAlpha):
                        #  print("kept = ", words)
                        writerDocs.writerows([words])
                        writerCorpus.writerows([[sent]])
                        #  docs.append(ss)
                        #  corpus.append(sCorpus)
                    else:
                        #  print("discarded ... ")
                        discarded += 1
                else:
                    #  print("discarded ... ")
                    discarded += 1

            print("{0:5d}/{1:5d} :: Reading file {2:10s} ".format(countFiles,
            totFiles, f))
            #  if countFiles > 1:
            #      break

    print("Discarded {0} sentences out of {1}".format(discarded, tot))

    fDocs.close()
    fCorpus.close()

    return docs, sentences, totFiles


def writeListsPerSentence():
    with open(premiumCorpus, 'r') as f:
        corpus = json.load(f)
    with open(premiumDocs, 'r') as f:
        docs = json.load(f)

    with open(premiumDocsXRow, "w") as f:
        writer =csv.writer(f)
        writer.writerows(docs)
    with open(premiumCorpusXRow, "w") as f:
        writer =csv.writer(f)
        writer.writerows(corpus)

    exit(111)
    

def readPremiumLists(year):

    print("Reading premium sentences for year ", year, " from ", premiumDocsXRow)
    
    fCorpus = open (premiumCorpusXRow, "r")
    readerCorpus = csv.reader(fCorpus)

    fDocs = open(premiumDocsXRow, "r")
    totSents = sum(1 for _ in fDocs)
    print("Tot sentences = ", totSents)
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


    return docs, corpus
    #  return corpus, docs


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


def createDoc2Vec(docs, year):
    
    print("## Init Doc2Vec Model Creation for year {0}...".format(year))

    start = timer()
    # instantiate model (note that min_count=2 eliminates infrequent words)
    model = doc2vec.Doc2Vec(size = 300, window = 300, min_count = 2, iter =
    50, workers = nCores, dm=0, max_vocab_size=10000)

    # we can also build a vocabulary from the model
    model.build_vocab(docs)
    #  print("Vocabulary was built : ", model.wv.vocab.keys(), " ----> this is a voc")

    # train the model
    model.train(docs, total_examples=model.corpus_count, epochs=model.iter)
    model.init_sims(replace=True)

    # if we want to save the model on disk (to reuse it later on without
    # training)
    nn = "doc2vec.model." + year
    modelname = path.join(prefix,ecco_models_folder,nn)
    model.save(modelname)
    print("Doc2Vec model saved ", modelname)

    # this can be used if we are done training the model. It will reelase some
    # RAM. The model, from now on, will only be queried
    model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    print("... Done in {0:5.2f} seconds.\n".format(timer() - start))

    return model


def readTargetSentence(targetFile):
    """
    Read the target sentence from disk.
    Note that we also read how many sentences should be included in the query.
    """
    ff = open(targetFile, "r")
    reader = csv.reader(ff)
    target = next(reader)[0]
    nTop   = int(next(reader)[0])

    ff.close()
    return target, nTop
    
def targetPreprocessing(doc):

    doc = word_tokenize(doc)
    doc = [w.lower() for w in doc if w.lower() not in stop_words] # remove stopwords

    doc = [w for w in doc if w.isalpha()] # remove numbers and pkt
    #  doc = [w for w in doc if w.isalpha() and w in vocab_dict] # remove numbers and pkt

    return doc

def readMappingYears():

    fullname = path.join(prefix,mappingYears)
    print(fullname)
    filename = []
    year     = []
    with open(fullname, "r") as f:
        for line in f:
            ff, yy = line.split()
            filename.append(ff)
            year.append(yy)

    file2year = {f:y for f,y in zip(filename,year)}
    print("Mapping filename -> year built ...")
    #  print(file2year.keys())

    return file2year

    


def main(argv):

    global premiumDocsXRow
    global premiumCorpusXRow
    premiumDocsXRowBase   = "preproc/premiumDocsXRow.csv"
    premiumCorpusXRowBase = "preproc/premiumCorpusXRow.csv"
    file2year = readMappingYears()

    vocabularyBuilding(prefix)
    builddoc2vecmodel =False
    if builddoc2vecmodel:
        premium = readPremiumList()
        for year in period:
            print("Reading and building Doc2Vec for year ", year)
            premiumDocsXRow = premiumDocsXRowBase + "." + year
            premiumCorpusXRow = premiumCorpusXRowBase + "." + year
            #  # activate this part if we want to read the original files, preprocess them
            #  # and store one sentence per row
            docs, corpus, totoFiles = readEcco(file2year, year)
            #
            #  # activate this part to create a Doc2Vec model from the list of sentences
            docs, corpus = readPremiumLists(year)
            docs = transform4Doc2Vec(docs)
            createDoc2Vec(docs, year)

    usedoc2vecmodel =True
    if usedoc2vecmodel: 
        for year in period:
            premiumDocsXRow = premiumDocsXRowBase + "." + year
            premiumCorpusXRow = premiumCorpusXRowBase + "." + year
            print("="*80)
            print("* Year ", year)
            print("="*80)
            docs, corpus = readPremiumLists(year)  #  NEEDED ???
            fullpath = path.join(prefix,ecco_models_folder)
            print("Loading ", modelnameD2V, ".", year)
            fullname = fullpath + modelnameD2V +"." + year
            modelDoc2Vec  = doc2vec.Doc2Vec.load(fullname)
            target, nTop = readTargetSentence(targetFile)
            targetTokenized = targetPreprocessing(target)
            inferred_vector = modelDoc2Vec.infer_vector(targetTokenized)
            sims = modelDoc2Vec.docvecs.most_similar([inferred_vector], topn=10)
            print(sims)
            for i,j in sims:
                print("="*80)
                print("Keywords({0}) = {1:5.3f} :: {2}".format(i, j, docs[i]))
                print("="*80)
                print(corpus[i-1][0])
                print("-"*80)
                print("** ", corpus[i][0])
                print("-"*80)
                print(corpus[i+1][0])
                print("="*80)
                print("\n")

if __name__ == '__main__':
    #  logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    logging.basicConfig(
        format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
        level=logging.INFO
    )
    main(sys.argv[1:])
    #unittest.main()

