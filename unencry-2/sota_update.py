from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
import os
import numpy as np
import pandas as pd
import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from nltk import stem
import gensim
import sys
import numpy as np
from sklearn import preprocessing
import operator
from nltk.corpus import wordnet
from collections import defaultdict
import time
from collections import defaultdict
import time


#############
################################
############## FINAL FILE
import re
from collections import Counter

def words(text): return re.findall(r'\w+', text.lower())

WORDS = Counter(words(open('big.txt').read()))

def P(word, N=sum(WORDS.values())): 
    "Probability of `word`."
    return WORDS[word] / N

def correction(word): 
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word): 
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words): 
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in WORDS)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word): 
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))
stopsets = ["can", "could", "may", "might", "also", "shall", "therefore","moreover", "however", "whose", "new", "furthermore",
                "aforementioned", "should", "since", "mine", "would", " "]

def buildModel(data,size):
    global MODEL
    MODEL = gensim.models.Word2Vec(data,size=size,window=5,min_count=1,workers=4)
#    if local:
#        model.save_word2vec_format(outdir+"term2vec_model_"+str(size))
#    else:
#        model.wv.save_word2vec_format(outdir+"term2vec_model_"+str(size))
    #model.save_word2vec_format(outdir+"term2vec_model_"+str(size))
    
def tokenize(document):
    """Returns a list whose elements are the separate terms in
    document.  Something of a hack, but for the simple documents we're
    using, it's okay.  Note that we case-fold when we tokenize, i.e.,
    we lowercase everything."""
    characters = "\-'.,!#$%^&*();:\n\t\\\"?!{}[]<>"
    terms = document.lower().split()
    return [term.strip(characters) for term in terms]


def cosineSim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

def filesToDataFrame():
    fileMat=[]
    global all_term
    for file in fileList:

        fileData = open(str(directory)+str(file),'r').read()
        word_list = tokenize(fileData)
        filtered_word_list = word_list[:]
        for word in word_list: # iterate over word_list
            if word in stopwords.words('english') and word in stopsets: 
                filtered_word_list.remove(word)
        #filtered_word_list = stemmer(filtered_word_list)
        
        fileMat.append(filtered_word_list)
    all_term = fileMat
    return fileMat



if __name__ == "__main__":
    
#    documents = ["This little kitty came to play when I was eating at a restaurant.",
#                 "Merley has the best squooshy kitten belly.",
#                 "Google Translate app is incredible.",
#                 "If you open 100 tab in google you get a smiley face.",
#                 "Best cat photo I've ever taken.",
#                 "Climbing ninja cat.",
#                 "Impressed with google map feedback.",
#                 "Key promoter extension for Google Chrome."]

    duration = 1  # second
    freq = 440  # Hz
#    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))
    
    
    vectorizer = TfidfVectorizer(stop_words='english')
    outdir = './outputs/'
    start= time.clock()
    stop_words = set(stopwords.words('english'))
    global local
    local = True
    if local:
        directory = "/home/c00300901/Desktop/key files data sets/amazon_mod/"
    else:
        directory = "./data/tech/" #ignore
    
    fileList = os.listdir(directory)
    new_word_list = []
    fileCount = 0
#    data=filesToDataFrame()
#    docDir = './data/smalltech/'
    docDir = directory
    directory_cluster='/home/c00300901/PycharmProjects/clusters_sota_allbbc/300 cluster/'
    documents = os.listdir(docDir)
    allDocuments = []
    t=1
    stemmer = nltk.PorterStemmer()
    total_token_count=0
    for eachFile in documents:
        gh1=open(docDir+eachFile,'r').read()
        try:
            gh2=tokenize(gh1)
            gh3 = [stemmer.stem(tagged_word) for tagged_word in gh2]
            gh = set(gh3) #unique
            c=""
            for g in gh:
                g= correction(g)
                c+=g
                c+="\n"
            allDocuments.append(c)
        except:
            print("do nothing")
        for e_word in allDocuments:
            total_token_count+=1
        t+=1
    print("zobaed")
    
    X = vectorizer.fit_transform(allDocuments)
    print("true_k")
    #print(X)
#    data=filesToDataFrame()
#    size = 40
#    buildModel(data,size)
    howmanyKs = [300]
    cnt=0
    recheck=[]
    for true_k in howmanyKs:
        print(true_k)
        model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
        model.fit(X)
        
        #print("Top terms per cluster:")
        order_centroids = model.cluster_centers_.argsort()[:, ::-1]
        terms = vectorizer.get_feature_names()
        print("zobaed")
#        unique_terms = set(terms)
#        kl=[]
#        for wi in unique_terms:
#            if(str.isalpha(wi)):
                
#                kl.append(correction(wi))
        for i in range(true_k):
            fh=open(directory_cluster+"Cluster_sample_"+str(true_k)+"_"+str(i)+".txt", "w")

            print("Cluster %d:" % i)
            for ind in order_centroids[i,:int(len(terms)/true_k) ]:     #int(len(unique_terms)/true_k)

                
                try: 
                    if not terms[ind][0].isdigit():
#                        if i==9: print ("b write")
                        fh.write(str(terms[ind]))
#                        if i==9: print ("a write")
                        fh.write('\n')
#                    if str(kl[ind]) not in recheck:
#                        fh.write(str(kl[ind]))
#                        fh.write('\n')
#                        recheck.append(str(kl[ind]))
                except:
                    cnt+=1
            fh.close()
    os.system('spd-say "your program has finished"')
    os.system('play --no-show-progress --null --channels 1 synth %s sine %f' % (duration, freq))            
    """
    find similarity from each cluster for every word pairs
    get average similarity from each cluster
    compute average similarity over all clusters
    """

    
    
#    clusterFiles = os.listdir(directory_cluster)
#    finalSimilarity = 0
#    avgSimilarity = 0
#    tempSim = 0
#    
#    for cfile in clusterFiles:
#        
#        fileData = pd.read_csv(str(directory_cluster)+str(cfile), sep='\n')
#        curSim = 0
#        length = (len(fileData.word_list) * (len(fileData.word_list)-1))/2 
#        for i in range(len(fileData.word_list)):
#            try:
#                w1 = MODEL[fileData.word_list[i]]
#            except:
#                w1 = size * [0]
#            #print("w1: ", w1)
#            j = i+1
#            for j in range(len(fileData.word_list)):
#                print (fileData.word_list[i], fileData.word_list[j] )
#                try:
#                    w2 = MODEL[fileData.word_list[i]]
#                except:
#                    w2 = size * [0]
#                if cosineSim(w1,w2) >= 1:
#                    # some cases similarity becomes more than zero like 1.0001
#                    tempSim = 0
#                    
#                else:
#                    curSim = curSim + tempSim
#
#                
#        avgSimilarity = avgSimilarity + (curSim / length)
#    finalSimilarity = 0.5 * avgSimilarity/ len(clusterFiles)*(len(clusterFiles)-1)
#    
#    print("Final sim for K-means: ", finalSimilarity)
#    
    
    
    
#    print("\n")
#    print("Prediction")
#    
#    Y = vectorizer.transform(["chrome browser to open."])
#    prediction = model.predict(Y)
#    print(prediction)
#    
#    Y = vectorizer.transform(["My cat is hungry."])
#    prediction = model.predict(Y)
#    print(prediction)
#    