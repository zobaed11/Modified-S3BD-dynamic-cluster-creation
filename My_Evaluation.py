#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 18:17:31 2018

@author: c00300901
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 00:37:36 2018

@author: c00300901
"""

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









##########################################
###################

#directory= "/home/c00300901/eclipse-workspace/S3BD-JasonThesis/cloud/cloudserver/utilities_original/modified/"
#direct= "/home/c00300901/eclipse-workspace/S3BD-JasonThesis/cloud/cloudserver/utilities/modified_446_clusters/"
direct2= "/home/c00300901/PycharmProjects/clusters_sota_allbbc/10 cluster corrected/"



#FileList = os.listdir(directory)

#####################################
###########File Creation ############
####################################

#for file in FileList:
#    big_list=[]
#    with open(directory+file,'r') as f:
#        count=0
#        list=[]
#        list2=[]
#        list3=[]
#        big=""
#        for line in f:
#            text = line.split('|')[0]
#            big=big+text+"\n"
#    with open(direct+file,'w') as w:
#        w.write(str(big))
#        
        


clusterFiles = os.listdir(direct2)
#clusterFiles = os. listdir ("/home/c00300901/Desktop/results_sota2/rfc/cluster 10/")
col_name= ["words"]
list_all=[]
k=0



model = gensim.models.KeyedVectors.load_word2vec_format('/home/c00300901/eclipse-workspace/S3BD-JasonThesis/cloud/cloudserver/GoogleNews-vectors-negative300.bin', binary=True) 
for cfile in clusterFiles:
    fileData = pd.read_csv(str(direct2)+str(cfile), sep='\n', names=col_name)
    word_list=[]
    word_list=fileData['words'].values.tolist()
    
    i=0
    s=0
    list3 = []
    lea=0  #No word set make lea++ to subtract from list3's len
    temp_list = []
    temp_list = word_list[:]

    for w1 in word_list:
        
        temp_list = word_list[:]
        word1= str(w1)

        
        
        if " " in word1:
            word1= word1[:word1.index(" ")]
        if "-" in word1:
            word1= word1[:word1.index("-")]
        if "_" in word1:
            word1= word1[:word1.index("_")]
        word1=correction(word1)
        
        
        j=i+1
        
#        if(j < len(temp_list)):
        for w2 in temp_list[j:]:
            word2= str(w2)
#                print (word2)
#                sys.exit()
            
            
            if " " in word2:
                word2= word2[:word2.index(" ")]
            if "_" in word2:
                word2= word2[:word2.index("_")]
            if "-" in word2:
                word2= word2[:word2.index("-")]
            word2=correction(word2)
            
            try:
#                print (word1, word2)
                s=(model.similarity(word1, word2))
                
#                wordFromList1 = wordnet.synsets(word1)
#                wordFromList2 = wordnet.synsets(word2)
#                if wordFromList1 and wordFromList2: #Thanks to @alexis' note
#                    s = wordFromList1[0].wup_similarity(wordFromList2[0])
         
                list3.append(s)
            except:
                s=0
                list3.append(s)
                lea+=1;
        i+=1
           
    
    clean = [x for x in list3 if x != None]
#    print(sum(clean)/len(list3)) 
    if (sum(clean)/len(list3) <0.08):
#        print(k, cfile)
        k+=1
    lea=len(list3)-lea
    list_all.append(sum(clean)/lea)
    clean = []    
        
        
        
print("ALL")
print (sum(list_all)/(len(list_all)))
        
