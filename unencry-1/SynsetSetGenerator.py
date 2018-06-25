#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  8 01:50:32 2018

@author: c00300901
"""


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 19 15:24:01 2018

@author: zobaed  Synset generator for all files. 
"""

from nltk.corpus import wordnet as wn

from nltk.stem import WordNetLemmatizer
from nltk import pos_tag, word_tokenize
from nltk.corpus import stopwords
import os
import sys

def penn_to_wn(tag):
    if tag.startswith('J'):
        return wn.ADJ
    elif tag.startswith('N'):
        return wn.NOUN
    elif tag.startswith('R'):
        return wn.ADV
    elif tag.startswith('V'):
        return wn.VERB
    return None


synclear=[]
synsets=[]

directory= "/home/c00300901/Desktop/key files data sets/all_bbc_mod/"
FileList = os.listdir(directory)
unique=[]
with open ("pos_synset_words_allbbc.txt", "w")as fw:
    i=0
    for file in FileList:
        i+=1
        with open (directory+file, "r") as fr:
            sentence=fr.read()
#            print(i)

    
    #sentence = "I am going to buy some gifts"
    
    
            
            word_tokens = word_tokenize(sentence)
            stop_words = set(stopwords.words('english'))
            filtered_sentence = [w for w in word_tokens if not w in stop_words]
            
            tagged = pos_tag(filtered_sentence)
            
            
            lemmatzr = WordNetLemmatizer()
            #        sys.exit()
            for token in tagged:
                wn_tag = penn_to_wn(token[1])
                if not wn_tag:
                    continue
            
                lemma = lemmatzr.lemmatize(token[0], pos=wn_tag)
                try:
                    
                    synsets.append(wn.synsets(lemma, pos=wn_tag)[0])
                    a= str(wn.synsets(lemma, pos=wn_tag)[0])
                    a=str(synsets[-1]).strip('Synset(')
                    a=a[1:]
                    a=a[:-1]
                    a=a[:-1]
                    if (a not in unique):
                        fw.write(a)
#                        print(a)
                        fw.write("\n")
                        unique.append(a)
                except:
                    c=4
    


  


