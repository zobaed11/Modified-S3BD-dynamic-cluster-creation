
"""
Created on Sat May 19 18:17:31 2018

@author: c00300901

"""

import sys
import os
import numpy as np
import pandas as pd
import os
import pandas as pd
import json
from pprint import pprint



if __name__ == "__main__":
    number_of_cluster = 165
#    unique=[]
    with open ("/home/c00300901/Desktop/results_sota2/clusters_wn_wv_k165_allbbc.json", "r") as f:
       data = json.load(f) 
       print(data["cluster1"][0])
       for i in range(number_of_cluster):
           with open ("/home/c00300901/Desktop/results_sota2/all bbc/cluster"+" "+str(number_of_cluster)+"/cluster"+str(i)+".txt", "w") as ft:
               with_pos=data[str("cluster"+str(i))]
               for l in with_pos:
                   l= l[:l.index(".")]
#                   if (l not in unique):
                   ft.write(l)
                   ft.write("\n")
#                       unique.append(l)
                
    
    