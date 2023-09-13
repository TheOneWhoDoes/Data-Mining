# -*- coding: utf-8 -*-
"""
Created on Fri Apr 23 09:57:59 2021

@author: Amirhosein Khoshbin
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
from sklearn import metrics
import matplotlib
from sklearn import datasets
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA
 

def dbscan(X, eps, min_samples):
    db = DBSCAN(eps=eps, min_samples=min_samples)
    db.fit(X)
    for i in range(0, len(db.labels_)):
        if db.labels_[i] != -1:
           # print('mmd')
           break
    y_pred = db.fit_predict(X)
    plt.scatter(X[:,0], X[:,1], s=0.009, c=y_pred, cmap='Paired')
    plt.title("DBSCAN" + "eps: " + str(eps) + " min_samples: " + str(min_samples))
    
#Load Data
matplotlib.rc('figure', figsize=[10, 9])

worms = pd.read_csv('worms.csv')
worms = worms.iloc[: , 1:]

print(worms)

#Applying our function
 
#Visualize the results
X = worms.to_numpy()

dbscan(X, 16.6, 29)





