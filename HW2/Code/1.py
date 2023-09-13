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
from sklearn import datasets
 
 
def kmeans(x,k, no_of_iterations):
    idx = np.random.choice(len(x), k, replace=False)
    #Randomly choosing Centroids 
    centroids = x[idx, :] #Step 1
     
    print(np.array(x).shape, np.array(centroids).shape)
    #finding the distance between centroids and all the data points
    distances = cdist(x, centroids ,'euclidean') #Step 2
     
    #Centroid with the minimum Distance
    points = np.array([np.argmin(i) for i in distances]) #Step 3
     
    #Repeating the above steps for a defined number of iterations
    #Step 4
    for _ in range(no_of_iterations): 
        centroids = []
        for idx in range(k):
            #Updating Centroids by taking mean of Cluster it belongs to
            temp_cent = x[points==idx].mean(axis=0) 
            centroids.append(temp_cent)
 
        centroids = np.vstack(centroids) #Updated Centroids 
         
        distances = cdist(x, centroids ,'euclidean')
        points = np.array([np.argmin(i) for i in distances])
         
    return points 
 
        #return np.mean(distances) 
#Load Data
worms = pd.read_csv('worms.csv')
print(worms)
worms = worms.iloc[: , 1:]
#worms = worms.to_numpy()
 
#Applying our function
 
#Visualize the results
wormsnp = worms.to_numpy()
print(wormsnp)
label = kmeans(wormsnp,8,100)

u_labels = np.unique(label)
for i in u_labels:
    plt.scatter(wormsnp[label == i , 0] , wormsnp[label == i , 1] , s=0.001, label = i)
plt.legend()
plt.show()

