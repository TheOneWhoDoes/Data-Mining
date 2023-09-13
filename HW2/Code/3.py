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
 
#Defining our function 
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
 
 
 
def meanError(data, label):
    u_labels = np.unique(label)
    for i in u_labels:
        cmean = np.mean(data[label == i], axis = 0)
        x = []
        [x.append(cmean[i]) for i in range(len(cmean))]
        x = [x]
        distances = cdist(data[label == i], x ,'euclidean')
        print("mean distance for cluster" + str(i) + " " + str(np.mean(distances)))
        plt.scatter(data[label == i , 0] , data[label == i , 1] , label = i)
        
    plt.legend()
    plt.show()
        #return np.mean(distances) 
#Load Data
iris = pd.read_csv('iris.csv')
iris =iris.drop(columns=['Species'])
iris = iris.to_numpy()
 
#Applying our function
label = kmeans(iris,1,1000)
 
#Visualize the results
 



meanError(iris, label)


