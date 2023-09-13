# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import math
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
from numpy import linalg
from scipy.spatial import distance


class KNN:

    def __init__(self, k, dist_kind='co'):

        self.k = k
        self.dist_kind = dist_kind


    def cosin_dist(self, a, b):
        return 1-sum([i*j for i,j in zip(a, b)])/(math.sqrt(sum([i*i for i in a]))* math.sqrt(sum([i*i for i in b])))
    
    
    def euclidean_dist(self, array1, array2):

       # cdist(array1, array2 ,'euclidean')

        array1 = np.array(array1)

        array2 = np.array(array2)

        return linalg.norm(array1 - array2)


    def k_neighbors(self, test_row):

        distances = []
        if self.dist_kind == 'eu':
            for i in range(len(self.X_train)):

                distance = self.euclidean_dist(test_row, self.X_train[i])

                distances.append((distance, self.y_train[i]))
        
        if self.dist_kind == 'co':
            for i in range(len(self.X_train)):

                distance = self.cosin_dist(test_row, self.X_train[i])

                distances.append((distance, self.y_train[i]))
                

        distances.sort()

        return distances[:self.k]


    def get_nn(self):

        self.X_train = np.array(self.X_train)

        self.X_test = np.array(self.X_test)

        self.y_train = np.array(self.y_train)

        neighbors = []

        for j in range(len(self.X_test)):

            neighbors.append(self.k_neighbors(self.X_test[j]))

        return neighbors


    def vote_count(self, lst):
        """
        returns dictionary containing counts of each element of list
        """

        lst_count = dict()

        for element in lst:

            if element in lst_count:

                lst_count[element] += 1

            else:

                lst_count[element] = 1

        return lst_count


    def fit(self, X_train, y_train):

        self.X_train = X_train

        self.y_train = y_train
        


    def predict(self, X_test):

        self.X_test = X_test

        nbrs = self.get_nn()

        predictions = []

        for row in nbrs:

            dist, labels = zip(*row)

            label_dict = self.vote_count(labels)

            predictions.append(max(label_dict, key = label_dict.get))

        return predictions

    def evaluate(self, y_pred, y_test):

        count = 0

        for act, pred in zip(y_pred, y_test):
            if act == pred:
                count += 1

        return count / len(y_test)
 

def main():
    #Load Data
    train = pd.read_csv('segmentationTrain.csv')
    X_train =train.drop(columns=['LABEL'])
    y_train = train['LABEL']
 
    test = pd.read_csv('segmentationTest.csv')
    X_test =test.drop(columns=['LABEL'])
    y_test = test['LABEL']
 
    print(X_train, X_test, y_train, y_test)
    #Applying our function
    #nn = KNN(3)
    #nn.fit(X_train, y_train)
    #print(nn)
    #print(nn.evaluate(nn.predict(X_test), y_test))
    
    for k in range(1, 9):
        for t in range(1, 3):
            if t == 1:
                nn = KNN(k, 'co')
                nn.fit(X_train, y_train)
                plt.scatter(k , nn.evaluate(nn.predict(X_test), y_test) , label = k)    
            if t == 2:
                nn = KNN(k, 'eu')
                nn.fit(X_train, y_train)
                plt.scatter(-k , nn.evaluate(nn.predict(X_test), y_test) , label = -k)    
    plt.legend()
    plt.show()


if __name__ == '__main__':
    main()


