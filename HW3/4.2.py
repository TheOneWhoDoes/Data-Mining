# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:19:40 2021

@author: Amirhosein Khoshbin
"""

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist 
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import export_graphviz
from six import StringIO 
from IPython.display import Image  
import pydotplus
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn import svm
from sklearn import metrics

#Load Data
col_names = ['x', 'y', 'label']

data = pd.read_csv('binary_2d.csv', header=None, names=col_names)
X = data.iloc[:,0:2]
y = data.iloc[:,-1]
X = X.to_numpy()
y = y.to_numpy()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)
support_vectors = clf.support_vectors_
print("Accuracy svm:",metrics.accuracy_score(y_test, y_pred))
print("coef:", clf.coef_)
plt.scatter(X[:,0], X[:,1])
plt.scatter(support_vectors[:,0], support_vectors[:,1], color='red')

from mlxtend.plotting import plot_decision_regions
plot_decision_regions(X_test, y_test, clf=clf, legend=2)
plt.show()