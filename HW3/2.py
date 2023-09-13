# -*- coding: utf-8 -*-
"""
Created on Wed Jun 30 19:30:28 2021

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

def prepare_inputs(X_train, X_test):
	oe = OrdinalEncoder()
	oe.fit(X_train)
	X_train_enc = oe.transform(X_train)
	X_test_enc = oe.transform(X_test)
	return X_train_enc, X_test_enc


#Load Data
col_names = ['price', 'keeping', 'doors', 'volume', 'back', 'security', 'label']

data = pd.read_csv('car.csv', header=None, names=col_names)
X = data.iloc[:,0:6]
y = data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

X_train, X_test = prepare_inputs(X_train, X_test)

# Create Decision Tree classifer object
clf = DecisionTreeClassifier()

# Train Decision Tree Classifer
clf = clf.fit(X_train,y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)


feature_cols = ['price', 'keeping', 'doors', 'volume', 'back', 'security']


print("Accuracy:",metrics.accuracy_score(y_test, y_pred), "score: ",
      clf.score(X_test,y_test))

print(classification_report(y_test, y_pred))

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['vgood','good', 'acc', 'unacc'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())

