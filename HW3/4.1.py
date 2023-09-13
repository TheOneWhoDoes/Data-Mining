# -*- coding: utf-8 -*-
"""
Created on Thu Jul  8 12:10:36 2021

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
import nltk
import string
import re
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn import metrics
import math
from sklearn import svm

porter_stemmer = PorterStemmer()
stopwords = nltk.corpus.stopwords.words('english')
stopwords.append('')
stopwords.append('\t')
def remove_punctuation(text):
    punctuationfree="".join([i for i in text if i not in string.punctuation])
    return punctuationfree

def tokenization(text):
    text = remove_punctuation(text)
    tokens = re.split(' ', text)
    return tokens[: len(tokens)-2]

def remove_stopwords(text):
    output= [i for i in text if i not in stopwords]
    return output

def lower_words(text):
    output= [i.lower() for i in text]
    return output

def stemming(text):
    stem_text = [porter_stemmer.stem(word) for word in text]
    return stem_text

def tagging(text):
    tagged = nltk.pos_tag(text)
    return tagged

def computeReviewTFDict(review):
    """ Returns a tf dictionary for each review whose keys are all
    the unique words in the review and whose values are their
    corresponding tf.
    """
    # Counts the number of times the word appears in review
    reviewTFDict = {}
    for word in review:
        if word in reviewTFDict:
            reviewTFDict[word] += 1
        else:
            reviewTFDict[word] = 1
    # Computes tf for each word
    for word in reviewTFDict:
        reviewTFDict[word] = reviewTFDict[word] / len(review)
    return reviewTFDict


def computeTF(wordDict, doc):
    tfDict = {}
    corpusCount = len(doc)
    for word, count in wordDict.items():
        tfDict[word] = count/float(corpusCount)
    return(tfDict)

def computeIDF(docList):
    idfDict = {}
    N = len(docList)
   
    idfDict = dict.fromkeys(docList[0].keys(), docList[0].values())
    for word, val in docList[0].items():
        idfDict[word] = math.log10(N / float(val))
        
    return(idfDict)


def computeTFIDF(tfBow, idfs):
    tfidf = {}
    for word, val in tfBow.items():
        tfidf[word] = math.log10(1 + val)*idfs[word]
    return(tfidf)



col_names = ['feature', 'label']
             
import os
path = "IMDB_review_labels.txt"

temp = open(path).read()

a_labels, a_texts = [], []
for i, line in enumerate(temp.split('\n')):
    content = line.split('|')
    if len(content) > 1:
        a_texts.append(content[0])
        a_labels.append(content[1])

print(len(a_labels))

data = pd.DataFrame()
data['label'] = a_labels
data['feature'] = a_texts



#data = pd.read_csv('IMDB_review_labels.txt' , sep = "|", names=col_names, header=None)

#review = data['feature']

data['feature']= data['feature'].apply(lambda x: tokenization(x))


from pprint import pprint

data['feature']= data['feature'].apply(lambda x: lower_words(x))

data['feature']= data['feature'].apply(lambda x:remove_stopwords(x))

data['feature']= data['feature'].apply(lambda x:stemming(x))

total = data['feature'][0]

for i in range (1,len(data['feature'])):
    total= set(total).union(set(data['feature'][i]))

wordDict = [dict.fromkeys(total, 0)]*len(data['feature'])

for i in range(0, len(data['feature'])):
    for word in data['feature'][i]:
        wordDict[i][word] = wordDict[i][word] + 1
        

        

tf = [0]*len(data['feature'])
for i in range(0, len(data['feature'])):
    tf[i] = computeTF(wordDict[i], data['feature'][i])


idfs = computeIDF([wordDict[i] for i in range(0, len(data['feature']))])



idflist = [0]*len(data['feature'])
for i in range(0, len(data['feature'])):
#running our two sentences through the IDF:
    idflist[i] = computeTFIDF(tf[i], idfs)

print(type(idflist[1]))

vector = []
for i in range(0, len(data['feature'])):
    temp = []
    for key, value in idflist[i].items():
        temp.append(value)
    vector.append(temp)

#data['feature']=data['feature'].apply(lambda x:tagging(x))

#print(computeReviewTFDict(review))

X_train, X_test, y_train, y_test = train_test_split(vector, data['label'], test_size=0.1)

#tf_vectorizer = CountVectorizer()
gnb = GaussianNB()

#Train the model using the training sets
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy naive bayes:",metrics.accuracy_score(y_test, y_pred))

#Create a svm Classifier
clf = svm.SVC() # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred = clf.predict(X_test)

print("Accuracy svm:",metrics.accuracy_score(y_test, y_pred))