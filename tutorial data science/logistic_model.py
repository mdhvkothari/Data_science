# -*- coding: utf-8 -*-
"""
Created on Fri May  3 14:50:44 2019

@author: mdhvk
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv('D:\Machine Learning A-Z Template Folder\Part 3 - Classification\Section 14 - Logistic Regression\Social_Network_Ads.csv')
X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

#Splitting data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state =0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Logistic regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,Y_train)

#Predicting the test set result
Y_pred = classifier.predict(X_test)


#Now we make confusion matrix to evaluate the  accuracy of a classification
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)
#this will give output like [[65,  3],[ 8, 24]] which means that our model predict 65+24 correct prediction
#and 3+8 are incorrect

#Visulation the trainig set
















































