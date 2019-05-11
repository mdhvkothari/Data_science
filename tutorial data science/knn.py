# -*- coding: utf-8 -*-
"""
Created on Sat May 11 15:11:33 2019

@author: mdhvk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('D:\Machine Learning A-Z Template Folder\Part 3 - Classification\Section 15 - K-Nearest Neighbors (K-NN)\Social_Network_Ads.csv')

X = dataset.iloc[:,[2,3]].values
Y = dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Fitting into Training set
from sklearn.neighbors import KNeighborsClassifier
#by default it have 5 neighbor and we use euclidean distance which is under root x1-x2 square and y1-y2 square
classifier = KNeighborsClassifier(n_neighbors= 5,metric = 'minkowski',p=2)
classifier.fit(X_train,Y_train)

Y_pred = classifier.predict(X_test)

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)




