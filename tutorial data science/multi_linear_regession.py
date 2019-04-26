# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:41:57 2019

@author: mdhvk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 5 - Multiple Linear Regression\Startups.csv')
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,4].values

#We have to encode city 
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
X[:,3]=labelencoder_X.fit_transform(X[:,3])
onehotencoder = OneHotEncoder(categorical_features=[3])
X = onehotencoder.fit_transform(X).toarray()
    
#avoid dummy variable trap
X = X[:,1:]

#now we have to train our data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.2,random_state = 0)







