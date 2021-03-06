# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 14:02:33 2019

@author: mdhvk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

datasets = pd.read_csv('D:\Data_science\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Data.csv')

#now we have to make a materix of independent variable which is first three coloum of dataset
#here first is the whole row and :-1 is not include which is last coloun of the dataset which is dependent variable
X = datasets.iloc[:, :-1].values
#now we are  going to create the dependent variable 
Y = datasets.iloc[:, 3].values  


#taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy = 'mean', axis = 0)
#we have to pass index 
imputer = imputer.fit(X[:,1:3])
#now we have to replace value
X[:,1:3] = imputer.transform(X[:,1:3])


#Encoding Categorical data
#now we have to make catagory of the data like Country and purchase 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#now labelencoder will give the value in number like france is equal to 0 and spain is equal to 2 and germany is equal to 1
#now to have to put this value back to array x
X[:,0]=labelencoder_X.fit_transform(X[:,0])
#now we have to create dummy value of all if there is country is present then it will show 1 else 0 
onehotcoder = OneHotEncoder(categorical_features =[0])
X= onehotcoder.fit_transform(X).toarray()
#we have to do this same for purchase coloum
labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)


#splition dataset into test and training sets
from sklearn.model_selection import train_test_split 
#there train size is means 20% of data set is used to test the data best practice is used 20 to 30% used for test the data
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=.2,random_state=0)

#feature scaling
#we should done this beacuse the salary and the age both feature have big difference which gives a problem to draw a graph
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
        #now all the values of the tarining set will come under the same range
























