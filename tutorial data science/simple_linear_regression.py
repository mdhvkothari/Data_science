# -*- coding: utf-8 -*-
"""
Created on Thu Apr 25 19:13:28 2019

@author: mdhvk
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\Data_science\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 4 - Simple Linear Regression\Salary_Data.csv')
#now we have to preprocess our data for this we have to seperate dependent and independent varaible
#X for independent variable and X is the matrix
X = dataset.iloc[:,-1].values.reshape(-1,1)
#Y for dependent variable and Y is the vector
Y = dataset.iloc[:,1].values.reshape(-1,1)

# now spliting dataset into test and training set
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y, test_size=1/3)
#now we train our model with the help of X_train and Y_train

#now we have to feature scaling of our data
''' form sklearn.preprecossing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.fit_transform(X_test)
sc_Y = StandardScaler()
Y_train = sc_Y.fit_transform(Y_train)'''

#Now we fitting Simple Linear Regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#predicting the test set results
Y_perdict = regressor.predict(X_test)

#plotting a graph
plt.scatter(X_train,Y_train,color='red')
plt.plot(X_train, regressor.predict(X_train),color='blue')
plt.title('Salary Vs Experience')
plt.xlabel('Year of Experience')
plt.ylabel('Salary') 







