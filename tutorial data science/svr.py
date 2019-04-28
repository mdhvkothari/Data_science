# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 22:42:59 2019

@author: mdhvk
"""
#Simple Vector regressor
#SVR is a support regression machine for both linear regression machine and polynomial regression machine

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 7 - Support Vector Regression (SVR)\Position_Salaries.csv')
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values

#Fitting SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X,Y)

#we have to feature scale our data because SVM as does not have capacity to do automatic feature scailing
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_Y = StandardScaler()
X = sc_X.fit_transform(X)
Y = sc_Y.fit_transform(Y)


#Predict a new result
Y_pre = sc_Y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

#Visualization
plt.scatter(X,Y,color='red')
plt.plot(X,regressor.predict(X),color='red')
plt.title('SVR')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()



