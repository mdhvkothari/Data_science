# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 23:31:38 2019

@author: mdhvk
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('D:\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 6 - Polynomial Regression\Position_Salaries.csv')
#we have to convert X into matrix hence we use 1:2  while Y is a vector
X = dataset.iloc[:,1:2].values
Y = dataset.iloc[:,2].values
#we don't need there training set or test set because we don't have enough data and we want our prediction more accurate

#Linear regression model
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X,Y)

#Polynomial regression model
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
#after seeing grap if we increase degree more accurate graph will
#this will convert X features into a another matrix which have polynomial degree
X_poly = poly_reg.fit_transform(X) 
poly_reg.fit(X_poly,Y)
#now we have to make a linear regression and fit it into polynomial model
lin_reg2 =LinearRegression()
lin_reg2.fit(X_poly,Y)

#Visulising the Linear regression
plt.scatter(X,Y,color='Red')
plt.plot(X,lin_reg.predict(X),color = 'blue')
plt.title('Linear Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()


#Visualising Polynomial regression
plt.scatter(X,Y,color='Red')
plt.plot(X,lin_reg2.predict(poly_reg.fit_transform(X)),color = 'blue')
plt.title('Plynomial Regression')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()











