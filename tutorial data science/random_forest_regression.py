# -*- coding: utf-8 -*-
"""
Created on Mon Apr 29 22:40:11 2019

@author: mdhvk
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('D:\Machine Learning A-Z Template Folder\Part 2 - Regression\Section 8 - Decision Tree Regression\Position_Salaries.csv')
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Fitting dataset into random forest regression 
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=300,random_state =0)
regressor.fit(X,y)

#predicting result
y_pred = regressor.predict(np.array([[6.5]]))

#Graph 
X_grid = np.arange(min(X),max(X),0.1)
X_grid = X_grid.reshape((len(X_grid),1))
plt.scatter(X,y,color='red')
plt.plot(X_grid,regressor.predict(X_grid),color='blue')
plt.title('Random_forest_regression')
plt.xlabel('position')
plt.ylabel('Salary')
 