# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 17:41:57 2019

@author: mdhvk
"""
#Methods to multi linear regression-
# 1. Backward Elimination
# 2.Forward Selection
# 3.Bidirectional Elimination
# 4. Score Comparison
 
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

#fit multiple regression into training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)

#predict the test set result
Y_pred = regressor.predict(X_test)

#Building the optimal model using Backward elimination
import statsmodels.formula.api as sm
X = np.append(arr=np.ones((50,1)).astype(int),values = X,axis =1)
# Now we have to create a new optimal matrix
#basically we remove unsignifcant data   
# we have to select the Significance level (e.g = .05) and if P value to lager than this value we have to remove the  predictor
X_opt = X[:,[0,1,2,3,4,5]]
regressor_OLS = sm.OLS(endog= Y,exog=X_opt).fit()
print(regressor_OLS.summary())
# 1 have larger p value than SL(Significance level) then we have to delete it
X_opt = X[:,[0,1,3,4,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
print(regressor_OLS.summary()) 
# again 1 is greater than SL
X_opt = X[:,[0,3,4,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
print(regressor_OLS.summary()) 
# again 2  is greater than SL 
X_opt = X[:,[0,3,5]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
print(regressor_OLS.summary()) 
#now 5 is greater than SL
X_opt = X[:,[0,3]]
regressor_OLS = sm.OLS(endog=Y,exog=X_opt).fit()
print(regressor_OLS.summary()) 





















