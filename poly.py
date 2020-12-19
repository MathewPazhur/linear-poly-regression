# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 21:35:52 2020

@author: mathew.a.pazhur
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
# from sklearn.preprocessing import LabelEncoder
# from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

dataset=pd.read_csv("Admission_Predict.csv", encoding= 'unicode_escape')

X=dataset.iloc[:,[1,2,4,5,6]]
X2=X
poly= PolynomialFeatures(2)

X=poly.fit_transform(X)

# labelencoder=LabelEncoder()
# X['CITY']=labelencoder.fit_transform(dataset['CITY'])
# enc = OneHotEncoder(handle_unknown='ignore')

# enc_df = pd.DataFrame(enc.fit_transform(X[['CITY']]).toarray())
# X=X.join(enc_df)
# X=X.drop(columns=['CITY'])


Y=dataset.iloc[:, [8]]

X_train, X_test, y_train, y_test= train_test_split(X,Y,test_size=0.25)

regressor= LinearRegression()
regressor=regressor.fit(X_train,y_train)

y_pred_poly=regressor.predict(X_test)

X_train2, X_test2, y_train2, y_test2= train_test_split(X2,Y,test_size=0.25)

regressor2= LinearRegression()
regressor2=regressor2.fit(X_train2,y_train2)

y_pred_lin=regressor2.predict(X_test2)

print(mean_squared_error(y_test,y_pred_lin))
print(mean_squared_error(y_test,y_pred_poly))








# import statsmodels.api as sm
# Xtest= np.append(arr=np.ones((400,1)).astype(int),values=X,axis=1)

# X_opt=np.array(Xtest[:,[0,1,2,5,6]], dtype=float)
# regressorOLS=sm.OLS(exog=X_opt,endog=Y).fit()
# regressorOLS.summary()

# y_pred2=regressorOLS.predict(X_test)
