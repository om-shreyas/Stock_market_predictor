# -*- coding: utf-8 -*-
"""
Created on Sat Oct  1 15:47:37 2022

@author: Lenovo
"""
import warnings
warnings.filterwarnings("ignore")

import yfinance as yf
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from data_transformer import data_transform

stock = "MSFT"

final_data = data_transform(stock)

X = final_data.drop(["Answer Open"],axis=1)
X = final_data.drop(["Answer Close"],axis=1)

y = final_data["Answer Open"]

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet

base_elastic_model = ElasticNet()
param_grid = {'alpha':[0.1,1,5,10,50,100],'l1_ratio':[.1, .5, .7, .9, .95, .99, 1]}
grid_model = GridSearchCV(estimator=base_elastic_model,param_grid=param_grid,scoring='neg_mean_squared_error',cv=5,verbose=1)
grid_model.fit(scaled_X_train,y_train)

y_pred = grid_model.predict(scaled_X_test)

error = (((y_pred-y_test)/y_test)*100).mean()

print(100-error)

fig = plt.figure()
plt.figure(figsize=(10,10))
plt.scatter(y_test, y_pred, c='crimson')
plt.yscale('log')
plt.xscale('log')
p1 = max(max(y_test), max(y_pred))
p2 = min(min(y_test), min(y_pred))
plt.plot([p1, p2], [p1, p2], 'b-')
plt.xlabel('True Values', fontsize=15)
plt.ylabel('Predictions', fontsize=15)
plt.axis('equal')
plt.savefig("Predictions.png")