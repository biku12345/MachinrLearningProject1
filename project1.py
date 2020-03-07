# -*- coding: utf-8 -*-
"""
Created on Sat Mar  7 22:43:13 2020

@author: sourav kumar
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sms
import numpy as np
#i have tolaod the data
dataset = pd.read_csv('games.csv')
#print names of the columns and shapeof the dataset
print(dataset.columns)
print(dataset.shape)

#making the histogram of all the rating  in average rating columns
plt.hist(dataset['average_rating'])
plt.show()

#print first row of all the games whose scaore is zero
print(dataset[dataset['average_rating']==0].iloc[0])

#print first row of all the games with score greater than zero
print(dataset[dataset['average_rating']>0].iloc[0])

#remove any row without user_review
dataset = dataset[dataset['users_rated']>0]
#remove all rows with missing values
dataset= dataset.dropna(axis=0)

#make histogram of average rating
plt.hist(dataset['average_rating'])

#printing the column of the datset
print(dataset.columns)

#correlation matrix
cormat = dataset.corr()
fig = plt.figure(figsize = (12,9))

sms.heatmap(cormat,vmax = .8, square = True)
plt.show()

#get all the column the datset
columns = dataset.columns.tolist()

#filter remove all the columns of the dataset
columns = [c for c in columns if c not in ['bayes_average_rating','id','average_rating','type','name']]

#store the variable we will be predicting on
target = 'average_rating'

#genrate traing and test  dataset
from sklearn.model_selection import train_test_split
#genrating the training set
train = dataset.sample(frac = 0.8, random_state = 1)
#genrating the test state
test = dataset.loc[~dataset.index.isin(train.index)]
#printing the shape of traing and test
print(train.shape)
print(test.shape)

#now i am umporing the linearregression model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
lr = LinearRegression()
#Now we are fitting thel training  datsaet
lr.fit(train[columns],train[target])
#now the prediction time

pred = lr.predict(test[columns])
#now we are calculating the mean_squarred error
mean_squared_error(pred,test[target])

#now we areimporing the randomforest

from sklearn.ensemble import RandomForestRegressor
#initialzing the models
rf = RandomForestRegressor(n_estimators = 100,min_samples_leaf = 1,random_state = 1)

#fitting the data in the rf
rf.fit(train[columns],train[target])

#Now the prediction time
pred1 = rf.predict(test[columns])








#