'''
Created on Sep 26, 2017

@author: C213220
'''

import pandas as pd
import quandl 
import numpy as np
import math

from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression

df = quandl.get("WIKI/GOOGL")

# Manipulate the data sets without having to define type first
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Open', 'Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]

print(df.head())

#Set "Feature" - X axis, and "Label" - Y axis
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))

df['label'] = df[forecast_col].shift(-forecast_out)

df.dropna(inplace=True)
X = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

X = preprocessing.scale(X)

#Get training / testing sets by using 80/20 rate
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)

#Use the simplest algorithm (classifier) Support Vector Regression without any parameter
clf = svm.SVR()

#Train the classifier by using our training set
clf.fit(X_train, y_train)

#Test and validate the results by our testing set
confidence = clf.score(X_test, y_test)

#Get accuracy (confident level)
print(confidence)

#Try another algorithm: Linear Regression 
clf = LinearRegression()
#Train the classifier by using our training set
clf.fit(X_train, y_train)

#Test and validate the results by our testing set
confidence = clf.score(X_test, y_test)

#Get accuracy (confident level)
print(confidence)
