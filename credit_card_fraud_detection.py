# -*- coding: utf-8 -*-
"""
Created on Fri Mar  1 16:25:21 2024

@author: nikhilve
"""

import pandas as pd

data= pd.read_csv('fraudTrain.csv')

data=data.iloc[:,2:]


# Splitting the comma-separated string into a list of integers
data['merchant'] = data['merchant'].str.split(',').apply(lambda x: [str(i) for i in x])
data['job'] = data['job'].str.split(',').apply(lambda x: [str(i) for i in x])

# Unnesting the list to individual rows
data = data.explode('merchant')
data = data.explode('job')

# Checking for repetations in each column.
a=data["merch_lat"].value_counts()

#Label encoding categorical features

columns_to_endode=['merchant', 'category', 'gender', 'job']

from sklearn.preprocessing import LabelEncoder
label = LabelEncoder()

for columns in columns_to_endode:
    data[columns]= label.fit_transform(data[columns])

#Dropping un-necessary columns from the data set.
columns_to_drop = ['first', 'last', 'trans_num', 'dob', 'street', 'city', 'state']
data= data.drop(columns=columns_to_drop)
#data= data.drop(['first', 'last', 'trans_num', 'dob', 'street', 'city', 'state'], axis=1)

#Dividing my features in X and target in Y.
X=data.iloc[:,:-1]
Y=data.iloc[:,-1:]

#Splitting my data in training and testing data. used test_train_split from sklearn kit.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#A K-Nearest Neighbors (KNN) model with k=5 is created and fitted on the training data. Predictions are made on the test set.
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train.values.ravel())
#model.fit(x_train, y_train)
yp=model.predict(x_test)

#Checking the accuracy and metrics
from sklearn import datasets, linear_model, metrics
print("Logistic Regression model accuracy(in %):", metrics.accuracy_score(y_test, yp)*100)

#Calculating confusion matrix
from sklearn.metrics import confusion_matrix
results = confusion_matrix(y_test, yp)
print ('Confusion Matrix :')
print(results)

#Checking classification Report(precision, recall, f1-score)
from sklearn.metrics import classification_report
c_report=classification_report(yp,y_test)
print(c_report)

#Handling imbalanced data using smote
from collections import Counter
counter = Counter(Y)
from imblearn.over_sampling import SMOTE
smote = SMOTE()
x_train_smote, y_train_smote = smote.fit_resample(x_train.astype("float"), y_train)

#Calculating accuracy in % after smote. 
model.fit(x_train_smote,y_train_smote)
yp=model.predict(x_test)
print("Logistic Regression model accuracy(in %):",
metrics.accuracy_score(y_test, yp)*100)

#Confusion matrix gives us TP=You predicted positive and it’s true, TN=You predicted negative and it’s true, FP=You predicted positive and it’s false and FN=You predicted negative and it’s false.
from sklearn.metrics import confusion_matrix
results = confusion_matrix(y_test, yp)
print ('Confusion Matrix :')
print(results)

#Classification report
from sklearn.metrics import classification_report
c_report=classification_report(y_test,yp)
print(c_report)


