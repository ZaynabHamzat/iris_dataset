# -*- coding: utf-8 -*-
"""
Created on Mon Feb  10 19:58:15 2020

@author: zaynab
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from sklearn import tree

   #importing dataset
R = pd.read_csv("iris.csv")

    #checking for missing values
R.isnull().any()


   #splitting dataset into train and test set
X = R[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]].values
Y = R[["Species"]].values
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.30, random_state = 0)


   #building decision tree model
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, Y_train)


   #predicting for test dataset
Y_pred = clf.predict(X_test)


   #model accuracy
print("Accuracy Score :", metrics.accuracy_score(Y_test,Y_pred))


   #visualizing decision tree
fn = ["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]
cn = ["Iris-setosa", "Iris-versicolor", "Iris-virginica"]
fig, axes = plt.subplots(nrows = 1, ncols = 1, figsize = (4,4), dpi = 300)
tree.plot_tree(clf, feature_names = fn, class_names = cn, filled = True)
fig.savefig("specie.png")

