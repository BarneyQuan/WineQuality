# -*- coding: utf-8 -*-
"""
Created on Tue Feb  1 11:04:44 2022

@author: 85384
"""

# machine learning library
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


# library to work with data
import pandas as pd

# libraries to plot confusion matrices
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()


# Import data
winequality = pd.read_csv(r"C:\\Users\85384\Desktop\Econ 2824\HW1\assignment1_winequality.csv")

winequality.info()
#winequality.isnull.sum()

winequality.mean()
winequality.describe()
winequality.head(10)
winequality.shape
winequality["quality"].unique()
#Cleaning dataset
winequality = winequality[winequality["fixed acidity"] > 0]
winequality = winequality[winequality["density"] > 0]
winequality = winequality[winequality["chlorides"] < 10]
winequality = winequality[winequality["pH"] < 20]
winequality = winequality.dropna()

#Check the distribution of this dataset
winequality["quality"].value_counts()
import seaborn as sbn; sbn.set()
sbn.countplot(x = "quality", data = winequality)
plt.show()

#Create the loop to check the quality of the wine compare to the other factors
winequality_plot = winequality.select_dtypes([np.int, np.float])

for i, col in enumerate(winequality_plot.columns):
    plt.figure(i)
    sns.barplot(x = "quality", y = col, data = winequality_plot)
  
plt.figure(figsize=(15,5))
sns.boxplot(x = "quality", y = "alcohol", data = winequality)


sns.lmplot(x = "alcohol", y = "quality", data = winequality)

#check the correlation between the factors
plt.figure(figsize=(20,20))
sns.heatmap(winequality.corr(), color = "k", annot = True)

sns.lmplot(x = "total sulfur dioxide", y = "free sulfur dioxide", data = winequality)

#Creating Classification bins
bins = (2, 6.5, 8)
group_names = ["good", "bad"]
winequality["quality"] = pd.cut(winequality["quality"], bins = bins, labels = group_names)

from sklearn.preprocessing import LabelEncoder

# Assigning a label to our quality variable
label_quality = LabelEncoder()


# Now changing our dataframe to reflect our new label
winequality["quality"] = label_quality.fit_transform(winequality["quality"])


# Create the X, Y, Training and Test
train, test = train_test_split(winequality, test_size = 0.2)

xtrain = train.drop("quality", axis = 1)
ytrain = train.loc[:, "quality"]
xtest = test.drop("quality", axis = 1) 
ytest = test.loc[:, "quality"]


#y = train.loc[:, "quality"]
#xtrain, ytrain, xtest, ytest = train_test_split(x, y, test_size = 0.2, random_state=0)

# Define the naive Bayes Classifier
model = GaussianNB()

# Train the model 
model.fit(xtrain, ytrain)

# Predict Output using the test data
pred = model.predict(xtest)

# Plot Confusion Matrix
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt="d", cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')

#store an array of predicted and actual labels
d = {"quality":ytest, "prediction":pred}

# turn the array into a data frame and print it
output = pd.winequality(data=d)
output

# Create the X, Y, Training and Test
train, test = train_test_split(winequality, test_size = 0.2)


xtrain = train.drop("quality", axis = 1)
ytrain = train.loc[:, "quality"]
xtest = test.drop("quality", axis = 1) 
ytest = test.loc[:, "quality"]



#Creating Classification bins
bins = (2, 6.5, 8)
group_names = ["good", "bad"]
winequality["quality"] = pd.cut(winequality["quality"], bins = bins, labels = group_names)

from sklearn.preprocessing import LabelEncoder

# Assigning a label to our quality variable
label_quality = LabelEncoder()


# Now changing our dataframe to reflect our new label
winequality["quality"] = label_quality.fit_transform(winequality["quality"])

import pandas as pd
from sklearn.naive_bayes import BernoulliNB
# Define the naive Bayes Classifier
model = BernoulliNB()

# Train the model 
model.fit(xtrain, ytrain)

# Predict Output using the test data
pred = model.predict(xtest)

# Plot Confusion Matrix
mat = confusion_matrix(pred, ytest)
names = np.unique(pred)
sns.heatmap(mat, square=True, annot=True, fmt="d", cbar=False,
            xticklabels=names, yticklabels=names)
plt.xlabel('Truth')
plt.ylabel('Predicted')

#store an array of predicted and actual labels
d = {"quality":ytest, "prediction":pred}

# turn the array into a data frame and print it
output = pd.winequality(data=d)
output






