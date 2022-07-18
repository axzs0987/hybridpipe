import numpy as np 
import pandas as pd 
import os

dataset=pd.read_csv('/kaggle/input/social-network-ads/Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
Y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split

xtrain, xtest, ytrain, ytest = train_test_split(X, Y, train_size=0.8,\
     test_size=1-0.8, random_state=0)
from sklearn.preprocessing import StandardScaler

sc=StandardScaler()
x_transform=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model.logistic import LogisticRegression
print("start running model training........")
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(xtrain, ytrain)
y_pred = model.predict(xtest)
score = accuracy_score(ytest, y_pred)
