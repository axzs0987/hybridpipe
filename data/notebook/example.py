import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('../input/voicegender/voice.csv')

data_x = df[df.columns[0:20]].copy()
data2 = data_x.drop(['kurt','centroid','dfrange'],axis=1).copy()

male_outlier = df[((df['meanfun'] < 0.085) \
                    | (df['meanfun'] > 0.180)) \
                    & (df['label'] == 'male')].index
female_outlier = df[((df['meanfun'] < 0.165) \
                    | (df['meanfun'] > 0.255)) \
                    & (df['label'] == 'female')].index
index_to_remove = list(male_outlier) + list(female_outlier)
data2 = data2.drop(index_to_remove,axis=0)

y = df[df.columns[-1]].values
data_y = pd.Series(y).drop(index_to_remove,axis=0)

from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(data2, data_y)

import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.linear_model.logistic import LogisticRegression
print("start running model training........")
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(Xtrain, ytrain)
y_pred = model.predict(Xtest)
score = accuracy_score(ytest, y_pred)


