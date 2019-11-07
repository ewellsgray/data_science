# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:28:21 2019

@author: ewell
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

# read in csv file with photoreceptor parameters
df = pd.read_csv("cell_data.csv", index_col=0)

# "type" is 1 for cones, 0 for rods. This is the targe
y_all = df.type
X_all = df.copy()
X_all.drop(["type"], axis=1, inplace=True)

X_train, X_valid, y_train, y_valid = train_test_split(X_all,y_all, test_size=0.2, random_state=42)

#%%
X_train.describe()

clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
clf.fit(X_train,y_train)

y_predict = clf.predict(X_valid)

auc = roc_auc_score(y_valid, y_predict)
print('AUC (  no_scaling )=',round(auc,5))

#%%

model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
clf2 = Pipeline(steps = [('scaler',StandardScaler()),('model',model)])

clf2.fit(X_train,y_train)

y_predict = clf2.predict(X_valid)
auc = roc_auc_score(y_valid, y_predict)
print('AUC (stand.scaler)=',round(auc,5))

#%%
# Drop the slope column
X_test_no_slope = X_train.copy()
X_test_no_slope.drop(["slope"], axis=1, inplace=True) 

X_valid_no_slope = X_valid.copy()
X_valid_no_slope.drop(["slope"], axis=1, inplace=True) 

clf3 = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
clf3.fit(X_test_no_slope,y_train)

y_predict = clf3.predict(X_valid_no_slope)

auc = roc_auc_score(y_valid, y_predict)
print('AUC (  no_scaling )=',round(auc,5))

