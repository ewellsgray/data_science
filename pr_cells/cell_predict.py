# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 13:28:21 2019

@author: ewell
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score,recall_score

from imblearn.over_sampling import SMOTE

# read in csv file with photoreceptor parameters
df = pd.read_csv("cell_data.csv", index_col=0)

# "type" is 1 for cones, 0 for rods. This is the targe
y_all = df.type
X_all = df.copy()
X_all.drop(["type"], axis=1, inplace=True)

x_train, x_valid, y_train, y_valid = train_test_split(X_all,y_all, test_size=0.2, random_state=42)

#%%
x_train.describe()

clf = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
clf.fit(x_train,y_train)

y_predict = clf.predict(x_valid)

auc = roc_auc_score(y_valid, y_predict)
print('AUC (  no_scaling )=',round(auc,5))
recall = recall_score(y_valid, y_predict)
print('Recall( no scaling )',round(recall,5))
#%%

model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
clf2 = Pipeline(steps = [('scaler',StandardScaler()),('model',model)])

clf2.fit(x_train,y_train)

y_predict2 = clf2.predict(x_valid)
auc = roc_auc_score(y_valid, y_predict2)
print('AUC (stand.scaler)=',round(auc,5))
recall = recall_score(y_valid, y_predict2)
print('Recall(stand.scaler)',round(recall,5))

#%%
# =============================================================================
# # Drop the slope column
# X_test_no_slope = x_train.copy()
# X_test_no_slope.drop(["slope"], axis=1, inplace=True) 
# 
# x_valid_no_slope = x_valid.copy()
# x_valid_no_slope.drop(["slope"], axis=1, inplace=True) 
# 
# clf3 = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
# clf3.fit(X_test_no_slope,y_train)
# 
# y_predict3 = clf3.predict(x_valid_no_slope)
# 
# auc = roc_auc_score(y_valid, y_predict3)
# print('AUC (no_scaling no slope)=',round(auc,5))
# recall = recall_score(y_valid, y_predict3)
# print('Recall(no scaling no slope)',round(recall,5))
# =============================================================================
#%%

from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    #classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
#%%
cm = confusion_matrix(y_valid, y_predict)
plot_confusion_matrix(y_valid, y_predict, classes=['rods','cones'],
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues)


#%% Use SMOTE algorithm to over-sample the cones
# Note: MUST over-sample AFTER splitting the test/val sets so test information 
# does not "bleed" in to the validation data

sm = SMOTE(random_state=0, ratio = 1.0)
x_train_res, y_train_res = sm.fit_sample(x_train, y_train)

model = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
model.fit(x_train_res,y_train_res)
 
y_predict_res = model.predict(x_valid)
 
auc = roc_auc_score(y_valid, y_predict_res)
print('AUC (SMTOE)=',round(auc,5))
recall = recall_score(y_valid, y_predict_res)
print('Recall(SMOTE)',round(recall,5))

#%% Comgine scaling with SMOTE
# Had troupble getting it working with pipeline, but should be do-able

#pipeline_res_sc = Pipeline([("scaler",StandardScaler()),("model",model)])
#pipeline_res_sc.fit(x_train_res, y_train_res)
#y_predict_res_sc = pipeline_res_sc.predict(x_valid)
 
sc = StandardScaler()
x_train_sc = sc.fit_transform(x_train) 
x_valid_sc = sc.transform(x_valid) 

sm2 = SMOTE(random_state=0, ratio = 1.0)
x_train_sc_res, y_train_res = sm.fit_sample(x_train_sc, y_train)

model3 = LogisticRegression(random_state=0, solver='liblinear', multi_class='ovr')
model3.fit(x_train_sc_res, y_train_res)
y_predict_sc_res = model3.predict(x_valid_sc)

auc2 = roc_auc_score(y_valid, y_predict_sc_res)
print('AUC (SMTOE+rescale)=',round(auc2,5))
recall2 = recall_score(y_valid, y_predict_sc_res)
print('Recall(SMOTE+rescale)',round(recall2,5))

cm = confusion_matrix(y_valid, y_predict_sc_res)
plot_confusion_matrix(y_valid, y_predict_sc_res, classes=['rods','cones'],
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues)

#%%

clf_rf = RandomForestClassifier(n_estimators=25, random_state=0)

clf_rf.fit(x_train_sc_res, y_train_res)
y_predict_clf_rf = clf_rf.predict(x_valid_sc)

auc_rf = roc_auc_score(y_valid, y_predict_clf_rf)
print('AUC (SMTOE+rescale)=',round(auc_rf,5))
recall_rf = recall_score(y_valid, y_predict_clf_rf)
print('Recall(SMOTE+rescale)',round(recall_rf,5))

cm = confusion_matrix(y_valid, y_predict_clf_rf)
plot_confusion_matrix(y_valid, y_predict_clf_rf, classes=['rods','cones'],
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues)









