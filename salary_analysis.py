# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 15:55:36 2020

@author: ewell
"""

import pandas as pd
import scipy.stats as stats
import numpy as np
import os
import matplotlib.pyplot as plt

#%%
path = 'C:\\Users\\ewell\\Documents\\CV\\nationwide data scientist specialist'
df_o = pd.read_csv(os.path.join(path,"salaries.csv"))

#%% Question 1. What is the percentage of records are Assistant Professors with 
#               less than 5 years of experience?

# make a working copy
df = df_o.copy()
num_all_records = len(df) # 397
print(df.shape) # (397,6)

# filter dataframe for rank == AsstProf and yrs.service <5

# Rename the columns/variables with dots, for convenience
# Renamed "rank" to "title" due to confile with "rank" used a df method
df.rename(columns={"yrs.since.phd":"years_phd","yrs.service":"years_service","rank":"title"}, inplace=True)
df.columns

# Check to make sure consistent entry for the rank and years_service variables
df["title"].value_counts()
df["years_service"].value_counts()

# Filter and count number of entries
df_filt1 = df.loc[(df.title=="AsstProf") & (df.years_service<5)]
print(df_filt1)
print(len(df_filt1))
num_filterd = len(df_filt1)
percentage = (num_filterd / num_all_records) * 100


#%% (2) Is there a statistically significant difference between female and male salaries?

# Check to make sure consistent entry for the salary and sex variables
df["salary"].value_counts()
df["sex"].value_counts() # 39 Female, 358 Male

sal_f = df.salary.loc[df.sex=="Female"]
sal_f.shape # (39,)

sal_m = df.salary.loc[df.sex=="Male"]
sal_m.shape # (358,)

# Two-sided independent-sample t-test, unequal variance (Welches t-test)
t_stat, p_val = stats.ttest_ind(sal_m, sal_f, equal_var=False)
print("t stat: "+str(round(t_stat,4)))
print("p val: "+str(round(p_val,4)))

#%%
# -- PREPARE THE DATA
#%% binarize catagorical data

df.loc[df.discipline=="A","discipline"] = 0
df.loc[df.discipline=="B","discipline"] = 1

df.loc[df.sex=="Male","sex"] = 0
df.loc[df.sex=="Female","sex"] = 1

#%% split x and y
x = df.iloc[:,0:-1]
y = df.iloc[:,-1]

#%% Model --Linear Regression, no scaling

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn import metrics
log_class = LogisticRegression()

#%%
def encode_column(encoding_type, col_name):
    
    if encoding_type=='label':
        le=LabelEncoder()
        le.fit(x[col_name])
        title_order = list(le.classes_)
        x[col_name] = le.fit_transform(x[col_name])
        print("Label Encoded")
    if encoding_type=='ordinal':
        oe=OrdinalEncoder()
        oe.fit(x[col_name])
        title_order = list(oe.classes_)
        x[col_name] = oe.fit_transform(x[col_name])
        print("Ordinal Encoded")

    return
#%%
#encoding_type = 'label'
encode_column("label","title")
#%%
scalar = StandardScaler()
xc = scalar.fit_transform(x)
#%%

#%%
xc_train, xc_val, yc_train, yc_val = train_test_split(xc,y,test_size=0.2, random_state=0)
lin_reg_c = LinearRegression()
lin_reg_c.fit(xc_train,yc_train).score(xc_train,yc_train)
yc_pred = lin_reg_c.predict(xc_val)

print(metrics.mean_absolute_error(lin_reg_c.predict(xc_train),yc_train))
#metrics.r2_score(lin_reg_c.predict(xc_train),yc_train)
print(metrics.mean_absolute_error(yc_val,yc_pred))
#metrics.r2_score(yc_val,yc_pred)

#%%
x_train, x_val, y_train, y_val = train_test_split(x,y,test_size=0.2, random_state=0)
lin_reg = LinearRegression()
lin_reg.fit(x_train,y_train).score(x_train,y_train)
y_pred = lin_reg_c.predict(x_val)

print(metrics.mean_absolute_error(lin_reg.predict(x_train),y_train))
#metrics.r2_score(lin_reg.predict(x_train),y_train)
print(metrics.mean_absolute_error(y_val,y_pred))
#metrics.r2_score(y_val,y_pred)



