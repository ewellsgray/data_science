# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 21:25:35 2019

@author: ewell
"""
import os
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
data_path = []
for dirname, _, filenames in os.walk('.'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        data_path.append(os.path.join(dirname, filename))
        
df_o=pd.read_csv(data_path[1])

print(df_o.sample(10))
print(df_o.columns)
#%% 

# Data are mostly fluats, with a couple ints and a couple objects
print('\n Rows and Columns:')
print(df_o.shape)
print('\n Data Types:')
print(df_o.dtypes)

#%%
# =============================================================================
# 
# df = df_o.dropna(axis = 0, how ='any') 
# 
# print("Old data frame length:", len(df_o), "\nNew data frame length:",  
#        len(df), "\nNumber of rows with at least 1 NA value: ", 
#        (len(df_o)-len(df))) 
# =============================================================================

#%%
import seaborn as sn

corr = df.corr()
#cmap = sn.diverging_palette(255, 133, l=60, n=7, center="dark")
ax = sn.heatmap(corr,cmap='RdBu_r',vmin=-1,vmax=1)

ax.set_title("Correlation for all Features")

