# -*- coding: utf-8 -*-
"""
Created on Fri Nov  1 10:05:43 2019

@author: ewell
"""
#https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#cookbook-selection

import pandas as pd
import numpy as np

#%% Creating a DataFrame

# Creating df from dict;  each key creates a column heading,
#  with the values supplying each entry
D = {"AAA":[1,2,3],"BBB":[4,5,6],"CCC":[7,8,9]}

df = pd.DataFrame(D)
c=df[(df.AAA>=2) & (df.index.isin([0,2]))]

df2 = pd.DataFrame(D,index=["i1","i2","i3"])
print(df2)
#%% Indexing
df.set_index('ID',inplace=True)
# Can also assign directly on csv import
df_feat = pd.read_csv(data_path[0], index_col=[0,1,2])

#%% Return index for given values (filtering)

df = pd.DataFrame({'BoolCol1': [True, False, False, True, True],
                    'BoolCol2': [False,False,False,False,True ]},
       index=[10,20,30,40,50])

#%%
df = pd.DataFrame({'BoolCol': [True, False, False, True, True]},
       index=[10,20,30,40,50])
# the following appear to be equivalent?
#df.index[df['BoolCol']].tolist()
df[df['BoolCol']].index.tolist()
#%% Return index for given values (filtering)

df = pd.DataFrame({'Col1': [1,2,3,4,5]},
       index=[10,20,30,40,50])
#%%
# the following appear NOT to be equivalent
df.index[df<=3].tolist()
df[df<=3].index.tolist()

#%% Slicing

# There are 2 explicit slicing methods, with a third general case

# 1. Positional-oriented (Python slicing style : exclusive of end)
# 2. Label-oriented (Non-Python slicing style : inclusive of end)
# 3. General (Either slicing style : depends on if the slice contains labels or positions)
# CAUTION: amiguity arises when the specified index begins with 1 (or other non-zero value
)

# Positional - EXCLUSIVE of end
#print(df2.iloc[0:2])
#print(df2.iloc[:2])

# Label Oriented - INCLUSIVE of end
print(df2.loc["i1":"i2"])
print(df2.index.isin(["i1","i2"]))
#%% Descriptive
df["education"].value_counts()


#%% Filtering and Selecting
reviews['country'][0]
reviews.iloc[0]
df.loc[df["type"]==1]
df.loc[df.type==1]
reviews.iloc[:, 0]
reviews.loc[(reviews.country == 'Italy') & (reviews.points >= 90)]
reviews.loc[(reviews.country == 'Italy') | (reviews.points >= 90)]
reviews.loc[reviews.country.isin(['Italy', 'France'])]
reviews.loc[reviews.price.notnull()]
#%% -- New Columns 

source_cols = df.columns   # Or some subset would work too
new_cols = [str(x) + "_cat" for x in source_cols]
categories = {1: 'Alpha', 2: 'Beta', 3: 'Charlie'}
df[new_cols] = df[source_cols].applymap(categories.get)

# Rename Columns
df.rename(str.lower, axis='columns')
df.rename(columns={"A": "a", "B": "c"})
#%% If-Then asignments, from pandas cookbook

# two columns asignments (df.AAA>=5 is the "if)
df.loc[df.AAA >= 5, ['BBB', 'CCC']] = 555
# can add another line to the the else (df.AAA<5) 
df.loc[df.AAA < 5, ['BBB', 'CCC']] = 2000

# can uses where with a boolean mask. 
df.where(df_mask, -1000)

# Can use np where also
df['logic'] = np.where(df['AAA'] > 5, 'high', 'low')
#%% -- Misc Useful Operations

# Shuffle/Randomize your DF
df = df.sample(frac=1).reset_index(drop=True)

# An alternative to one-hot enoding?
pd.get_dummies(dataframe)

#%% -- Prepping a data set for machine learning

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# To keep things simple, we'll use only numerical predictors
X = X_full.select_dtypes(exclude=['object'])
X_test = X_test_full.select_dtypes(exclude=['object'])

# Break off validation set from training data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8, test_size=0.2,
                                                      random_state=0)

#%% C
df = df_o.dropna(axis = 0, how ='any') 

print("Old data frame length:", len(df_o), "\nNew data frame length:",  
       len(df), "\nNumber of rows with at least 1 NA value: ", 
       (len(df_o)-len(df))) 

# Set the ID column as the index
df.set_index('ID',inplace=True)


#%%


import pandas as pd
def quick_analysis(df):
    print(“/nData Types:”)
    print(df.dtypes)
    print(“/nRows and Columns:”)
    print(df.shape)
    print(“/nColumn Names:”)
    print(df.columns)
    print(“/nNull Values:”)
    print(df.apply(lambda x: sum(x.isnull()) / len(df)))
quick_analysis(train)



#%%
df = pd.DataFrame(np.random.randn(3, 8), index=['A', 'B', 'C'], columns=index)






#%%




