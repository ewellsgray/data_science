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
import seaborn as sn

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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
#%% binarize catagorical data and ordinal encode the title

df.loc[df.discipline=="A","discipline"] = 0
df.loc[df.discipline=="B","discipline"] = 1

df.loc[df.sex=="Male","sex"] = 0
df.loc[df.sex=="Female","sex"] = 1

df.loc[df.title=="AssocProf","title"] = 1
df.loc[df.title=="AsstProf","title"] = 0
df.loc[df.title=="Prof","title"] = 2
#%%
def encode_column(encoding_type, col_name):
    
    if encoding_type=='label':
        le=LabelEncoder()
        le.fit(df[col_name])
        title_order = list(le.classes_)
        df[col_name] = le.fit_transform(df[col_name])
        print("Label Encoded")
    if encoding_type=='ordinal':
        oe=OrdinalEncoder()
        Ord = [["AssocProf",1],["AsstProf",0],["Prof",2]]
        oe.fit(Ord)
        title_order = Ord
        oe.transform(df[col_name])
        print("Ordinal Encoded")

    return
#%%
#encoding_type = 'label'
#encode_column("ordinal","title")

#%%
    
#sn.scatterplot("years_phd","salary", hue="sex",data=df)
#plt.show
#sn.scatterplot("years_phd","salary", hue="discipline",data=df)
#sn.scatterplot("years_phd","salary", hue="title",data=df)
sn.scatterplot("years_phd","salary", hue="title",data=df)
#%%
sn.scatterplot("title","salary", hue="sex",data=df)

#%%
# -- EXPLORE THE DATA ------------------------------------
#df.salary.hist()
from pandas.plotting import scatter_matrix
scatter_matrix(df, figsize=(12,9))
#%%
def plot_corr_matrix(df, subset=[], details="All Features"):
    """Plots a correlation matrix of all the features. Can specify subset of features if desired"""

    if len(subset)>0:
        df = df[subset]
        
    corr = df.corr()
    #print("First 3 correlations: "+str(corr.iloc[0,0:3]))
    #cmap = sn.diverging_palette(255, 133, l=60, n=7, center="dark")
    
    # The values number here ar just for "nice" scaling
    xx=(df.shape[1]/3,(df.shape[1]/3)*.85 )
    fig=plt.figure(figsize=xx)
    ax = sn.heatmap(corr,cmap='RdBu_r',vmin=-1,vmax=1)

    ax.set_title("Correlation for "+details)
#%%
plot_corr_matrix(df)
print("Corr Coef between years_phd and years_service: "+str(round(df.corr().iloc[2,3],2)))
print("Corr Coef between title and TARGET: "+str(round(df.corr().iloc[0,5],2)))
print("Corr Coef between discipline and TARGET: "+str(round(df.corr().iloc[1,5],2)))
print("Corr Coef between years_phd and TARGET: "+str(round(df.corr().iloc[2,5],2)))
print("Corr Coef between years_service and TARGET: "+str(round(df.corr().iloc[3,5],2)))
print("Corr Coef between sex and TARGET: "+str(round(df.corr().iloc[4,5],2)))
#%%
df.groupby(["sex"]).mean()
df.groupby(["title"]).mean()

#%%
#  ---- FEATURE ENGINEERIGN

#%%
#  1. Catagory for years_phd >45 and  25<years_phd<=45
df["over_40"] = (df.years_phd>40)*1

# 2. 
df["year_diff"] = df.years_phd - df.years_service
#%%
def move_target_to_end(df,target_label):
    
    """Moves the target colums to the end of the DataFrame for easier book-keeping
    and correlation matrix display"""
    
    cols = df.columns.tolist()
    
    p = cols.index(target_label)
    cols_new=cols[0:p]
    cols_new = cols_new + cols[p+1:]
    cols_new.append(cols[p])
    
    df=df[cols_new]
    return df

#%%
df = move_target_to_end(df,"salary")
scatter_matrix(df, figsize=(12,9))
plot_corr_matrix(df)

#%%
# --- CREATE TEST SET -----------------------------------------

#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

#x, x_test, y, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#%%
bins=np.array([50,80,100,120,160,250])*1000
df["salary_cat"] = pd.cut(df["salary"], bins=bins,labels=[1,2,3,4,5])
df.salary_cat.hist()

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=15)
for train_index, test_index in split.split(df,df["salary_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

"""    
strat_test_set["salary_cat"].value_counts() / len(strat_test_set)

def cat_proportions(data):
    return data["salary_cat"].value_counts() / len(data)

# ------------------------------------------------------------------------
train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
# ------------------------------------------------------------------------
compare_props = pd.DataFrame({
    "Overall": cat_proportions(df),
    "Stratified": cat_proportions(strat_test_set),
    "Random": cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
compare_props"""
#%%
"""def cat_proportions_sex(data):
    return data["sex"].value_counts() / len(data)

compare_props_sex = pd.DataFrame({
    "Overall": cat_proportions_sex(df),
    "Stratified": cat_proportions_sex(strat_test_set),
    "Random": cat_proportions_sex(test_set),
}).sort_index()
compare_props_sex["Rand. %error"] = 100 * compare_props_sex["Random"] / compare_props_sex["Overall"] - 100
compare_props_sex["Strat. %error"] = 100 * compare_props_sex["Stratified"] / compare_props_sex["Overall"] - 100
compare_props_sex"""
#%% split x and y

strat_train_set.drop(["salary_cat"],axis=1, inplace=True)
strat_test_set.drop(["salary_cat"],axis=1, inplace=True)
df.drop(["salary_cat"],axis=1, inplace=True)

x = strat_train_set.iloc[:,0:-1]
y = strat_train_set.iloc[:,-1]
x_o=x.copy()
y_o=y.copy()

#%% Model --Linear Regression, no scaling

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import power_transform
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.decomposition import PCA
log_class = LogisticRegression()


#%%
scalar = StandardScaler()
xc = scalar.fit_transform(x)










#%%

def linear_reg_simple(X,Y,stsc=False, do_pca=False, boxcox=False):
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.2, random_state=42)
    
    # Make the PipeLine list
    S=[]
    S.append(("lin_reg",LinearRegression()))
    print("Linear Regression")
    #bc = PowerTransformer(method='box-cox')
    if do_pca==True:
        S.insert(0,("pca",PCA(n_components=4)))
        print("Using PCA")
    if stsc==True:
        S.insert(0,("stsc",StandardScaler()))
        #model =make_pipeline(StandardScaler(),LinearRegression())
        print("Using StandardScaler")
        #print(X.iloc[5,:])


    model = Pipeline(S)
    model.fit(X_train,Y_train).score(X_train,Y_train)
    Y_pred = model.predict(X_val)
    
    #print("Training MAE: {:.2}" .format(metrics.mean_absolute_error(model.predict(X_train),Y_train)))
    #print("Validation MAE: {:.2}" .format(metrics.mean_absolute_error(Y_val,Y_pred)))
    print(metrics.mean_absolute_error(model.predict(X_train),Y_train))
    print(metrics.mean_absolute_error(Y_val,Y_pred))
    print("Training R2: {:.2}" .format(metrics.r2_score(model.predict(X_train),Y_train)))
    print("Validation R2: {:.2}" .format(metrics.r2_score(Y_val,Y_pred)))
    print("\n")
#%%
def dec_tree_simple(X,Y,stsc=False, boxcox=False):
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.2, random_state=42)
    
    if stsc==True:
        model =make_pipeline(StandardScaler(),DecisionTreeRegressor())
        print("Using StandardScaler")
        #print(X.iloc[5,:])
    else:
        model =make_pipeline(DecisionTreeRegressor(max_depth=5,random_state=42))
        print("Not Using StandardScaler")
        #print(X.iloc[5,:])
    #model = LinearRegression()
    model.fit(X_train,Y_train).score(X_train,Y_train)
    Y_pred = model.predict(X_val)
    
    print("Training MAE: {:.2}" .format(metrics.mean_absolute_error(model.predict(X_train),Y_train)))
    print("Validation MAE: {:.2}" .format(metrics.mean_absolute_error(Y_val,Y_pred)))
    #print(metrics.mean_absolute_error(model.predict(X_train),Y_train))
    #print(metrics.mean_absolute_error(Y_val,Y_pred))
    print("Training R2: {:.2}" .format(metrics.r2_score(model.predict(X_train),Y_train)))
    print("Validation R2: {:.2}" .format(metrics.r2_score(Y_val,Y_pred)))
    print("\n")
#%%
def svm_simple(X,Y,stsc=False):
    X_train, X_val, Y_train, Y_val = train_test_split(X,Y,test_size=0.2, random_state=42)
    
    S=[]
    S.append(("lin_SVR",LinearSVR()))
    if stsc==True:
        S.insert(0,("stsc",StandardScaler()))
        #model =make_pipeline(StandardScaler(),NuSVR())
        print("Using StandardScaler")
        #print(X.iloc[5,:])
    else:
        model =make_pipeline(NuSVR())
        print("Not Using StandardScaler")
        #print(X.iloc[5,:])
    #model = LinearRegression()
    model.fit(X_train,Y_train).score(X_train,Y_train)
    Y_pred = model.predict(X_val)
    
    print("Training MAE: {:.2}" .format(metrics.mean_absolute_error(model.predict(X_train),Y_train)))
    print("Validation MAE: {:.2}" .format(metrics.mean_absolute_error(Y_val,Y_pred)))
    #print(metrics.mean_absolute_error(model.predict(X_train),Y_train))
    #print(metrics.mean_absolute_error(Y_val,Y_pred))
    print("Training R2: {:.2}" .format(metrics.r2_score(model.predict(X_train),Y_train)))
    print("Validation R2: {:.2}" .format(metrics.r2_score(Y_val,Y_pred)))
    print("\n")
    
#%%
linear_reg_simple(x,y,stsc=False)
linear_reg_simple(x,y,do_pca=True, stsc=True)
#dec_tree_simple(x,y,stsc=False)
#svm_simple(x,y,stsc=True)

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



