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
y_binary = (y<y.median(axis=0))*1

#%% Model --Linear Regression, no scaling

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import power_transform
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import LinearSVR
from sklearn.svm import NuSVR
from sklearn.svm import SVR
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
log_class = LogisticRegression()
#%%
lin_reg =LinearRegression()
scores1 = cross_val_score(lin_reg,x,y,scoring="neg_mean_squared_error",cv=6)
lin_scores = np.sqrt(-scores1)

def display_score(scores):
    print("Scores:", scores.round(2))
    print("Mean:", scores.mean().round(2))
    print("StDev:", scores.std().round(2))

display_score(lin_scores)

#%%
pipeline = Pipeline([('std_scaler',StandardScaler())])
x_tr = pipeline.fit_transform(x)
#%%
full_pipeline = ColumnTransformer([
        ("num", pipeline, list(x.columns))])
x_prepared = full_pipeline.fit_transform(x)
#%% -- Linear Regression

#housing_prepared = full_pipeline.fit_transform(housing)

#lin_reg =LinearRegression()
lin_stsc =LinearRegression()
scores2 = cross_val_score(lin_stsc,x_prepared,y,scoring="neg_mean_squared_error",cv=6)
lin_stsc_scores = np.sqrt(-scores2)

print("\nLinear Regression:")
def display_score(scores):
    print("Scores:", scores.round(2))
    print("Mean:", scores.mean().round(2))
    print("StDev:", scores.std().round(2))

display_score(lin_stsc_scores)

#%% -- Elastic Net
from sklearn.linear_model import ElasticNet
ALPHA, L1_r = 0.05,0.4
elast_net =ElasticNet(alpha=ALPHA, l1_ratio=L1_r)
scores = cross_val_score(elast_net,x_prepared,y,scoring="neg_mean_squared_error",cv=6)
elast_net_scores = np.sqrt(-scores)

print("\nElastic Net: alpha="+str(ALPHA)+", l1_ratio="+str(L1_r))
def display_score(scores):
    print("Scores:", scores.round(2))
    print("Mean:", scores.mean().round(2))
    print("StDev:", scores.std().round(2))

display_score(elast_net_scores)

#%%
MAX_DEPTH=5
tree_reg =DecisionTreeRegressor(max_depth=MAX_DEPTH)
scores3 = cross_val_score(tree_reg,x_prepared,y,scoring="neg_mean_squared_error",cv=6)
tree_reg_scores = np.sqrt(-scores3)

print("\nDecision Tree: Max Depth=", MAX_DEPTH)
def display_score(scores):
    print("Scores:", scores.round(2))
    print("Mean:", scores.mean().round(2))
    print("StDev:", scores.std().round(2))

display_score(tree_reg_scores)

#%%
svreg = SVR(degree=3, gamma='auto', kernel='poly')
scores4 = cross_val_score(svreg,x_prepared,y,scoring="neg_mean_squared_error",cv=6)
svreg_scores = np.sqrt(-scores4)

print("\nSVM")
def display_score(scores):
    print("Scores:", scores.round(2))
    print("Mean:", scores.mean().round(2))
    print("StDev:", scores.std().round(2))

display_score(svreg_scores)
#%%%
from sklearn.ensemble import RandomForestRegressor
N_EST, MAX_DEPTH_r=10,10
rand_for =RandomForestRegressor(n_estimators=N_EST, max_depth=MAX_DEPTH_r,random_state=0)
scores = cross_val_score(rand_for,x_prepared,y,scoring="neg_mean_squared_error",cv=6)
rand_for_scores = np.sqrt(-scores )

print("\nRandom Forest: n_est="+str(N_EST)+", max_depth="+str(MAX_DEPTH_r))
def display_score(scores):
    print("Scores:", scores.round(2))
    print("Mean:", scores.mean().round(2))
    print("StDev:", scores.std().round(2))

display_score(rand_for_scores)

#%%
from sklearn.preprocessing import PolynomialFeatures

# Add polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=True)
x_poly=poly_features.fit_transform(x_prepared)

ALPHA, L1_r = .1,0.6
elast_net2 =ElasticNet(alpha=ALPHA, l1_ratio=L1_r)
scores = cross_val_score(elast_net2,x_poly,y,scoring="neg_mean_squared_error",cv=6)
elast_net_scores2 = np.sqrt(-scores)

print("\n2nd Deg Poly + Elastic Net: alpha="+str(ALPHA)+", l1_ratio="+str(L1_r))
def display_score(scores):
    print("Scores:", scores.round(2))
    print("Mean:", scores.mean().round(2))
    print("StDev:", scores.std().round(2))

display_score(elast_net_scores2)
#%%

# Add polynomial features
poly_features = PolynomialFeatures(degree=2, include_bias=True)
x_poly=poly_features.fit_transform(x_prepared)
#pca=PCA()
#pca.fit(x_poly)
#cumsum = np.cumsum(pca.explained_variance_ratio_)
#plt.plot(cumsum)

#pca=PCA(n_components=.94)
#pca.fit_transform(x_poly)
ALPHA, L1_r = .5,0.5
elast_net2 =ElasticNet(alpha=ALPHA, l1_ratio=L1_r)
pca_elastnet = Pipeline([("pca",PCA(n_components=.95)),("elast_net3",ElasticNet(alpha=ALPHA, l1_ratio=L1_r))])

scores = cross_val_score(pca_elastnet,x_poly,y,scoring="neg_mean_squared_error",cv=6)
pca_elastnet_scores = np.sqrt(-scores)

print("\nElastic Net: alpha="+str(ALPHA)+", l1_ratio="+str(L1_r))
def display_score(scores):
    print("Scores:", scores.round(2))
    print("Mean:", scores.mean().round(2))
    print("StDev:", scores.std().round(2))

display_score(pca_elastnet_scores)
#%% --------------------------------------
#---  GRID SEARCH -- Elastic Net
# ----------------------------------------

def two_param_grid_search(model, x_data):
    # Set Parameter Names and search values
    p_names = ["alpha","l1_ratio"]
    p_vectors = [list(np.linspace(.05,.5,10)),list(np.linspace(.01,.99,10))]
    #alpha_list = list(np.linspace(.05,.5,10))
    #l1_ratio_list = list(np.linspace(.01,.99,10))
    
    param_grid = [{p_names[0]:p_vectors[0],p_names[1]:p_vectors[1]}]
    #elast_net4 =ElasticNet()
    grid_search = GridSearchCV(model, param_grid, cv=6,scoring="neg_mean_squared_error",
                               return_train_score=True, iid="False")
    grid_search.fit(x_data,y)
    
    elast_net_GS_scores = np.sqrt(-grid_search.best_score_)
    #cv_results = grid_search.cv_results_
    print(grid_search.best_params_)
    print(round(elast_net_GS_scores))
    
    results = grid_search.cv_results_
    # Lengths of grid_search atribute vecotrs
    v = list(param_grid[0])
    l0 = len(param_grid[0][v[0]])
    l1 = len(param_grid[0][v[1]])
    # all training results
    grid_mean_test_score = np.sqrt(-results["mean_test_score"]).round(0).reshape((l0,l1))
    # all testing results
    grid_mean_train_score = np.sqrt(-results["mean_train_score"]).round(0).reshape((l0,l1))
 
    ind = []
    for i in range(len(p_names)):
        ind.append(p_vectors[i].index(grid_search.best_params_[p_names[i]]))
        #print(ind)
    plt.plot(np.array(grid_mean_test_score)[ind[0]])
    #plt.plot(np.array(grid_mean_train_score)[ind[0]])
    plt.show()
    plt.plot(np.array(grid_mean_test_score).T[ind[1]])
    plt.show()
    #plt.plot(np.array(grid_mean_train_score).T[ind[1]])
    
    return
#%%
#two_param_grid_search(model=ElasticNet(max_iter=10000), x_data=x_prepared)
two_param_grid_search(model=ElasticNet(max_iter=100000), x_data=x_poly)


#%% Logistic Regression Classifier

log_clf = LogisticRegression(solver="lbfgs",random_state=0)
scores = cross_val_score(log_clf,x_prepared,y_binary,cv=6)
#predictions = cross_val_score(log_clf,x_prepared,y_binary,cv=6)

print("\nLog Regression Classiifer:")
def display_score(scores):
    print("Scores:", scores.round(2))
    print("Mean:", scores.mean().round(2))
    print("StDev:", scores.std().round(2))

display_score(scores)

#%%
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_predict

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
y_pred = log_clf.fit(x_prepared, y_binary).predict(x_prepared)
print("Training Accuracy: "+str(round(log_clf.score(x_prepared, y_binary),2)))
plot_confusion_matrix(y_binary, y_pred, classes=["below median","above median"])




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
        model =make_pipeline(display_score)
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



