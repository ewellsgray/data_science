
import pandas as pd
import scipy.stats as stats
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sn
#%% Check versions

print("System")
print("os name: %s" % os.name)
print()
print()
print("Python Packages")
#print("jupterlab==%s" % jupyterlab.__version__)
print("pandas==%s" % pd.__version__)
print("numpy==%s" % np.__version__)
print("matplotlib==%s" % matplotlib.__version__)
print("seaborn==%s" % sn.__version__)

#%%
path = os.getcwd()
df_o = pd.read_csv(os.path.join(path,"salaries.csv"))
#%%
# make a working copy
df = df_o.copy()
num_all_records = len(df) # 397
print(df.shape) # (397,6)

# Rename the columns/variables with dots, for convenience
# Renamed "rank" to "title" due to confile with "rank" used a df method
df.rename(columns={"yrs.since.phd":"years_phd","yrs.service":"years_service","rank":"title"}, inplace=True)
print(df.columns)

#%% --------------------------------
# -- PREPARE/EPLORE THE DATA
#------------------------------------
#%% binarize catagorical data and ordinal encode the title

df.loc[df.discipline=="A","discipline"] = 0
df.loc[df.discipline=="B","discipline"] = 1

df.loc[df.sex=="Male","sex"] = 0
df.loc[df.sex=="Female","sex"] = 1

df.loc[df.title=="AssocProf","title"] = 1
df.loc[df.title=="AsstProf","title"] = 0
df.loc[df.title=="Prof","title"] = 2

#%% Various Distribution Plots    
#sn.scatterplot("years_phd","salary", hue="sex",data=df)
#plt.show
#sn.scatterplot("years_phd","salary", hue="discipline",data=df)
#sn.scatterplot("years_phd","salary", hue="title",data=df)
sn.scatterplot("years_phd","salary", hue="title",data=df)
#df.salary.hist()
#%% Scatter matrix of existing features
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

#%% --------------------------------------------------------------
#  ---- FEATURE ENGINEERING
# -------------------------------------------------------------

#  1. Catagory for years_phd >45 and  25<years_phd<=45
df["over_40"] = (df.years_phd>40)*1

# 2. Years difference between these highly correlated features
df["year_diff"] = df.years_phd - df.years_service

#3. "year_max" is the max of either of these two features, which is a bit more
#    correlated with Salary than they are (they will be dropped to simplify feature space)
df["year_max"] = df[["years_phd","years_service"]].max(axis=1)
#df["year_min"] = df[["years_phd","years_service"]].min(axis=1)
#df["year_mean"] = df[["years_phd","years_service"]].mean(axis=1)
df.drop(["years_phd","years_service"], inplace=True, axis=1)

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
#%% Let's take a nother look with the new feautres
df = move_target_to_end(df,"salary")
scatter_matrix(df, figsize=(12,9))
plot_corr_matrix(df)

#%% Plot Bar-chart of correlation with salary
plt.figure(figsize=(10,3.5))
#%matplotlib inline
df.corr().iloc[-1].sort_values(ascending=False).plot(kind="bar")
#plt.show()
plt.xlabel('feature')
plt.ylabel('Pearson Correaltion')
plt.title('Feature Correlation (with Target)')

#%% --------------------------------------------------------------
# --- DATASET PREP 
# ----------------------------------------------------------------
#%%
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit

#x, x_test, y, y_test = train_test_split(x,y,test_size=0.2, random_state=0)

#%% Prepare dummy catagory "salary cat" to enforce stratified sampling
bins=np.array([50,80,100,120,160,250])*1000
df["salary_cat"] = pd.cut(df["salary"], bins=bins,labels=[1,2,3,4,5])
df.salary_cat.hist()

# split Traing/Test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=15)
for train_index, test_index in split.split(df,df["salary_cat"]):
    strat_train_set = df.loc[train_index]
    strat_test_set = df.loc[test_index]

#%%  create x and y variable for modeling

#First drop the dummy catagory
strat_train_set.drop(["salary_cat"],axis=1, inplace=True)
strat_test_set.drop(["salary_cat"],axis=1, inplace=True)
df.drop(["salary_cat"],axis=1, inplace=True) 

# take the y variable as the last column
x = strat_train_set.iloc[:,0:-1]
y = strat_train_set.iloc[:,-1]

# make a clean backup of pre-pipeline variables
x_o=x.copy()
y_o=y.copy()

# All operation repeated on test set 
x_test = strat_test_set.iloc[:,0:-1]
y_test = strat_test_set.iloc[:,-1]

# And repeated on full set "_all"
x_all = df.iloc[:,0:-1]
y_all = df.iloc[:,-1]

# Binarize the target, For classification model
y_binary = (y<y.median(axis=0))*1
y_binary_Test = (y_test<y_test.median(axis=0))*1
y_binary_all = (y_all<y_all.median(axis=0))*1

#%% Model --Linear Regression, no scaling

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import ElasticNet
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
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import QuantileTransformer
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import ColumnTransformer

#%% Generate the "x_prepared" sets ------------------------
#     Only uses standard scaling (i.e., no poly tranforming)

stdsc =StandardScaler()
# Transform the data and re-frame the results as pandas DataFrame
x_prepared = pd.DataFrame(stdsc.fit_transform(x),index=x.index,columns=x.columns)
# Repeat for test set
x_prepared_test = pd.DataFrame(stdsc.fit_transform(x_test),
                               index=x_test.index,columns=x_test.columns)
# Repeat "all" set
x_prepared_all = pd.DataFrame(stdsc.fit_transform(x_all),
                              index=x_all.index,columns=x_all.columns)
#%% Generate "x_poly" sets -----------------------
#    Pipeline uses Poly + Standard Scaling

pipeline = Pipeline([("poly_features",PolynomialFeatures(degree=2, include_bias=True)),('std_scaler',StandardScaler())])
pipeline.fit(x_prepared)
# Retrieve the column names, for book-keeping
poly_cols = pipeline.named_steps["poly_features"].get_feature_names(x_prepared.columns)
# Transform the data and re-frame the results as pandas DataFrame
x_poly=pd.DataFrame(pipeline.fit_transform(x_prepared),
                    index=x_prepared.index,columns=poly_cols)
# Repeat for test set
x_poly_test=pd.DataFrame(pipeline.fit_transform(x_prepared_test),
                         index=x_prepared_test.index,columns=poly_cols)
# repeat for "all" set
x_poly_all=pd.DataFrame(pipeline.fit_transform(x_prepared_all),
                        index=x_prepared_all.index,columns=poly_cols)

#%% Take a look at the transformations before proceeding -----------------

# Need to recombine x and y for easy correlation plotting
df_temp = x_poly.copy()
df_temp["salary"] = y
plt.figure(figsize=(10,3.5))
# Absoloute Magnitude--
np.abs(df_temp.corr()).iloc[-1].sort_values(ascending=False).plot(kind="bar")
# or maintain negative
#df_temp.corr().iloc[-1].sort_values(ascending=False).plot(kind="bar")
#plt.show()
plt.xlabel('feature')
plt.ylabel('Pearson Correaltion')
plt.title('Feature Correlation (with Target)')

#%% Remove low correlated features -----------------------------------
#   Many of the new attributes have very low correlation with salary. 
#   Remove them if below the cutoff thresh

cutoff=.1
boo = np.abs(df_temp.corr().iloc[-1])<cutoff
drop_cols=df_temp.columns[boo].to_list()
# Dummy DataFrame
df_new =df_temp.drop(drop_cols,inplace=False, axis=1)

#%% Take a look again after dropping those poly features ------------------

plt.figure(figsize=(10,3.5))
np.abs(df_new.corr()).iloc[-1].sort_values(ascending=False).plot(kind="bar")
#plt.show()
plt.xlabel('feature')
plt.ylabel('Pearson Correaltion')
plt.title('Feature Correlation (with Target)')

# removing these because they are somewhat redundand; I think the "1" is an 
#  index artifact of recombining dataframes
df_new.drop(["1","sex^2","discipline^2"],inplace=True, axis=1)

#%% drop_cols now has all the columns to be dropped

for i in ["1","sex^2","discipline^2"]:
    drop_cols.append(i)
    
# Do the drop for our 3 x sets
x_poly.drop(drop_cols,axis=1,inplace=True)
x_poly_test.drop(drop_cols,axis=1,inplace=True)
x_poly_all.drop(drop_cols,axis=1,inplace=True)

#%% ------------------------------------------------------
#---  GRID SEARCH -- 
# --------------------------------------------------------

def two_param_grid_search(x_data, y_data, model, p_names, p_vectors):
    
    """ Grid search over two parametrs for any model. returns the best_model.
    Also prints some key results. """

    # Set Parameter Names and search values
    param_grid = [{p_names[0]:p_vectors[0],p_names[1]:p_vectors[1]}]
    #elast_net4 =ElasticNet()
    grid_search = GridSearchCV(model, param_grid, cv=6,scoring="neg_mean_squared_error",
                               return_train_score=True, iid="False")
    grid_search.fit(x_data,y_data)
    
    elast_net_GS_scores = np.sqrt(-grid_search.best_score_)
    #cv_results = grid_search.cv_results_
    print("\n")
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
    
    return grid_search.best_estimator_
#%% --------------------------------------
#---  GRID SEARCH -- Elastic Net
#-------------------------------------------
p_names = ["alpha","l1_ratio"]
p_vectors = [list(np.linspace(.05,.5,10)),list(np.linspace(.01,.99,10))]
best_model=two_param_grid_search(x_data=x_poly, y_data=y,model=ElasticNet(max_iter=1000000),
                      p_names=p_names, p_vectors=p_vectors)

#%%  Elastic Net Grid Search Model results, Training Set
y_pred = best_model.predict(x_poly)
np.sqrt(mean_squared_error(y, y_pred))

#%%  Elastic Net Grid Search Model results, Test set 
y_pred_test = best_model.predict(x_poly_test)
np.sqrt(mean_squared_error(y_test, y_pred_test))
#%% Re-train on all data -- keeping hyperparamets from above

# These are the values from the grid search using the poly set
ALPHA, L1_r = .05,0.6333
elast_net_final =ElasticNet(alpha=ALPHA, l1_ratio=L1_r)
elast_net_final.fit(x_poly_all, y_all)

y_pred_all = elast_net_final.predict(x_poly_all)
print("RMS Error, all data: "+str(np.sqrt(mean_squared_error(y_all, y_pred_all))))
elast_net_final.coef_
elast_net_final.intercept_

coefs = elast_net_final.coef_
inter = elast_net_final.intercept_

plt.figure(figsize=(12,6))
plt.barh(x_poly.columns,coefs,height=.8)
plt.show()

#%% ----------------------------------------------------
# Binary Classification
# -------------------------------------------------------

# Baseline: logistic regresstion
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

#%% Re-defining the grid search function 

def two_param_grid_search_clf(x_data, y_data, model, p_names, p_vectors):
    # Set Parameter Names and search values
    param_grid = [{p_names[0]:p_vectors[0],p_names[1]:p_vectors[1]}]
    #elast_net4 =ElasticNet()
    grid_search = GridSearchCV(model, param_grid, cv=3,scoring="neg_mean_squared_error",
                               return_train_score=True, iid="False")
    grid_search.fit(x_data,y_data)
    
    #elast_net_GS_scores = np.sqrt(-grid_search.best_score_)
    #cv_results = grid_search.cv_results_
    print("\n")
    print(grid_search.best_params_)
    print(round(grid_search.best_score_))
    
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
    #plt.plot(np.array(grid_mean_test_score)[ind[0]])
    #plt.plot(np.array(grid_mean_train_score)[ind[0]])
    #plt.show()
    #plt.plot(np.array(grid_mean_test_score).T[ind[1]])
    #plt.show()
    #plt.plot(np.array(grid_mean_train_score).T[ind[1]])
    
    return grid_search.best_estimator_

#%%
#y_pred = log_clf.fit(x_poly_all, y_binary_all).predict(x_poly_all)
#print("Training Accuracy: "+str(round(log_clf.score(x_poly_all, y_binary_all),2)))
#plot_confusion_matrix(y_binary_all, y_pred, classes=["below median","above median"])


#%% Grid_search on Training Set ------------------------
    
p_names = ["C","l1_ratio"]
p_vectors = [list(np.linspace(.01,.3,5)),list(np.linspace(.01,.99,5))]
best_model=two_param_grid_search_clf(x_data=x_poly, y_data=y_binary, model=LogisticRegression(penalty="elasticnet",
                      solver="saga",max_iter=1000000),
                      p_names=p_names, p_vectors=p_vectors)
        
print("Training Accuracy: "+str(round(best_model.score(x_poly, y_binary),2)))
y_pred = best_model.predict(x_poly)
plot_confusion_matrix(y_binary, y_pred, classes=["below median","above median"])

#%% Results on the Testing Set --------------------------

print("Testing Accuracy: "+str(round(best_model.score(x_poly_test, y_binary_Test),2)))
y_pred = best_model.predict(x_poly_test)
plot_confusion_matrix(y_binary_Test, y_pred, classes=["below median","above median"])

#%%  Final Model ----------------------------------------

model_final_clf=LogisticRegression(penalty="elasticnet",C=2275,l1_ratio=.5,
                      solver="saga",max_iter=1000000)
model_final_clf.fit(x_poly_all, y_binary_all)

print("Testing Accuracy: "+str(round(model_final_clf.score(x_poly_all, y_binary_all),2)))
y_pred = best_model.predict(x_poly_all)
plot_confusion_matrix(y_binary_all, y_pred, classes=["below median","above median"])




