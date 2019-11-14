# Chapter 1 - Exploratory Data Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

#%% Simulate populations for testing
# The aim here have a large enough sample point that it fully represents the 
# population distribution and statistics
  
# Population 1: normal distribution with mean = 10 and standard deviation = 2
u1, sig1 = 10, 2
n1 = 1000000
norm1 = np.random.normal(u1,sig1,n1)

# Population 2: normal distribution with mean = 10 and standard deviation = 4
u2, sig2 = 10, 4
n2 = 1000000
norm2 = np.random.normal(u2,sig2,n2)

# Population 3: a Gamma dist with shape(k) = 2, scale(t) = 1
k3, t3 = 2, 3
n3 = 1000000
gamma3 = np.random.gamma(k3,t3,n3)


fig,axes = plt.subplots(2,3,figsize=(10,5),sharex=True,squeeze=True)
axes[0,0].hist(norm1,20,color='blue', edgecolor='black')
axes[0,1].hist(norm2,20,color='red', edgecolor='black')
axes[0,2].hist(gamma3,20,color='green', edgecolor='black')

plt.sca(axes[1,0])
sn.distplot(norm1,bins=20)
plt.sca(axes[1,1])
sn.distplot(norm2,bins=20)
plt.sca(axes[1,2])
sn.distplot(gamma3,bins=20)
#plt.hist(dist,40,color=color, edgecolor='black')

# Annotate the top row
axes[0,0].set(xlabel="value", ylabel="count",title="Distribution 1")
axes[0,1].set(xlabel="value", title="Distribution 2")
axes[0,2].set(xlabel="value", title="Distribution 3")

# Annotate the bottom row
axes[1,0].set(xlabel="value", ylabel="prob",title="Probability Density")
axes[1,1].set(xlabel="value", title="Probability Density")
axes[1,2].set(xlabel="value", title="Probability Density")



#%%
n=20
sample1 = np.random.choice(norm1, size=n)
sample2 = np.random.choice(norm2, size=n)
sample3 = np.random.choice(gamma3, size=n)

#%% Estimates of Location

# Mean (population)
mean1 = norm1.mean()
mean2 = norm2.mean()
mean3 = gamma3.mean()
# Mean (population)
mean1s = sample1.mean()
mean2s = sample2.mean()
mean3s = sample3.mean()

# Weighted Mean

# Median

# Weighted median

# Anomoly Detection

#%% Estimates of Variability

# Variance

# Standard Deviation

# Mean Absolute Ddeviation

# Median Absolute Deviation from the Median

# Range

# Percentiles

# Interquaritle Range

#%% Exploring the Data Distribution

