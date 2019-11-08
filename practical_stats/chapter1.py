# Chapter 1 - Exploratory Data Analysis

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#%% Simulate populations for testing
# The aim here have a large enough sample point that it fully represents the 
# population distribution and statistics

def plot_dist(dist, color,new=True):
    if new == True:
        plt.figure(figsize=(3,3))
        
    plt.hist(dist,100,color=color)
    plt.xlabel("value")
    plt.ylabel("count")    
    return 0

# Population 1: normal distribution with mean = 10 and standard deviation = 2
u1, sig1 = 10, 2
n1 = 1000000
norm1 = np.random.normal(u1,sig1,n1)
plot_dist(norm1, "b")

# Population 2: normal distribution with mean = 10 and standard deviation = 4
u2, sig2 = 10, 4
n2 = 1000000
norm2 = np.random.normal(u2,sig2,n2)
plot_dist(norm2,"r", new=False)

# Population 3: a Gamma dist with shape(k) = 2, scale(t) = 1
k3, t3 = 2, 3
n3 = 1000000
gamma3 = np.random.gamma(k3,t3,n3)
plot_dist(gamma3,"g", new=True)

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

