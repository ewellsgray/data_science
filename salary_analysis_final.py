
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
#print("Python")
#print("version: %s" % python_version())
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
df["title"].value_counts() # Prof:266, AsstProf:67, AssocProf: 64
df["years_service"].value_counts() # int 64 ranging from 3-60. No Nans

# Filter and count number of entries
df_filt1 = df.loc[(df.title=="AsstProf") & (df.years_service<5)]
print(len(df_filt1))
num_filtered = len(df_filt1) #63
percentage = (num_filtered / num_all_records) * 100
#%% Fewer female full profs?

df_fem_prof = df.loc[(df.title=="AsstProf") & (df.sex=="Female")]
print(len(df_fem_prof))
df_male_prof = df.loc[(df.title=="AsstProf") & (df.sex=="Male")]
print(len(df_male_prof))
#%% Is there a differnece in mean years_service beteen men and women, at AsstProf level?

prof_female_years = df.loc[(df.title=="AsstProf") & (df.sex=="Female")].years_service
prof_female_years.std()
#print(len(df_fem_prof))
prof_male_years = df.loc[(df.title=="AsstProf") & (df.sex=="Male")].years_service
prof_male_years.mean()
#print(len(df_male_prof))

# Two-sided independent-sample t-test, unequal variance (Welches t-test)
t_stat, p_val = stats.ttest_ind(prof_female_years, prof_male_years, equal_var=False)
print("t stat: "+str(round(t_stat,4)))
print("p val: "+str(round(p_val,4)))

#%% (2) Is there a statistically significant difference between female and male salaries?

# Check to make sure consistent entry for the salary and sex variables
df["salary"].value_counts()
df["sex"].value_counts() # 39 Female, 358 Male

sal_f = df.salary.loc[df.sex=="Female"]
sal_f.shape # (39,)
sal_f.mean()

sal_m = df.salary.loc[df.sex=="Male"]
sal_m.shape # (358,)
sal_m.mean()

# Two-sided independent-sample t-test, unequal variance (Welches t-test)
t_stat, p_val = stats.ttest_ind(sal_m, sal_f, equal_var=False)
print("t stat: "+str(round(t_stat,4)))
print("p val: "+str(round(p_val,4)))

#%% % Salary distribution for mean and women
plt.figure()
sn.distplot(df.loc[df["sex"]=="Male"].salary, color="blue", label="Male")
sn.distplot(df.loc[df["sex"]=="Female"].salary, color="red", label="Female")      
plt.legend()
plt.title("Salary Distributions by Sex")

mean_sex = df.groupby(["sex"]).mean().salary
med_sex = df.groupby(["sex"]).median().salary
std_sex = df.groupby(["sex"]).std().salary

#%% Question 3. Distribution of salary with discipline and Rank

mean_rank = df.groupby(["title"]).mean().salary
med_rank = df.groupby(["title"]).median().salary
std_rank = df.groupby(["title"]).std().salary

mean_disc = df.groupby(["discipline"]).mean().salary
med_disc = df.groupby(["discipline"]).median().salary
std_disc = df.groupby(["discipline"]).std().salary

mean_disc_rank = df.groupby(["discipline","title"]).mean().salary
med_disc_rank = df.groupby(["discipline","title"]).median().salary
std_disc_rank = df.groupby(["discipline","title"]).std().salary

#%% Figures for Question 3

plt.figure()
sn.distplot(df.loc[df["title"]=="AsstProf"].salary, color="blue", label="AsstProf")
sn.distplot(df.loc[df["title"]=="AssocProf"].salary, color="red", label="AssocProf")
sn.distplot(df.loc[df["title"]=="Prof"].salary, color="green", label="Prof")        
plt.legend()
plt.title("Salary Distributions by Rank")

plt.figure()
sn.distplot(df.loc[df["discipline"]=="A"].salary, color="blue", label="A:Theoretical")
sn.distplot(df.loc[df["discipline"]=="B"].salary, color="red", label="B:Applied")
plt.legend()
plt.title("Salary Distributions by Discipline")

g=sn.catplot(x="title", y="salary", kind="box", order=["Prof","AssocProf","AsstProf"],
           hue="discipline", data=df);
g.fig.set_size_inches(7,5)
sn.set_style("whitegrid", {'axes.grid' : False})
#sn.set_style('axes.grid'=True)
#ax.grid(True)
#%% Number os samples needed to test if women make less within first 2 years

from statsmodels.stats.power import TTestIndPower
ALPHA = .01
# Exploring a $4000 difference
es = 4000/df.loc[df.years_service<=2].std().salary
test=TTestIndPower()
# There are roughtly 5x more male sample points.
test.solve_power(effect_size=es, nobs1=None, 
                                 alpha=ALPHA, power=0.8, ratio=5.0, alternative='two-sided')

#%%
plt.figure()
sn.distplot(df.loc[df["title"]=="AsstProf"].years_service, color="blue", label="AsstProf")
sn.distplot(df.loc[df["title"]=="AssocProf"].years_service, color="red", label="AssocProf")
sn.distplot(df.loc[df["title"]=="Prof"].years_service, color="green", label="Prof")
plt.legend()
plt.title("Years Service Distributions by Discipline")
#%%
sn.scatterplot(df.years_service,df.salary,hue="title",data=df)
sn.scatterplot(df.years_phd,df.salary,hue="title",data=df)
sn.scatterplot(df.years_phd,df.salary,hue="sex",data=df)
#%% Difference in salary between all profs over 40 and 30-40 years service
df_over40 = df.loc[df.years_phd>40]
df_over40.salary.mean()
df_over40.salary.std()
len(df_over40.salary)

df_30_40 = df.loc[(df.years_phd>30)&(df.years_phd<=40)]
df_30_40.salary.mean()
df_30_40.salary.std()
len(df_30_40.salary)

ALPHA = .01
# effect size
es = 10000/df_over40.salary.std()
test=TTestIndPower()
# ~ 2x as many samples in th 30-40 group
test.solve_power(effect_size=es, nobs1=None, 
                                 alpha=ALPHA, power=0.8, ratio=2.0, alternative='two-sided')

# Two-sided independent-sample t-test, unequal variance (Welches t-test)
t_stat, p_val = stats.ttest_ind(df_over40.salary, df_30_40.salary, equal_var=False)
print("t stat: "+str(round(t_stat,4)))
print("p val: "+str(round(p_val,4))) # p=.1769 at current level

#%% Three-group salary comparison.Full Profs only. Groupled by years_phd
df_over40 = df.loc[((df.years_phd>40)&(df.title=="Prof"))]
df_over40.salary.mean()
df_over40.salary.std()
len(df_over40.salary)

df_25_40 = df.loc[((df.years_phd>25)&(df.years_phd<=40)&(df.title=="Prof"))]
df_25_40.salary.mean()
df_25_40.salary.std()

df_10_25 = df.loc[((df.years_phd>10)&(df.years_phd<=25)&(df.title=="Prof"))]
df_10_25.salary.mean()
df_10_25.salary.std()
len(df_10_25.salary)

# This may be off due to uneven sample sizes!!
stats.f_oneway(df_10_25.salary,df_25_40.salary,df_over40.salary)
#%% ANOVA power calculation
from statsmodels.stats.power import FTestAnovaPower
ftestAnova=FTestAnovaPower()
#ftestAnova.power(effect_size=es, nobs=180,alpha=ALPHA)
# This assumes equal sizes for all groups
ftestAnova.solve_power(effect_size=es, nobs=None,alpha=ALPHA, power=0.8)
#%% Years service histogram for full Profs
df_prof_phd = df.loc[df.title=="Prof"].years_phd
plt.hist(df_prof_phd,xlabel="years_phd")

#%% binarize catagorical data and ordinal encode the title

df.loc[df.discipline=="A","discipline"] = 0
df.loc[df.discipline=="B","discipline"] = 1

df.loc[df.sex=="Male","sex"] = 0
df.loc[df.sex=="Female","sex"] = 1

df.loc[df.title=="AssocProf","title"] = 1
df.loc[df.title=="AsstProf","title"] = 0
df.loc[df.title=="Prof","title"] = 2
