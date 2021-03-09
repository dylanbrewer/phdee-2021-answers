# Homework 6 code -- Dylan Brewer

# Clear all
from IPython import get_ipython
get_ipython().magic('reset -sf')

# Import packages
import os
import numpy as np
import pandas as pd
import random
import statsmodels.api as sm

# Set working directories and seed
datapath = r'C:\Users\dbrewer30\Dropbox\teaching\Courses\BrewerPhDEnv\Homeworks\phdee-2021-homework\homework4'
outputpath = r'C:\Users\dbrewer30\Dropbox\teaching\Courses\BrewerPhDEnv\Homeworks\phdee-2021-answers\homework6\output'

os.chdir(datapath)

random.seed(5401)
np.random.seed(83147)

# Import homework data
datawide = pd.read_csv('fishbycatch.csv')

# Prepare data and pivot to long
datalong = pd.wide_to_long(datawide,stubnames = ['salmon','shrimp','bycatch'], i = 'firm', j = 'month')
datalong = datalong.reset_index(level=['firm', 'month'])

## Treatment group and treated variables
datalong['treatgroup'] = datalong['treated'] # Static variable for which firms are in the treatment group
datalong['treated2'] = np.where((datalong['treated'] == 1) & (datalong['month']>12) & (datalong['month']<25), 1, 0)
datalong['treated3'] = np.where((datalong['month']>24), 1, 0)
datalong['treated'] = datalong['treated2'] + datalong['treated3'] # Dynamic variable for when firms receive treatment

datalong = datalong.drop(columns = ['treated2', 'treated3']) # drop extra variables

# Part (a) -------------------------------------------------------------------
yvar1 = datalong['bycatch']

tvars1 = pd.get_dummies(datalong['month'],prefix = 'time',drop_first = True) # creates dummies from time variables
firmfe1 = pd.get_dummies(datalong['firm'],prefix = 'f',drop_first = True) # creates firm dummies

xvar1 = pd.concat([datalong[['treated','shrimp','salmon']],tvars1,firmfe1],axis = 1)

FE1 = sm.OLS(yvar1,sm.add_constant(xvar1,prepend = False)).fit()
FE1robust = FE1.get_robustcov_results(cov_type = 'cluster', groups = datalong['firm'])

print(FE1robust.summary())

# Part (b) -------------------------------------------------------------------
## Generate demeaned variables
yvar2 = yvar1.copy() # Initialize

yvar2 -= datalong.groupby('firm')['bycatch'].transform('mean')

xvar2_b = pd.concat([datalong[['firm','treated','shrimp','salmon']],tvars1],axis = 1)
xvar2 = xvar2_b.drop('firm',axis = 1).copy() # initialize
for var in xvar2.columns.values:
    xvar2[var] -= xvar2_b.groupby('firm')[var].transform('mean') # Found a faster way to do this than above

## Run FE regression
FE2 = sm.OLS(yvar2,xvar2).fit()
FE2robust = FE2.get_robustcov_results(cov_type = 'cluster', groups = datalong['firm'])
print(FE2robust.summary())

# Part (c) -------------------------------------------------------------------
## Estimates and parameters:
beta1 = np.round(FE1robust.params,2)
beta2 = np.round(FE2robust.params,2)

params1, = np.shape(beta1)
params2, = np.shape(beta2)

ci1 = pd.DataFrame(np.round(FE1robust.conf_int(),2))
ci2 = pd.DataFrame(np.round(FE2robust.conf_int(),2))
ci1_s = '(' + ci1.loc[:,0].map(str) + ', ' + ci1.loc[:,1].map(str) + ')'
ci2_s = '(' + ci2.loc[:,0].map(str) + ', ' + ci2.loc[:,1].map(str) + ')'

nobs1 = int(FE1robust.nobs)
nobs2 = int(FE2robust.nobs)

## Index variable locations
keepidx1 = []
keepidx2 = []
for i in ['treated','shrimp','salmon']:
    keepidx1.append(xvar1.columns.get_loc(i))
    keepidx2.append(xvar1.columns.get_loc(i))
    
## Build the table
outputhw5_1 = pd.DataFrame(pd.concat([pd.Series(beta1)[keepidx1],ci1_s[keepidx1]],axis = 1).stack())
outputhw5_2 = pd.DataFrame(pd.concat([pd.Series(beta2)[keepidx2],ci2_s[keepidx2]],axis = 1).stack()) 

outputhw5 = pd.concat([outputhw5_1,outputhw5_2], axis = 1)
outputhw5.columns = ['(a)','(b)']
outputhw5.index = pd.concat([pd.Series(['Treated', 'Shrimp', 'Salmon']),pd.Series([' ']*len(keepidx1))], axis = 1).stack()
outputhw5 = outputhw5.append(pd.DataFrame([[str(nobs1),str(nobs2)], ['Y', 'Y'], ['Y','Y']], index = ['Observations', 'Month indicators', 'Fixed effects'], columns = ['(a)', '(b)']))

os.chdir(outputpath)
outputhw5.to_latex('outputhw6.tex',column_format = 'lcc', na_rep = ' ')