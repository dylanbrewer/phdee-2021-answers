# Homework 5 code -- Dylan Brewer

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
outputpath = r'C:\Users\dbrewer30\Dropbox\teaching\Courses\BrewerPhDEnv\Homeworks\phdee-2021-answers\homework5\output'

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

# Problem 1 ------------------------------------------------------------------
yvar1 = datalong['bycatch']
tvars1 = pd.get_dummies(datalong['month'],prefix = 'time',drop_first = True) # creates dummies from time variables
xvar1 = pd.concat([datalong[['treatgroup','treated']],tvars1],axis = 1)

DID1 = sm.OLS(yvar1,sm.add_constant(xvar1,prepend = False)).fit()
DID1robust = DID1.get_robustcov_results(cov_type = 'cluster', groups = datalong['firm'])

## Output results
beta1 = np.round(DID1robust.params,2)
params1, = np.shape(beta1)

nobs1 = int(DID1robust.nobs)

ci1 = pd.DataFrame(np.round(DID1robust.conf_int(),2))
ci1_s = '(' + ci1.loc[:,0].map(str) + ', ' + ci1.loc[:,1].map(str) + ')'

### Build index of variables to keep
keepidx = []
for i in ['treatgroup','treated']: # Loops to get position of the variables to keep
    keepidx.append(xvar1.columns.get_loc(i))
keepidx.append(params1-1) # Adds constant term

### Build table
index1 = ['Treatment group','Treated','Constant']
output1 = pd.DataFrame(pd.concat([pd.Series(beta1)[keepidx],ci1_s[keepidx]],axis = 1).stack())
output1.index = pd.concat([pd.Series(index1),pd.Series(index1)+'_ci'], axis = 1).stack()
output1 = output1.append(pd.DataFrame([str(nobs1),'Y'],index = ['Observations', 'Month indicators'])) # appends nobs and month indicators rows
output1.columns = ['(1)']

# Problem 2 ------------------------------------------------------------------
yvar2 = datalong['bycatch']
tvars2 = pd.get_dummies(datalong['month'],prefix = 'time',drop_first = True) # creates dummies from time variables
xvar2 = pd.concat([datalong[['treatgroup','treated','shrimp','salmon','firmsize']],tvars2],axis = 1)

DID2 = sm.OLS(yvar2,sm.add_constant(xvar2,prepend = False)).fit()
DID2robust = DID2.get_robustcov_results(cov_type = 'cluster', groups = datalong['firm'])

datalong.to_csv('test4.csv', index = None, header=True)

## Output
beta2 = np.round(DID2robust.params,2)
params2, = np.shape(beta2)

nobs2 = int(DID2robust.nobs)

ci2 = pd.DataFrame(np.round(DID2robust.conf_int(),2))
ci2_s = '(' + ci2.loc[:,0].map(str) + ', ' + ci2.loc[:,1].map(str) + ')'

### Build index of variables to keep
keepidx2 = []
for i in ['treatgroup','treated','shrimp','salmon','firmsize']:
    keepidx2.append(xvar2.columns.get_loc(i))
keepidx2.append(params2-1)

### Build table
index2 = ['Treatment group','Treated','Shrimp','Salmon','Firm size','Constant']
output2 = pd.DataFrame(pd.concat([pd.Series(beta2)[keepidx2],ci2_s[keepidx2]],axis = 1).stack())
output2.index = pd.concat([pd.Series(index2),pd.Series(index2)+'_ci'], axis = 1).stack()
output2 = output2.append(pd.DataFrame([str(nobs2),'Y'],index = ['Observations', 'Month indicators']))
output2.columns = ['(2)']

### Combine tables 1 and 2 (Perhaps better ways to do this: see sm.summary_col)
index3 = ['Treatment group','Treated','Shrimp','Salmon','Firm size','Constant','Observations','Month indicators']
hw4_output = output1.join(output2,how = 'outer',sort = True)
hw4_output = hw4_output.reindex(pd.concat([pd.Series(index3), pd.Series(index3[0:6])+'_ci'],axis = 1).stack())
hw4_output.index = pd.concat([pd.Series(index3), pd.Series([' ']*len(index2))], axis = 1).stack()

os.chdir(outputpath)

hw4_output.to_latex('hw5_output.tex',column_format = 'lcc', na_rep = ' ')


