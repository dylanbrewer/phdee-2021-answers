# Homework 4 code -- Dylan Brewer

# Clear all
from IPython import get_ipython
get_ipython().magic('reset -sf')

# Import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import statsmodels.api as sm

# Set working directories and seed
datapath = r'C:\Users\dbrewer30\Dropbox\teaching\Courses\BrewerPhDEnv\Homeworks\phdee-2021-homework\homework4'
outputpath = r'C:\Users\dbrewer30\Dropbox\teaching\Courses\BrewerPhDEnv\Homeworks\phdee-2021-answers\homework4\output'

os.chdir(datapath)

random.seed(20121159)
np.random.seed(411128)

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
trends = datalong.groupby(['treatgroup','month']).mean()
controltrends = trends.loc[0, :]
controltrends = controltrends.reset_index()
treattrends = trends.loc[1, :]
treattrends = treattrends.reset_index()

## Build the plot
plt.plot(controltrends['month'], controltrends['bycatch'], marker = 'o')
plt.plot(treattrends['month'], treattrends['bycatch'], marker = 'o')
plt.axvline(x=13, color = 'red', linestyle = 'dashed') # Vertical line to indicate treatment year
plt.xlabel('Month')
plt.ylabel('Mean bycatch per firm (lbs)')
plt.legend(['Control', 'Treatment','Treatment date'])
os.chdir(outputpath) # Change directory
plt.savefig('hw4_q1.eps',format='eps')
plt.show()

# Problem 2 ------------------------------------------------------------------

DID = (trends.loc[(1,13),'bycatch'] - trends.loc[(1,12),'bycatch']) - (trends.loc[(0,13),'bycatch'] - trends.loc[(0,12),'bycatch'])

# Problem 3 ------------------------------------------------------------------
twoperiod = datalong[(datalong['month'] == 12) | (datalong['month'] == 13)]
pre = pd.get_dummies(twoperiod['month'],prefix = 'pre', drop_first = True)
twoperiod = pd.concat([twoperiod,pre],axis = 1)

yvar3 = twoperiod['bycatch']
xvar3 = twoperiod[['treatgroup','treated','pre_13']]

DID3 = sm.OLS(yvar3,sm.add_constant(xvar3, prepend = False)).fit()
DID3robust = DID3.get_robustcov_results(cov_type = 'cluster', groups = twoperiod['firm']) # Cluster-robust confidence intervals

## Output results
beta3 = np.round(DID3robust.params,2) # Estimates
params3, = np.shape(beta3) # Parameter size

nobs3 = np.array(DID3robust.nobs) # Nobs

ci3 = pd.DataFrame(np.round(DID3robust.conf_int(),2)) # CIs
ci3_s = '(' + ci3.loc[:,0].map(str) + ', ' + ci3.loc[:,1].map(str) + ')' # Formats CIs

output3 = pd.DataFrame(pd.concat([pd.Series(np.append(beta3,nobs3)),ci3_s],axis = 1).stack()) # Note also appends nobs3
output3.columns = ['(3)']
output3.index = pd.concat([pd.Series(['Treatment group','Treated','Pre-period','Constant','Observations']),pd.Series([' ']*params3)], axis = 1).stack()

output3.to_latex('hw4_output3.tex',column_format = 'lc')

