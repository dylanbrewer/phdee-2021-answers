# Homework 1 data generation code

# Clear all

from IPython import get_ipython
get_ipython().magic('reset -sf')

# Import packages

import os
import numpy as np
import pandas as pd

# Set working directories

outputpath = r'C:\Users\dbrewer30\Dropbox\teaching\Courses\BrewerPhDEnv\Homeworks\phdee-2021-homework\homework2'

# Set number of observations and seed

nobs = 1000
sqftnobs1 = int(nobs/2)
sqftnobs2 = int(nobs/2)
np.random.seed(3280129176)
    
# Draw variables
sqft1 = np.round_(np.random.normal(2300,300,sqftnobs1),0)
sqft2 = np.round_(np.random.normal(1000,100,sqftnobs1),0)
sqft = np.transpose(np.append(sqft1, sqft2))

temp = np.transpose(np.round_(np.random.normal(80,2,nobs),2))

mf1 = np.zeros((sqftnobs1,1))
mf2 = np.ones((sqftnobs2,1))
multifamily = np.transpose(np.append( np.zeros((sqftnobs1,1)), np.ones((sqftnobs2,1)) ))

error = np.transpose(np.random.normal(1,0.1,nobs))

retrofit = np.transpose(np.rint(np.random.uniform(size = nobs)))


# Simulate outcome variable around 900 kwh per month

alpha = 2.5
delta = 0.9
gamma1 = 0.1
gamma2 = 0.78
endog = 0.9

electricity = np.zeros((nobs,1))

for i in range(nobs) : 
    electricity[i] = alpha*delta**(retrofit[i])*temp[i]**(gamma1)*sqft[i]**(gamma2)*endog**(multifamily[i])*error[i]

# Export csv

kwh = pd.DataFrame({'electricity': electricity[:,0], 'sqft': sqft[:], 'retrofit': retrofit[:], 'temp': temp[:]})
kwh = kwh.sort_values('electricity')

os.chdir(outputpath)
kwh.to_csv('kwh.csv', index = None, header=True)
