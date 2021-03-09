# Homework 7 data simulation -- Dylan Brewer
# Based loosely off of Stata's auto.dta dataset

# Clear all

from IPython import get_ipython
get_ipython().magic('reset -sf')

# Import packages

import os
import numpy as np
import pandas as pd

# Set working directories and seed

datapath = r'C:\Users\dbrewer30\Dropbox\teaching\Courses\BrewerPhDEnv\Homeworks\phdee-2021-homework\homework7'

os.chdir(datapath)

np.random.seed(7327)

# Observations
nobs1 = 500
nobs2 = 500

nobs = nobs1 + nobs2


# Car and SUV classes
vehicles = np.concatenate([np.ones((nobs1,1)),np.zeros((nobs2,1))],axis = 0)


# Length, weight, and height instruments
carlength = np.random.normal(188,40,(nobs1,1))
carweight = np.random.normal(2400,500,(nobs1,1))
carheight = np.random.normal(48,2,(nobs1,1))

suvlength = np.random.normal(210,40,(nobs2,1))
suvweight = np.random.normal(2800,500,(nobs2,1))
suvheight = np.random.normal(66,5,(nobs2,1))

length = np.concatenate([carlength,suvlength],axis = 0)
weight = np.concatenate([carweight,suvweight],axis = 0)
height = np.concatenate([carheight,suvheight],axis = 0)

# MPG Error
carerror = np.random.normal(0,5,(nobs1,1))
suverror = np.random.normal(0,5,(nobs1,1))

# MPG
carmpg = 40 - 0.003 * carweight - 0.02 * carheight + carerror
suvmpg = 35 - 0.003 * suvweight - 0.02 * suvheight + suverror

mpg = np.concatenate([carmpg,suvmpg],axis = 0)

features = np.random.normal(0,10,(nobs,1)) - 1 * (mpg + 0.003*weight + 0.02*height)

## RDD
for i in range(nobs):
    if length[i] < 225:
        mpg[i] = mpg[i] - length[i]*.01 - length[i]*.002 + 9
    else:
        mpg[i] = mpg[i] - length[i]*.01 - length[i]*.002

# Unobserved features
#features = np.zeros((nobs,1))
#for i in range(nobs):
#    features[i] = np.random.normal(-0.25*mpg[i],25)


# Price error
priceerror = np.random.normal(0,1000,(nobs,1))

# Price
price = 28000 -2978 * vehicles + 150 * mpg + 300 * features + priceerror

# Export
instrumentalvehicles = pd.DataFrame(np.concatenate([price,vehicles,mpg,weight,height,length],axis = 1),columns = ['price','car','mpg','weight','height','length'])
instrumentalvehicles.to_csv('instrumentalvehicles.csv', index = None, header=True)