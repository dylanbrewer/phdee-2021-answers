# Homework 9 data simulation -- Dylan Brewer

# Clear all

from IPython import get_ipython
get_ipython().magic('reset -sf')

# Import packages

import os
import numpy as np
import pandas as pd

# Set working directories and seed

outputpath = r'C:\Users\dbrewer30\Dropbox\teaching\Courses\BrewerPhDEnv\Homeworks\phdee-2021-homework\homework9'

os.chdir(outputpath)

np.random.seed(8133)

# Generate some random data --------------------------------------------------
nobs = 50 # set number of firms
tobs = 36 # set number of months

#firmsize = np.round(1 + 30*np.random.power(0.7,(nobs,1)),0) # size of firms power law version
#firmsize = np.round(1+30*np.random.random((nobs,1)),0) # size of firms uniform version
firmsize = np.round(np.random.normal(10,5,(nobs,1)),0)

for i in range(nobs):
    if firmsize[i] < 1: # ensure there are no zero-sized firms
        firmsize[i] = 1

## Draw treatments
treatdraw = np.random.random((nobs,1))
treated = treatdraw > 0.5
treated = treated.astype(int)
threatened = treatdraw > 0.75
threatened = threatened.astype(int)
threatened2 = treatdraw < 0.25
threatened2 = threatened2.astype(int) + threatened

## Random walks for shrimp and salmon prices
pshrimp = np.zeros([1,tobs])
psalmon = np.zeros([1,tobs])
delta = [0.65,0.81]
start = [12.50,15.80]
prob1 = 0.45
prob2 = 0.75

draws = np.random.random([1,tobs+1])

for i in range(tobs):
    if i == 0:
        pshrimp[0,i] = start[0]
        psalmon[0,i] = start[1]
    elif draws[0,i] <= 0.45 :
        pshrimp[0,i] = pshrimp[0,i-1] + delta[0]
        psalmon[0,i] = psalmon[0,i-1] + delta[0]
    elif draws[0,i] <= 0.75:
        pshrimp[0,i] = pshrimp[0,i-1] - delta[0]
        psalmon[0,i] = psalmon[0,i-1] + delta[1]
    else:
        pshrimp[0,i] = pshrimp[0,i-1] - delta[0]
        psalmon[0,i] = psalmon[0,i-1] - delta[1]

## Draw firm supply characteristics: supply = (A - treatloss + B*price_t + trend + e_it)*firmsize*fraction
supshrimpa = np.random.normal(10000,2000,[nobs,1])
supshrimpb = np.random.normal(80,40,[nobs,1])
shrimptrend = np.reshape(np.arange(tobs)*20,(1,tobs))
shrimpit = np.random.normal(0,300,[nobs,tobs])

supsala = np.random.normal(50000,1000,[nobs,1])
supsalb = np.random.normal(40,20,[nobs,1])
saltrend = np.reshape(np.arange(tobs)*20,(1,tobs))
salit = np.random.normal(0,100,[nobs,tobs])

treatloss = np.random.uniform(800,200,[nobs,1])

## Calculate fish supplies for each firm, for each month
shrimp = np.zeros([nobs,tobs])
salmon = np.zeros([nobs,tobs])

### Harvests
for i in range(nobs):
    for t in range(12):
        shrimp[i,t] = (supshrimpa[i,0] + supshrimpb[i,0]*pshrimp[0,t] + shrimptrend[0,t] + shrimpit[i,t])*firmsize[i,0]*0.7
        salmon[i,t] = (supsala[i,0] + supsalb[i,0]*psalmon[0,t] + saltrend[0,t] + salit[i,t])*firmsize[i,0]*0.3
    for t in range(12,24):
        shrimp[i,t] = (supshrimpa[i,0] - threatened[i,0]*treatloss[i,0] + supshrimpb[i,0]*pshrimp[0,t] + shrimptrend[0,t] + shrimpit[i,t])*firmsize[i,0]*0.7
        salmon[i,t] = (supsala[i,0] + supsalb[i,0]*psalmon[0,t] + saltrend[0,t] + salit[i,t])*firmsize[i,0]*0.3
    for t in range(24,36):
        shrimp[i,t] = (supshrimpa[i,0] - treatloss[i,0] + supshrimpb[i,0]*pshrimp[0,t] + shrimptrend[0,t] + shrimpit[i,t])*firmsize[i,0]*0.7
        salmon[i,t] = (supsala[i,0] + supsalb[i,0]*psalmon[0,t] + saltrend[0,t] + salit[i,t])*firmsize[i,0]*0.3

## Draw firm bycatch characteristics bycatch = (alpha-d1)*shrimp + (beta - d2) salmon - d3 + etai + epsilon it
alpha = 1
beta = 0.5
d1 = 0.2
d2 = 0.1
etai = np.random.normal(100,50,[nobs,1])
d3 = 0
epsilonit = np.random.normal(0,10,[nobs,tobs])

## Calculate bycatch
bycatch = np.zeros([nobs,tobs])

for i in range(nobs):
    for t in range(12):
        bycatch[i,t] = alpha*shrimp[i,t] + beta*salmon[i,t] + etai[i,0] + epsilonit[i,t]
    for t in range(12,24):
        bycatch[i,t] = (alpha-(d1*threatened[i,0]))*shrimp[i,t] + (beta-(d2*threatened[i,0]))*salmon[i,t] - d3*threatened[i,0] + etai[i,0] + epsilonit[i,t]
    for t in range(24,36):
        bycatch[i,t] = (alpha-d1)*shrimp[i,t] + (beta-d2)*salmon[i,t] - d3 + etai[i,0] + epsilonit[i,t]

## Firm identification number
firm = np.arange(1,nobs+1).reshape((nobs,1))

# Convert to Pandas dataframe and export as csv

## Shrimp
colshrimp = []
for t in range(tobs):
    colshrimp.append('shrimp' + str(t + 1))

shrimp = pd.DataFrame(shrimp,columns = colshrimp)

## Salmon
colsalmon = []
for t in range(tobs):
    colsalmon.append('salmon' + str(t + 1))
    
salmon = pd.DataFrame(salmon,columns = colsalmon)
    
## Bycatch
colbycatch = []
for t in range(tobs):
    colbycatch.append('bycatch' + str(t + 1))

bycatch = pd.DataFrame(bycatch,columns = colbycatch)

## Other variables
labels = ['firm','firmsize','treated']

fishbycatch = pd.DataFrame(np.concatenate((firm, firmsize, treated),axis = 1), columns = labels)

## Put them all together
fishbycatch = pd.concat([fishbycatch,shrimp,salmon,bycatch],axis = 1)
fishbycatch.to_csv('fishbycatchupdated.csv', index = None, header=True)