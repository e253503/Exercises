#%%
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
%matplotlib inline
import matplotlib.pyplot as plt
import sklearn.mixture as mix
from sklearn.model_selection import train_test_split
#%%
data = pd.read_csv("EMDATA.csv")
data

#%% AIC & BIC ; Covarience = Full

n_components = np.arange(1, 101)
models1 = [mix.GaussianMixture(n, covariance_type='full',max_iter = 10000,
                              random_state= 0).fit(data)
          for n in n_components]
models1

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(n_components, [m.aic(data) for m in models1], label='AIC Full')
ax.plot(n_components, [m.bic(data) for m in models1], label='BIC Full')

ax.axvline(np.argmin([m.aic(data) for m in models1]), color='blue')
ax.axvline(np.argmin([m.bic(data) for m in models1]), color='green')

plt.legend(loc='best') 
plt.xlabel('n_components')
#%% AIC & BIC ; Covarience = Tied

n_components = np.arange(1, 101)
models2 = [mix.GaussianMixture(n, covariance_type='tied',max_iter = 10000,
                              random_state= 0).fit(data)
          for n in n_components]
models2

fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(n_components, [m.aic(data) for m in models2], label='AIC Tied')
ax.plot(n_components, [m.bic(data) for m in models2], label='BIC Tied')

ax.axvline(np.argmin([m.aic(data) for m in models2]), color='blue')
ax.axvline(np.argmin([m.bic(data) for m in models2]), color='green')

plt.legend(loc='best') 
plt.xlabel('n_components')
#%%#%% AIC & BIC ; Covarience = Diag
n_components = np.arange(1, 101)
models3 = [mix.GaussianMixture(n, covariance_type='diag',max_iter = 10000,
                              random_state= 0).fit(data)
          for n in n_components]
models3

fig, ax = plt.subplots(figsize=(9, 7))
ax.plot(n_components, [m.aic(data) for m in models3], label='AIC Diag')
ax.plot(n_components, [m.bic(data) for m in models3], label='BIC Diag')

ax.axvline(np.argmin([m.aic(data) for m in models3]), color='blue')
ax.axvline(np.argmin([m.bic(data) for m in models3]), color='yellow')

plt.legend(loc='best') 
plt.xlabel('n_components')

#%%#%% AIC & BIC ; Covarience = Spherical
n_components = np.arange(1, 101)
models4 = [mix.GaussianMixture(n, covariance_type='spherical',max_iter = 10000,
                              random_state= 0).fit(data)
          for n in n_components]
models4
fig, ax = plt.subplots(figsize=(9, 7))

ax.plot(n_components, [m.aic(data) for m in models4], label='AIC Spherical')
ax.plot(n_components, [m.bic(data) for m in models4], label='BIC Spherical')

ax.axvline(np.argmin([m.aic(data) for m in models4]), color='blue')
ax.axvline(np.argmin([m.bic(data) for m in models4]), color='black')

plt.legend(loc='best') 
plt.xlabel('n_components')

# From graphics, AIC & BIC Tied have chosen as they have lower values.


#%% Finding n for best models from Tied AIC and Tied BIC

md2a = {}
for i in range(0,100):
    md2a[f'{i}']= models2[i].aic(data)
    print(f'Component Number: {models2[i].n_components}',models2[i].aic(data))
md2a
mds2 = {k: v for k, v in sorted(md2a.items(), key=lambda item: item[1])}
mds2aR = list(mds2.keys())[0], list(mds2.values())[0]

md2b = {}
for i in range(0,100):
    md2b[f'{i}']= models2[i].bic(data)
    print(f'Component Number: {models2[i].n_components}',models2[i].bic(data))
md2b
mds2b = {k: v for k, v in sorted(md2b.items(), key=lambda item: item[1])}
mds2bR = list(mds2b.keys())[0], list(mds2b.values())[0]
print('Aic Tied:',mds2aR,'Bic Tied.:',mds2bR) 
### What I have done is  basically:
# create empty dic
# take all aic values into this dictionary.
# according to their component number, list them
# sort them according to their aic value and assign to new dictionary
# take first key and value of this dictionary and assign it to new list
# Repeat this cycle for BIC
# models2 used for covarience == tied

#%% Finding Means & Covariances
meanL = {}
covarianceL = {}
for i,j in zip(models2,range(0,100)):
    meanL[j] = models2[j].means_
    covarianceL[j] = models2[j].covariances_
# AIC Tied n = 71 is best value 
print( 'mean:', models2[71].means_,'\ncovarience: ', models2[71].covariances_)