#!/usr/bin/env python
# coding: utf-8

# In[410]:


from astropy.io import fits
import numpy as np
import astropy
from astropy.table import Table
import matplotlib.pyplot as plt


# In[411]:


path = '/home/suman/tifr/'
file = path + '202307011_1506_Am241_on_det1_30000pkts.fits'

hdul = fits.open(file)[1]


# In[412]:


data = hdul.data
tab = Table(data)
tab


# In[413]:


data['pixid']


# In[414]:


#mask = 0 < np.any(data['pixid']) < 3


# In[415]:


#plt.plot(data["pixid"][mask],data["pha"][mask])


# In[416]:


detid = data[data['detid']== 1]
data_ = detid[detid['pixid'] == 5]

np.shape(data_)
data_


# In[417]:


tab = Table(data)
#tab


# In[418]:


tab_ = Table(data_)
tab_


# In[419]:


len(np.unique(data_['time']))


# In[420]:


import matplotlib.pyplot as plt


# In[421]:


len(data_['pha'])


# In[422]:


np.sqrt(6164)


# In[514]:


plt.plot((data_['time']),(data_['pha']),".", label = 'Det 1, pixel = 5')
plt.xlabel("Aqusiton time($\mu$s)")
plt.legend()
plt.ylabel("PHA")
plt.xscale("log")
plt.yscale("log")
#plt.xlim(0, 100000000)


# In[516]:


N,bins,_ = plt.hist(data_['pha'], bins=300,label = 'Det 1, pixel = 5')
plt.xlabel("PHA")
plt.ylabel("Counts")
plt.legend()
plt.show()
len(bins)


# In[485]:


len(N)


# In[486]:


bin_centers = (bins[:-1] + bins[1:]) / 2


# In[487]:


def gauss(x,amp,mean,stdev):
    return amp*np.exp(-(x-mean)**2/(2*stdev**2))


# In[488]:


gauss(1,10,9.3,0.3)


# In[489]:


p0 = [np.max(N),np.mean(data_['pha']),np.std(data_['pha'])]


# In[490]:


p0


# In[491]:


from scipy.optimize import curve_fit


# In[492]:


params, pcov = curve_fit(gauss,bin_centers,N,p0)
errors = np.sqrt(np.diag(pcov))
params


# In[493]:


fit_curve = gauss(bin_centers,*params)


# In[518]:


plt.plot(bin_centers, fit_curve, 'r-', label='Gaussian Fit over full data')
N,bins,_ = plt.hist(data_['pha'], bins=len(bins)-1, label="Det 1,pixel =5")
plt.xlabel("PHA")
plt.ylabel("Counts")
plt.legend()
plt.show()
print("No. of bins, Mean, Deviation")
for p,e in zip(params, errors):
    print(f"{p:0.1f} +- {e:0.1f}")
print("Approximate resolution : {:0.1f}%".format(100* 2.35 * params[2] / params[1]))


# with a certain range fitting
# 

# In[495]:


min_val_PHA  = p0[1]-2*p0[2]
max_val_PHA  = p0[1]+3*p0[2]
min_val_PHA,max_val_PHA


# In[496]:


fit_range = np.where((min_val_PHA <=bins[:-1]) & (max_val_PHA >= bins[1:]))[0]
fit_range


# In[497]:


fit_range_final = N[fit_range]
#fit_range_final


# In[498]:


fit_bin_center = (bins[:-1] + bins[1:]) / 2
#fit_bin_center


# In[499]:


bin_center_fit_range = fit_bin_center[fit_range]
#bin_center_fit_range


# In[500]:


params1, pcov1 = curve_fit(gauss,bin_center_fit_range,fit_range_final,p0)
errors1 = np.sqrt(np.diag(pcov))
params1


# In[501]:


fit_curve1 = gauss(bin_center_fit_range, *params1)


# In[519]:


plt.bar(bin_center_fit_range,fit_range_final, alpha=0.5, label='Det 1,pixel =5  ')
plt.plot(bin_center_fit_range, fit_curve1, 'r-', label='Gaussian Fit Sliced over -2$\sigma$ to 3$\sigma$')
plt.xlabel("PHA")
plt.ylabel("Counts")
plt.legend()
plt.show()
print("No. of bins, Mean, Deviation")
for p,e in zip(params1, errors1):
    print(f"{p:0.1f} +- {e:0.1f}")
print("Approximate resolution : {:0.1f}%".format(100* 2.35 * params1[2] / params1[1]))


# In[ ]:





# In[ ]:





# In[ ]:




