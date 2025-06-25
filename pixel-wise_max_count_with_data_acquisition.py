#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astropy.io import fits
import numpy as np
import astropy
from astropy.table import Table
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import stats 
from scipy.optimize import curve_fit


# In[2]:


from matplotlib.backends.backend_pdf import PdfPages
            


# In[412]:


import astropy.io.fits as fits
file_path = '/home/suman/Desktop/30K_DATA/det0_good_data_30k/fits_file0/20230706_1006_ba133_30000pkts_ext_hv_600_20kev_spectra.fits'
hdulist = fits.open(file_path)


# In[413]:


data = hdulist[1].data
print(Table(data))


# In[414]:


# Filter data for detid == 1
detid = data[data['detid'] == int(input("Detector Id (0/1) = "))]


# In[441]:


source = input("source name (Am/Ba1/Ba2/Eu1/Eu2) = ")
if source == "Am":
    pha_min = 750  # Minimum PHA value
    pha_max = 1100 # Maximum PHA value
    Energy_value = 59.6
elif source == "Ba1":
    pha_min = 300 
    pha_max = 900 
    Energy_value = 30.85
elif source == "Ba2":
    pha_min = 900  
    pha_max = 1600 
    Energy_value = 81
elif source == "Eu2":
    pha_min = 1700  
    pha_max = 2000
    Energy_value = 105.31
else: 
    pha_min =  1000
    pha_max =1600
    Energy_value = 85.55
print(pha_min,pha_max,Energy_value)


# In[442]:


def gauss(x,amp,mean,stdev):
    return amp*np.exp(-(x-mean)**2/(2*stdev**2))


# In[450]:


# Number of bins for histograms
n = len(detid)
k = 1 + np.log2(n)
bin_count = int(4096 / k)
max_count_pha_list = []
max_count_pha_list_OFF = []
# Loop through pixid values from 0 to 255
for pixid in range(256):
    # Filter data for the current pixid
    data_ = detid[detid['pixid'] == pixid]
    data_ = data_['pha']
    filtered_data = data_[(data_ != 4095) & (data_ >= pha_min) & (data_ <= pha_max)]

    if len(filtered_data) > (pha_max - pha_min):  # Ensure there is data to process
        # Compute histogram
        N, bins = np.histogram(filtered_data, bins=bin_count)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        p0 = [np.max(N), np.mean(filtered_data), np.std(filtered_data)]

        try:
            # Fit the Gaussian function to the histogram data
            params, pcov = curve_fit(gauss, bin_centers, N, p0, maxfev=2000)
            # Append the mean value (mu) from the fit to the list
            max_count_pha_list.append((pixid, params[1], Energy_value))
        except RuntimeError:
            # If the fit fails, append None
            max_count_pha_list_OFF.append((pixid, None, Energy_value))
        
    else:
        # Append None if there is no data for the current pixid
        max_count_pha_list_OFF.append((pixid, None))

print("List of maximum count PHA values for each pixid:")
print(max_count_pha_list)


# In[451]:


# Extract the mean values (params[1]) from max_count_pha_list
mean_values = [entry[1] for entry in max_count_pha_list if entry[1] is not None]
len(mean_values)


# In[445]:


# Number of bins for histograms
n = len(mean_values)
k = 1 + np.log2(n)
bin_count = int((pha_max - pha_min) / k)
print(bin_count)
N, bins, _ = plt.hist(mean_values, bins=bin_count)


# In[446]:


p0 = [np.max(N),np.mean(mean_values),np.std(mean_values)]
p0


# In[447]:


def gauss(x,amp,mean,stdev):
    return amp*np.exp(-(x-mean)**2/(2*stdev**2))
bin_centers = (bins[:-1] + bins[1:]) / 2


# In[448]:


params, pcov = curve_fit(gauss,bin_centers,N,p0)
errors = np.sqrt(np.diag(pcov))
params


# In[449]:


fit_curve = gauss(bin_centers,*params)
plt.plot(bin_centers, fit_curve, 'r-', label='Gaussian Fit over full data')
N, bins, _ = plt.hist(mean_values, bins=bin_count)
plt.xlabel("PHA")
plt.ylabel("Counts")
plt.legend()
plt.show()
print("No. of bins, Mean, Deviation")
for p,e in zip(params, errors):
    print(f"{p:0.1f} +- {e:0.1f}")
print("Approximate resolution : {:0.1f}%".format(100* 2.35 * params[2] / params[1]))


# In[434]:


import csv
csv_filename = '/home/suman/Desktop/30K_DATA/det0_good_data_30k/gaussian_on_gaussian_count_D1.csv'

with open(csv_filename, mode='a', newline='') as f:
    writer = csv.writer(f)
    with open(csv_filename, mode='r') as file:
        lines = file.readlines()
        print(len(lines))
        if len(lines) == 0:
            writer.writerow(['count_pha', 'deviation', 'Energy_value'])
    
    # Access the specific elements in params
    count_pha = params[1]
    deviation = params[2]
    writer.writerow([count_pha, deviation, Energy_value])

print(f"List of maximum count PHA values and fixed number saved to '{csv_filename}'")


# In[ ]:





# In[ ]:




