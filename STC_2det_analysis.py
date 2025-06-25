import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import csv


with open('C:/Users/USER/Desktop/STC_Analysis/20230705/20230705_1156_ba133_30000pkts_ext_hv_600_20kev_spectra') as f:
    r = csv.reader(f)
    event_data = []
    event_data = [int(s) for line in r for s in line]

#%%

parse_main_buffer = []
for event in event_data:
    timestamp = (event & 0xffffffff00000000) >> 32
    det_id = (event & 0x00000000ff000000) >> 24
    pix_id = (event & 0x0000000000ff0000) >> 16
    energy = (event & 0x000000000000ffff) >> 0
    parse_main_buffer.append((timestamp, det_id, pix_id, energy))
#%%
det0 = []
det1 = []
for x in parse_main_buffer:
    if x[1]== 1:
        det1.append(x)
    elif x[1]== 0:
        det0.append(x)
        
times_det0 = [x[0] for x in det0]
pixels_det0 = [x[2] for x in det0]
energy_det0 = [x[3] for x in det0]

times_det1 = [x[0] for x in det1]
pixels_det1 = [x[2] for x in det1]
energy_det1 = [x[3] for x in det1]
#%%

pixhist = np.bincount(pixels_det0, minlength=256)
plt.figure(figsize=(14,12))
plt.title('det0')
sns.heatmap(pixhist.reshape((16,16)), cmap="icefire", linewidths=1, annot= True, fmt=".0f")
print(f"Number of pixels that gave data: {len(np.unique(pixels_det0))}")

#%%
pixhist = np.bincount(pixels_det1, minlength=256)
plt.figure(figsize=(14,12))
plt.title('det1')
sns.heatmap(pixhist.reshape((16,16)), cmap="icefire", linewidths=1, annot= True, fmt=".0f")
print(f"Number of pixels that gave data: {len(np.unique(pixels_det1))}")
#%%
plt.figure(figsize=(14,7))
plt.hist(energy_det0, bins=range(0,4096,10))
plt.hist(energy_det1, bins=range(0,4096,10))
plt.xlabel("PHA")
#plt.ylim(0,300)
plt.ylabel("Counts")
plt.show()