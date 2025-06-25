import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits

# --- USER INPUT ---
fits_file = r'/home/arya/Desktop/STC_Analysis/BKG_file/2dets_bkg_iitb_20230605_timestamps.fits'        # Replace with your FITS file name
target_pha = 550                     # Change this to the PHA value you're interested in
pha_window = 1                      # Window around target_pha, use 0 for exact match
time_bin_size = 10                  # Bin size in seconds
# ------------------

# Open FITS file and extract data
with fits.open(fits_file) as hdul:
    data = hdul[1].data
    time = data['time']
    pha = data['pha']

# Select PHA within the window
pha_mask = (pha >= target_pha - pha_window) & (pha <= target_pha + pha_window)
selected_time = time[pha_mask]

# Time binning
time_min, time_max = np.min(selected_time), np.max(selected_time)
bins = np.arange(time_min, time_max + time_bin_size, time_bin_size)
counts, bin_edges = np.histogram(selected_time, bins=bins)
bin_centers = 0.5 * (bin_edges[1:] + bin_edges[:-1])

# Count rate = counts / time_bin_size
count_rate = counts / time_bin_size

# Plotting
plt.figure(figsize=(10, 5))
plt.step(bin_centers, count_rate, where='mid', color='blue')
plt.xlabel('Time (s)')
plt.ylabel(f'Count Rate (counts/s) at PHA ≈ {target_pha}')
plt.title('Count Rate Variation Over Time')
plt.grid(True)
plt.tight_layout()
plt.show()
