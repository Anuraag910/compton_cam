# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 12:57:17 2023

@author: USER
"""

import numpy as np
# import pandas as pd
from astropy.table import Table
import os, glob

def data2fits(infile, outfile):
    with open(infile, 'r') as f:
        l1 = f.readline()

    biglist = l1[1:-1].split(",")
    event_data = np.array(biglist, dtype=np.uint64)
    timestamp = np.uint32((event_data & 0xffffffff00000000) >> 32)
    det_id = np.int16((event_data & 0x00000000ff000000) >> 24)
    pix_id = np.int16((event_data & 0x0000000000ff0000) >> 16)
    energy = np.int16((event_data & 0x000000000000ffff) >> 0)

    tab = Table(data=(timestamp, det_id, pix_id, energy), names=("time", "detID", "pixID", "PHA"))
    tab.write(outfile)

    # df = pd.DataFrame(data=np.array([timestamp, det_id, pix_id, energy]).T, 
    #                   columns=("time", "detid", "pixid", "pha"))
    # detdata = df.groupby(df['detid'])

if __name__ == "__main__":
    basepath = "C:/Users/USER/Desktop/STC_Analysis/20230711/"
    maxcount = -1 # Set <=0 if all files are to be processed, else end after this
    infile = 'datafile1.txt'
    for count, infile in enumerate(glob.glob1(basepath, "*.txt")):
        if count == maxcount:
            print(f"Stopping the loop, max count {maxcount} reached")
            break
        outfile = os.path.join(basepath, infile.replace(".txt", ".fits"))
        if not os.path.exists(outfile):
            print(f"Converting {infile} to {outfile}")
            data2fits(os.path.join(basepath, infile), outfile)
        else:
            print(f"Skipping {infile}: {outfile} already exists.")