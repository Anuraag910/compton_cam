# -*- coding: utf-8 -*-
"""
Created on Wed Jul  5 13:04:23 2023

@author: USER
"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table, join, Column
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
import os.path
from matplotlib.gridspec import GridSpec

class MplColorHelper:
    # from https://stackoverflow.com/questions/26108436/how-can-i-get-the-matplotlib-rgb-color-given-the-colormap-name-boundrynorm-an/26109298#26109298

    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = mpl.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)

def plotdph(ax, data, 
            drawgrid=True, orient_scat=False,
            pix_col='pixid', title="DPH"):
    """
    Calculate the DPH from given data, and plot it on the given plot axis object
    Add proper numbers, check orientation, add colourbar
    """
    pixels = np.arange(-0.5, 16, 1)
    pix_x = data[pix_col] % 16
    pix_y = data[pix_col] // 16
    hist, _, _ = np.histogram2d(pix_x, pix_y, bins=(pixels, pixels))
    if orient_scat: 
        hist = np.rot90(hist)
    image = ax.imshow(hist, origin="upper")
    ax.set_title(title)
    ax.set_aspect('equal')
    if orient_scat:
        ax.set_xticks(range(0, 16, 2))
        ax.set_yticks(range(0, 16, 2), reversed(range(0, 16, 2)))
        ax.set_xlabel("Y")
        ax.set_ylabel("X")
        ax.yaxis.set_label_position("right")
        ax.yaxis.tick_right()
        plt.colorbar(image, location="left")
    else:
        ax.set_xticks(range(0, 16, 2))
        ax.set_yticks(range(0, 16, 2))
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        plt.colorbar(image)
    if drawgrid:
        for i in range(15):
            plt.axvline(i+0.5, lw=1, color='#888', alpha=0.5)
            plt.axhline(i+0.5, lw=1, color='#888', alpha=0.5)
    return

def plotspec_single(ax, data, specbin=16, smin=0, smax=4096, symlog=True, 
             spec_col="pha", title="Spectrum"):
    """
    Make a spectrum histogram for given data.
    """
    bins = np.arange(smin, smax, specbin)
    ax.hist(data[spec_col], bins=bins, histtype='step')
    ax.set_xlabel(spec_col)
    ax.set_ylabel(f"Counts / {specbin} bins")
    ax.set_title(title)
    if symlog: ax.set_yscale("symlog")
    return

def overview_figure(data_abs, data_scat, ntbin = 100, 
                    specbin=16, smax=1500, suptitle="Overview"):
    """
    Get the basic data table, and make a common overview figure
    """
    fig = plt.figure(figsize=(6, 8))
    gs = GridSpec(3, 2, figure=fig)

    dph_a = fig.add_subplot(gs[0, 0])
    plotdph(dph_a, data_scat, orient_scat=True, title="Scatterer DPH (rotated)")
    dph_b = fig.add_subplot(gs[0, 1])
    plotdph(dph_b, data_abs, title="Absorber DPH")

    spec_a = fig.add_subplot(gs[1, 0])
    plotspec_single(spec_a, data_scat, title="Scatterer spectrum", 
            specbin=specbin, smax=smax)
    spec_b = fig.add_subplot(gs[1, 1], sharex=spec_a, sharey=spec_a)
    plotspec_single(spec_b, data_abs, title="Absorber spectrum", 
            specbin=specbin, smax=smax)

    lc = fig.add_subplot(gs[2, :])
    t_all = np.concatenate((data_abs['time'], data_scat['time']))
    t_min, t_max = np.min(t_all), np.max(t_all)
    tbins = np.linspace(t_min, t_max, ntbin)
    lc.hist(data_abs['time'], bins=tbins, histtype='step', label='Absorber')
    lc.hist(data_scat['time'], bins=tbins, histtype='step', label='Scatterer')
    lc.legend()

    fig.suptitle(suptitle)
    fig.tight_layout()
    return fig

def calculate_compton(data_abs, data_scat,
                      pha_max = 4096, pha_min = 0):
    """
    Find compton events
    Apply compton criterion as a column (don't delete events)
    Identify scatterer and absorber (defined as "1" and "2", due to default join names)
    """
    compton = join(data_abs, data_scat, keys='time')
    compton_energy = compton['pha_1']+compton['pha_2']
    compton_energy.name = "comp_en"
    compton.add_column(compton_energy)

    # Scatterer has lower energy
    scatterer = np.repeat(1, len(compton))
    scatterer[compton['pha_1'] > compton['pha_2']] = 2
    compton.add_column(Column(scatterer, name="det_scat"))

    # Compton criterion
    is_comp = np.repeat(True, len(compton))
    is_comp[compton_energy > pha_max] = False
    is_comp[compton_energy < pha_min] = False
    compton.add_column(Column(is_comp, name="is_comp"))

    return compton

def pix_en_scatter(ax, compton, detid,
                   pix_col='pixid', title="DPH"):
    """
    Scatter plot of energies of photons for a detector
    """
    comp_sel = compton[compton['is_comp']]
    pix_col = f"{pix_col}_{detid}"
    datalen = len(comp_sel)
    pix_x = comp_sel[pix_col] % 16 + np.random.uniform(-0.5, 0.5, datalen)
    pix_y = comp_sel[pix_col] // 16 + np.random.uniform(-0.5, 0.5, datalen)
    
    # Now see if the detector was a scatterer or absorber
    was_scat = (comp_sel["det_scat"] == detid)
    print(np.sum(was_scat), len(was_scat))

    scatplot = ax.scatter(pix_x[was_scat], pix_y[was_scat], c=comp_sel[was_scat]['comp_en'], marker='x', label="Scatterer")
    scatplot = ax.scatter(pix_x[~was_scat], pix_y[~was_scat], c=comp_sel[~was_scat]['comp_en'], marker='o', label="Absorber")
    ax.set_title(title)
    ax.legend()
    ax.set_xlabel("Pix X")
    ax.set_ylabel("Pix Y")
    plt.colorbar(scatplot)
    return

def energy_scatter(ax, compton, gain=16.66,
                   emin=0, emax=4095, 
                   energies=[31., 81., 302., 356], res=0.1):
    """
    Take compton data, make a scatter plot of e1 versus e2
    Mark out bands for where you expect single peaks and compton peaks
    """
    scatterplot = ax.scatter(compton['pha_1'], compton['pha_2'], c=compton['comp_en'])
    ax.set_xlabel("PHA 1")
    ax.set_ylabel("PHA 2")
    cbar = plt.colorbar(scatterplot)
    cbar.set_alpha(0.5)
    colormap = MplColorHelper(cbar.cmap.name, cbar.vmin, cbar.vmax)

    elow_calc = -1000

    res = np.repeat(res, len(energies))

    for thisres, energy in zip(res, energies):
        line_min = energy * gain * (1 - thisres/2)
        line_max = energy * gain * (1 + thisres/2)
        ax.axvspan(line_min, line_max, color='#555', alpha=0.2)
        ax.axhspan(line_min, line_max, color='#555', alpha=0.2)
        #
        # Draw a diagonal. Vertices to be picked outside range.

        xx = [elow_calc, elow_calc, line_max-elow_calc, line_min-elow_calc]
        yy = [line_min-elow_calc, line_max-elow_calc, elow_calc, elow_calc]
        ax.fill(xx, yy, color=colormap.get_rgb(energy * gain), alpha=0.2)
        cbar.ax.axhline(energy * gain, color='black')

    ax.set_xlim(emin, emax)
    ax.set_ylim(emin, emax)
    ax.set_title(f"{len(compton)} Compton events")


if __name__ == "__main__":
    infile = "C:/Users/USER/Desktop/STC_Analysis/data_plot/20230711_1717_100pkts_org_code.fits"
    # # Am
    # line_energies = [59.4]
    # smax = 2000
    # Ba
    line_energies=[31., 81., 302., 356]
    smax = 4096
    plotfilename = infile.replace(".fits", ".pdf")
    suptitle = os.path.basename(infile)
    showfigs = True
    savefigs = True
    
    # Read and split data
    data = Table.read(infile)
    data_abs = data[data['detid'] == 0]
    data_scat = data[data['detid'] == 1]

    # Set up plotting
    if savefigs: plotfile = PdfPages(plotfilename)

    # Quick view plot
    fig0 = overview_figure(data_abs, data_scat, smax=smax, suptitle=suptitle)
    if savefigs: plotfile.savefig(fig0)
    if showfigs: fig0.show()

    # Find compton events
    # Use astropy table join, it will now create columns called: detid_1 pixid_1 pha_1 detid_2 pixid_2 pha_2
    compton = calculate_compton(data_abs, data_scat)

    # Plot spectra
    fig_spec = plt.figure()
    specbins = np.arange(0, smax, 16)
    plt.hist(data_scat["pha"], bins=specbins, density=True, alpha=0.5, label="Scatterer")
    plt.hist(data_abs["pha"], bins=specbins, density=True, alpha=0.5, label="Absorber")
    plt.hist(compton['comp_en'], bins=specbins, density=True, alpha=0.5, label="Compton")
    plt.legend()
    plt.title(suptitle)
    if savefigs: plotfile.savefig(fig_spec)
    if showfigs: fig_spec.show()

    # Pixel locations of compton events
    fig_comp, (ax_scat, ax_abs) = plt.subplots(1, 2, figsize=(8, 4))
    pix_en_scatter(ax_scat, compton, 1, title="Det 1")
    pix_en_scatter(ax_abs, compton, 2, title="Det 2")
    fig_comp.tight_layout()
    if savefigs: plotfile.savefig(fig_comp)
    if showfigs: fig_comp.show()

    # Main compton plot: energy versus energy
    fig_1, ax = plt.subplots(1, 1)
    energy_scatter(ax, compton, energies=line_energies, emax=smax)
    if savefigs: plotfile.savefig(fig_1)
    if showfigs: fig_1.show()
    
    if savefigs: plotfile.close()