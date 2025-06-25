import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from pathlib import Path
import heapq
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.backends.backend_pdf import PdfPages
from IPython import embed



def generate_sigmahists(pixel_sigma_array_am,pixel_sigma_array_eu1,pixel_sigma_array_eu2,pixel_chi_array,pdf_name):
   
    fig = plt.figure(tight_layout=True,figsize=(15,10),dpi=200)
    fig.suptitle('PUT TITLE')
    gs = gridspec.GridSpec(2, 2)

    #CHI SQUARED DISTRIBUTION
    ax1 = fig.add_subplot(gs[0,0]) 
    H_chi = np.reshape(pixel_chi_array,(16,16))

    ax1.set_xticks(np.arange(0,16,1))
    ax1.set_yticks(np.arange(0,16,1))
    ax1.set_xticklabels(np.arange(0,16,1))
    ax1.set_yticklabels(np.arange(0,16,1))
    #minor tick labels
    ax1.set_xticks(np.arange(-0.5,15.6,1),minor=True)
    ax1.set_yticks(np.arange(-0.5,15.6,1),minor=True)
    ax1.grid(which='minor',color='w',ls='-',lw=1)
    
    im = ax1.imshow(H_chi, origin='lower',cmap='YlOrRd')
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("bottom", size="5%", pad=0.25)
    plt.colorbar(im, orientation="horizontal", cax=cax)
    plt.xlabel("Chi squared value")

    #AM SIGMAS
    ax2 = fig.add_subplot(gs[0,1])
    H_eu = np.reshape(pixel_sigma_array_am,(16,16))

    ax2.set_xticks(np.arange(0,16,1))
    ax2.set_yticks(np.arange(0,16,1))
    ax2.set_xticklabels(np.arange(0,16,1))
    ax2.set_yticklabels(np.arange(0,16,1))
    #minor tick labels
    ax2.set_xticks(np.arange(-0.5,15.6,1),minor=True)
    ax2.set_yticks(np.arange(-0.5,15.6,1),minor=True)
    ax2.grid(which='minor',color='w',ls='-',lw=1)
    
    im = ax2.imshow(H_eu, origin='lower',cmap='YlOrRd')
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("bottom", size="5%", pad=0.25)
    plt.colorbar(im, orientation="horizontal", cax=cax)
    plt.xlabel("Am sigma")

    #EU1 SIGMAS
    ax3 = fig.add_subplot(gs[1,0])
    H_eu1 = np.reshape(pixel_sigma_array_eu1,(16,16))
    
    ax3.set_xticks(np.arange(0,16,1))
    ax3.set_yticks(np.arange(0,16,1))
    ax3.set_xticklabels(np.arange(0,16,1))
    ax3.set_yticklabels(np.arange(0,16,1))
    #minor tick labels
    ax3.set_xticks(np.arange(-0.5,15.6,1),minor=True)
    ax3.set_yticks(np.arange(-0.5,15.6,1),minor=True)
    ax3.grid(which='minor',color='w',ls='-',lw=1)
    
    im = ax3.imshow(H_eu1, origin='lower',cmap='YlOrRd')
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("bottom", size="5%", pad=0.25)
    plt.colorbar(im, orientation="horizontal", cax=cax)
    plt.xlabel("Eu1 sigma")

    ax4 = fig.add_subplot(gs[1,1])
    H_eu2 = np.reshape(pixel_sigma_array_eu2,(16,16))
    
    ax4.set_xticks(np.arange(0,16,1))
    ax4.set_yticks(np.arange(0,16,1))
    ax4.set_xticklabels(np.arange(0,16,1))
    ax4.set_yticklabels(np.arange(0,16,1))
    #minor tick labels
    ax4.set_xticks(np.arange(-0.5,15.6,1),minor=True)
    ax4.set_yticks(np.arange(-0.5,15.6,1),minor=True)
    ax4.grid(which='minor',color='w',ls='-',lw=1)
    
    im = ax4.imshow(H_eu2, origin='lower',cmap='YlOrRd')
    divider = make_axes_locatable(ax4)
    cax = divider.append_axes("bottom", size="5%", pad=0.25)
    plt.colorbar(im, orientation="horizontal", cax=cax)
    plt.xlabel("Eu2 sigma")
    pdf_name.savefig()


#QUICKPLOT FUNCTION
def PHAs_of_Pixel(table,ID): 
        PHAs=[]
        pixid_table=table[table['pixID']==ID]
        counts=len(pixid_table)
        PHAs=(pixid_table['PHA'])
        return PHAs,counts



def pixid_counts_table(table,lower_range):
    counts_per_pixel_array=[]
    for ID in range(256):
        PHAs_pix,counts_pix=PHAs_of_Pixel(table, ID)
        counts_per_pixel_array.append(counts_pix)
    counts_per_pixel_array=np.array(counts_per_pixel_array)
    noisy_ids=heapq.nlargest(lower_range, range(len(counts_per_pixel_array)), counts_per_pixel_array.take)
    noisy_ids_hex = np.vectorize(hex)(noisy_ids)
    table_noise=np.vstack((noisy_ids_hex,counts_per_pixel_array[noisy_ids]))
    table_noise=np.transpose(table_noise)
    print(np.array(table_noise).shape)
    return table_noise


#FITTING FUNCTIONS
def compute_chi_square(x,y,function,*popt,compute_red_chi_sq=False):
    y_error=np.sqrt(1+y)
    chi_square = np.sum(((y-function(x,*popt)) / (y_error))**2)
    red_chi_sq = chi_square / (y.size - len(popt))
    #for i in range(0,y.size,1):
    #    print("y : {}    |   f(x) : {}   |    residual : {}   |   y_error : {}    ".format(y[i],function(x[i],*popt),y[i]-function(x[i],*popt),y_error[i]))

    if compute_red_chi_sq: 
        return chi_square,red_chi_sq
    else:
        return chi_square

def gauss(x, a, x0, sigma):
    return  a * np.exp(-(x - x0)**2 / (2 * sigma**2))

def multi_gauss(x,*args):
    f=np.zeros(x.shape)
    if  len(args)%3!= 0:
        print("error")
    else:
        n_gauss=len(args)//3
        for i in range(n_gauss):
            f+= gauss(x,*args[3*i:3*(i+1)])
    return f

def poly_multigauss(x,*args):
    a,b,c,d = args[:4]
    

    #function=(a*x**3 +b*x**2+c*x**1+d)+ multi_gauss(x,*args[4:])
    #print("############################")
    #print(x.shape)
    #print(np.min(x))
    #print(np.max(x))
    #print("############################")
    leg=np.polynomial.legendre.Legendre(args[:5],domain=[np.min(x),np.max(x)])
    function= leg(x)  + multi_gauss(x,*args[5:])
    
    return function

def pha2en(pha, gain, offset):
    return pha * gain + offset

def E_to_pha_values(det_gain,det_offset):
    E_am=[50,70]
    E_eu1=[75,95]
    E_eu2=[95,115]
    pha_am_min=(E_am[0] - det_offset) / det_gain
    pha_am_max=(E_am[1] - det_offset) / det_gain
    pha_eu1_min=(E_eu1[0] - det_offset) / det_gain
    pha_eu1_max=(E_eu1[1] - det_offset) / det_gain
    pha_eu2_min=(E_eu2[0] - det_offset) / det_gain
    pha_eu2_max=(E_eu2[1] - det_offset) / det_gain

    return [round(pha_am_min),round(pha_am_max)],[round(pha_eu1_min),round(pha_eu1_max)],[round(pha_eu2_min),round(pha_eu2_max)]
    

def polygauss_curvefit(channels, fullspec , am_reg = [220, 280],eu1_reg = [340, 420],eu2_reg = [420, 500]
    ,funct_noise=[0,0,0,0,0], phaspace= True):  

    if phaspace:

        am_min=am_reg[0]            
        am_max=am_reg[1]
        eu1_min=eu1_reg[0]            
        eu1_max=eu1_reg[1]
        eu2_min=eu2_reg[0]            
        eu2_max=eu2_reg[1]

        lower_bounds_pha=(-np.inf,  -np.inf,  -np.inf,  -np.inf,-np.inf,
                            0,      am_reg[0],      3,
                            0,      eu1_reg[0],     3,
                            0,      eu2_reg[0],     3  )
        upper_bounds_pha=(np.inf,   np.inf, np.inf   ,np.inf,   np.inf,    
                         np.inf,        am_reg[1],      40,   
                         np.inf,        eu1_reg[1],     60,   
                         np.inf,        eu2_reg[1],     60)
        
        amp_am=max(fullspec[am_min:am_max])
        mean_am=np.mean(channels[am_min:am_max])
        sigma_am=np.std(channels[am_min:am_max])
        
        amp_eu1=max(fullspec[am_min:am_max])
        mean_eu1=np.mean(channels[eu1_min:eu1_max])
        sigma_eu1=np.std(channels[eu1_min:eu1_max])
        
        amp_eu2=max(fullspec[eu2_min:eu2_max])
        mean_eu2=np.mean(channels[eu2_min:eu2_max])
        sigma_eu2=np.std(channels[eu2_min:eu2_max])

        #if np.sum(fullspec[mean_am-10:mean_am+10]) >= poly_multigauss()
        
        
        popt_polygauss,pcov_polygauss=curve_fit(poly_multigauss,channels[:-1],fullspec,
                                                p0=[*funct_noise,amp_am,mean_am,sigma_am,amp_eu1,mean_eu1,sigma_eu1,amp_eu2,mean_eu2,sigma_eu2],
                                                bounds=(lower_bounds_pha    ,   upper_bounds_pha),
                                                sigma=np.sqrt(1+fullspec),
                                                absolute_sigma=True,
                                                )#full_output = True
                                                
        return popt_polygauss,pcov_polygauss

    else:

        am_reg = [50, 70]
        eu1_reg = [75, 95]
        eu2_reg = [95, 115]
        funct_noise=[ 0,0,0,0,0]  
        am_min=am_reg[0]            
        am_max=am_reg[1]
        eu1_min=eu1_reg[0]            
        eu1_max=eu1_reg[1]
        eu2_min=eu2_reg[0]            
        eu2_max=eu2_reg[1]

        lower_bounds_energy=(-np.inf,-np.inf,  -np.inf,  -np.inf,  -np.inf,
                        0,   55,    0,
                        0,   70,    0,
                        0,   95,    0)
        upper_bounds_energy=(np.inf,np.inf,   np.inf,    np.inf,   np.inf, 
                      np.inf,   65,    15,  
                      np.inf,   95,    15,
                      np.inf,   115,   15   )

        fullspec=np.array(fullspec)
        amp_am=np.max(fullspec[am_min:am_max])
        mean_am=60
        sigma_am=np.std(channels[am_min:am_max])
        
        amp_eu1=np.max(fullspec[am_min:am_max])
        mean_eu1=85
        sigma_eu1=np.std(channels[eu1_min:eu1_max])
        
        amp_eu2=np.max(fullspec[eu2_min:eu2_max])
        mean_eu2=105
        sigma_eu2=np.std(channels[eu2_min:eu2_max])

        popt_polygauss,pcov_polygauss=curve_fit(poly_multigauss,channels[42:150],fullspec[42:150],
                                                p0=[*funct_noise,amp_am,mean_am,sigma_am,amp_eu1,mean_eu1,sigma_eu1,amp_eu2,mean_eu2,sigma_eu2],
                                                bounds=(lower_bounds_energy    ,   upper_bounds_energy),
                                                sigma=np.sqrt(1+fullspec[42:150]),
                                                absolute_sigma=True
                                                )#full_output = True
                                                
        return popt_polygauss,pcov_polygauss


###################################         line+ gauss Varun Fitting             ##########################################


def linegauss (x,m,c,a,mean,sigma):
    return m*(x) + c + a * np.exp(-0.5 *(x-mean)**2 / (sigma)**2 )



def iterative_2sig1sig_fit(channels, fullspec,src_channel,iterate=False):  #converted range form E to pha
    m=0
    c=0
    a=max(fullspec)
    mean=src_channel 
    sigma=0.2*src_channel /2.35
    channels=channels-mean

    p0=[m,c,a,0, sigma]
    lower_bounds=(-np.inf,-np.inf,  0,  -20,  0)
    upper_bounds=(np.inf, np.inf,   np.inf,    20,   80)
                    
    popt_iter,pcov_iter = curve_fit(linegauss, channels, fullspec, p0=p0,sigma=np.sqrt(1+fullspec),
                          bounds=(lower_bounds,upper_bounds),absolute_sigma=True)
    m_iter=popt_iter[0]
    c_iter=popt_iter[1]
    a_iter=popt_iter[2]
    mean_iter=popt_iter[3]
    sigma_iter=popt_iter[4]
    min_iter=mean_iter-sigma_iter
    max_iter=mean_iter+2*sigma_iter
    print("1st iteration completed, parameters :")
    #print(popt_iter,pcov_iter)
    if iterate: 
        for i in range(5):
            print("Iteration no .............. {}".format(i))
            range_iter=np.where((channels > min_iter)&(channels < max_iter))[0]
            try:
                popt_iter,pcov_iter=curve_fit(linegauss, channels[range_iter], fullspec[range_iter], 
                                            p0=[m_iter,c_iter,a_iter, mean_iter, sigma_iter])
            except:
                print("stopped at",i)

            m_iter=popt_iter[0]
            c_iter=popt_iter[1]
            a_iter=popt_iter[2]
            mean_iter=popt_iter[3]
            sigma_iter=popt_iter[4]
            min_iter=popt_iter[3]-popt_iter[4]
            max_iter=popt_iter[3]+3*popt_iter[4]
        
        mean_final=popt_iter[3]+mean
        sigma_final=popt_iter[4]
        range_final=[int(min_iter+mean),int(max_iter+mean)]

        fwhm_fit=2.355*popt_iter[4]
        resolution=fwhm_fit/(mean_final)*100
        
        mean_error=np.sqrt(np.diag(pcov_iter)[3])
        sigma_error=np.sqrt(np.diag(pcov_iter)[4])
        res_error=resolution * np.sqrt((sigma_error/sigma_final)**2 + (mean_error/mean_final)**2)
        
        return popt_iter,pcov_iter,resolution,res_error,mean_final,range_final
    else:
        mean_final=popt_iter[3]+mean
        sigma_final=popt_iter[4]
        range_final=[int(min_iter+mean),int(max_iter+mean)]

        fwhm_fit=2.355*popt_iter[4]
        resolution=fwhm_fit/(mean_final)*100

        mean_error=np.sqrt(np.diag(pcov_iter)[3])
        sigma_error=np.sqrt(np.diag(pcov_iter)[4])
        res_error=resolution * np.sqrt((sigma_error/sigma_final)**2 + (mean_error/mean_final)**2)

        return popt_iter,pcov_iter,resolution,res_error,mean_final,range_final
    

