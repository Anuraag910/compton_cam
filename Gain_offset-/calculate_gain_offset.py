#+-10kev range find maxima. that is central value. find place fir half max. difference bw both is fwhm.*2 is the fitting range

from astropy.io import fits
import numpy as np
import argparse
from pathlib import Path
from astropy.table import Table
from scipy.optimize import curve_fit
import Daksha_utils as D
from pylab import title, figure, xlabel, ylabel, xticks, bar, legend, axis, savefig
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from IPython import embed


parser = argparse.ArgumentParser(description="""Script to convert tdms files dumped from
                                 LabView to fits files for further processsing""")
parser.add_argument("inpath", type=str, help="Path to the reference .fits files")
args = parser.parse_args()


#READING PATHS AND CREATING FOLDERS
file=Path(args.inpath)
filename=file.stem
file_address=file.parent.parent
outpath=file_address.joinpath('postprocessing','ref_res_plots',filename[:-5]+'_gain_offset')
outpath.mkdir(parents=True,exist_ok=True)  
print("folder made at",outpath)

# Load data
infile = Path(args.inpath)
all_data = Table.read(infile) # FPGA_Time detID pixID  PHA
detids=np.unique(all_data['detID'])
print("Available detectors are: ",detids )
E_lines = [59.5, 86.5, 105.3]

for detid in detids:                               
    print("########################################     detid : {}  ############################".format(detid))
    data_det = all_data[all_data['detID'] == detid]
    pha_det=data_det['PHA']
    fullspec, channels = np.histogram(pha_det, bins=np.arange(0,1023,1))

    #setting pdf generation
    pp_pixels = PdfPages(outpath.joinpath(infile.stem[:-5]+'_detid_{}_gain_offset_ref_pixels.pdf'.format(detid)))
    pp_det = PdfPages(outpath.joinpath(infile.stem[:-5]+'_detid_{}_gain_offset_detananalysis.pdf'.format(detid)))
    
    #DET FITTING
    range_init_am =[200,300]
    range_init_eu1 =[300,420]
    range_init_eu2 =[420,500]
    
    mean_init_am=np.argmax(fullspec[range_init_am[0]:range_init_am[1]])+range_init_am[0]
    mean_init_eu1=np.argmax(fullspec[range_init_eu1[0]:range_init_eu1[1]])+range_init_eu1[0]
    mean_init_eu2=np.argmax(fullspec[range_init_eu2[0]:range_init_eu2[1]])+range_init_eu2[0]
    
    amp_det_am=np.max(fullspec[range_init_am[0]:range_init_am[1]])
    amp_det_eu1=np.max(fullspec[range_init_eu1[0]:range_init_eu1[1]])
    amp_det_eu2=np.max(fullspec[range_init_eu2[0]:range_init_eu2[1]])
    

    

    #iterfit returns        : popt_iter,pcov_iter,resolution,res_error,mean_final,range_final
    #popt and pcov orders   : m,c,a,mean,sigma

    popt_iterfit_det_am, pcov_iterfit_det_am, _,_,mean_iterfit_det_am,range_iterfit_det_am = D.iterative_2sig1sig_fit(channels[range_init_am[0]:range_init_am[1]],
                                                                                                                fullspec[range_init_am[0]:range_init_am[1]],
                                                                                                                mean_init_am,iterate=True)
    errors_iterfit_det_am=np.sqrt(np.diag(pcov_iterfit_det_am))

    popt_iterfit_det_eu1,pcov_iterfit_det_eu1, _,_,mean_iterfit_det_eu1,range_iterfit_det_eu1=D.iterative_2sig1sig_fit(channels[range_init_eu1[0]:range_init_eu1[1]],
                                                                                                               fullspec[range_init_eu1[0]:range_init_eu1[1]],
                                                                                                               mean_init_eu1,iterate=True)
    errors_iterfit_det_eu1=np.sqrt(np.diag(pcov_iterfit_det_eu1))

    popt_iterfit_det_eu2,pcov_iterfit_det_eu2, _,_,mean_iterfit_det_eu2,range_iterfit_det_eu2=D.iterative_2sig1sig_fit(channels[range_init_eu2[0]:range_init_eu2[1]],
                                                                                                               fullspec[range_init_eu2[0]:range_init_eu2[1]],
                                                                                                               mean_init_eu2,iterate=True)
    errors_iterfit_det_eu2=np.sqrt(np.diag(pcov_iterfit_det_eu2))

    #linefit

    E_lines = [59.54, 86.55, 105.31]
    pha_det_lines = [mean_iterfit_det_am,mean_iterfit_det_eu1,mean_iterfit_det_eu2]
    popt_pha2E_det,pcov_pha2E_det=curve_fit(D.pha2en,pha_det_lines,E_lines)
    errors_pha2E_det=np.sqrt(np.diag(pcov_pha2E_det))
        
    det_gain=popt_pha2E_det[0]
    det_offset=popt_pha2E_det[1]
    print("det gain : {} , det offset : {} :".format(det_gain,det_offset))
    
    chi_value_det_am,red_chi_sq_det_am=D.compute_chi_square(channels[range_iterfit_det_am[0]:range_iterfit_det_am[1]]-mean_init_am,
                                                        fullspec[range_iterfit_det_am[0]:range_iterfit_det_am[1]],
                                                        D.linegauss,*popt_iterfit_det_am,compute_red_chi_sq=True)
    chi_value_det_eu1,red_chi_sq_det_eu1=D.compute_chi_square(channels[range_iterfit_det_eu1[0]:range_iterfit_det_eu1[1]]-mean_init_eu1,
                                                    fullspec[range_iterfit_det_eu1[0]:range_iterfit_det_eu1[1]],
                                                    D.linegauss,*popt_iterfit_det_eu1,compute_red_chi_sq=True)
    chi_value_det_eu2,red_chi_sq_det_eu2=D.compute_chi_square(channels[range_iterfit_det_eu2[0]:range_iterfit_det_eu2[1]]-mean_init_eu2,
                                                    fullspec[range_iterfit_det_eu2[0]:range_iterfit_det_eu2[1]],
                                                    D.linegauss,*popt_iterfit_det_eu2,compute_red_chi_sq=True)
    
    #CALL THE PLOTTING FUNCTION AND THE TABLE FUNCTION HERE AFTER MAKING THEM IN DAKSHA_UTILS

    fig=plt.figure(tight_layout=True,figsize=(16,9))
    fig.suptitle('detector ID : {}'.format(detid))  #put all imp info here with pixel
    gs = gridspec.GridSpec(1, 2,width_ratios=[3,1])
    ax1 = fig.add_subplot(gs[0,0])
    ax1.plot(channels[range_iterfit_det_am[0]:range_iterfit_det_am[1]],D.linegauss(channels[range_iterfit_det_am[0]:range_iterfit_det_am[1]]-int(mean_init_am),*popt_iterfit_det_am))
    ax1.plot(channels[range_iterfit_det_eu1[0]:range_iterfit_det_eu1[1]],D.linegauss(channels[range_iterfit_det_eu1[0]:range_iterfit_det_eu1[1]]-int(mean_init_eu1),*popt_iterfit_det_eu1))
    ax1.plot(channels[range_iterfit_det_eu2[0]:range_iterfit_det_eu2[1]],D.linegauss(channels[range_iterfit_det_eu2[0]:range_iterfit_det_eu2[1]]-int(mean_init_eu2),*popt_iterfit_det_eu2))
    ax1.errorbar(channels[140:600],fullspec[140:600],yerr=np.sqrt(fullspec[140:600]),alpha=0.3,color='k',label="raw data",ds='steps-mid',lw=0.8)
    ax1.legend()
    ax1.set_title("detid : {}".format(detid))
    ax1.set_xlabel("PHA")
    ax1.set_ylabel("counts")
    ax1.set_xlim([140,600])

    detparameters_array=np.array([['Gain',                  '{:.4f}'.format(popt_pha2E_det[0]),        '{:.4f}'.format(errors_pha2E_det[0])],
                                      ['Offset',                '{:.2f}'.format(popt_pha2E_det[1]),        '{:.2f}'.format(errors_pha2E_det[1])],
                                      
                                      ['Chi sq am',                '{:.2f}'.format(chi_value_det_am),                       'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                      ['Red Chi sq am',            '{:.2f}'.format(red_chi_sq_det_am),                      'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                      ['Chi sq eu1',                '{:.2f}'.format(chi_value_det_eu1),                       'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                      ['Red Chi sq eu1',            '{:.2f}'.format(red_chi_sq_det_eu1),                      'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                      ['Chi sq eu2',                '{:.2f}'.format(chi_value_det_eu2),                       'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                      ['Red Chi sq eu2',            '{:.2f}'.format(red_chi_sq_det_eu2),                      'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                        
                                      ['Am amp',                '{:.2f}'.format(popt_iterfit_det_am[2]),      '{:.1e}'.format(errors_iterfit_det_am[2])],
                                      ['Am mean',               '{:.2f}'.format(popt_iterfit_det_am[3]),      '{:.1e}'.format(errors_iterfit_det_am[3])],
                                      ['Am sigma',              '{:.2f}'.format(popt_iterfit_det_am[4]),      '{:.1e}'.format(errors_iterfit_det_am[4])],
                                      
                                      ['Eu1 amp',                '{:.2f}'.format(popt_iterfit_det_eu1[2]),      '{:.1e}'.format(errors_iterfit_det_eu1[2])],
                                      ['Eu1 mean',               '{:.2f}'.format(popt_iterfit_det_eu1[3]),      '{:.1e}'.format(errors_iterfit_det_eu1[3])],
                                      ['Eu1 sigma',              '{:.2f}'.format(popt_iterfit_det_eu1[4]),      '{:.1e}'.format(errors_iterfit_det_eu1[4])],
                                      
                                      ['Eu2 amp',                '{:.2f}'.format(popt_iterfit_det_eu2[2]),      '{:.1e}'.format(errors_iterfit_det_eu2[2])],
                                      ['Eu2 mean',               '{:.2f}'.format(popt_iterfit_det_eu2[3]),      '{:.1e}'.format(errors_iterfit_det_eu2[3])],
                                      ['Eu2 sigma',              '{:.2f}'.format(popt_iterfit_det_eu2[4]),      '{:.1e}'.format(errors_iterfit_det_eu2[4])],
                                      ],dtype=object)
    
    ax2=fig.add_subplot(gs[0,1])
    columns = ("Parameter", "value", "error")   
    table = ax2.table(cellText=detparameters_array,loc=0,colLabels=columns)
    ax2.axis('off')
    pp_det.savefig()
    plt.close()

    #DEFINE PIXEL PARAMETERS FROM ENERGY SPACE
    pix_gains=np.zeros(256)
    pix_offsets=np.zeros(256)
    pix_gain_errors=np.zeros(256)
    pix_offsets_errors=np.zeros(256)
    
    range_pixel_am,range_pixel_eu1,range_pixel_eu2=D.E_to_pha_values(det_gain,det_offset)
    counts_det=np.sum(fullspec)

    ############################################################    PIXEL WISE FITTING STARTS    ###################################################################

    for pixel in range(256):

        showfit_am=True
        showfit_eu1=True
        showfit_eu2=True
 
        avg_pix_amp_am=amp_det_am/256
        avg_pix_amp_eu1=amp_det_eu1/256
        avg_pix_amp_eu2=amp_det_eu2/256

        sigma_pix_amp_am=np.sqrt(avg_pix_amp_am)
        sigma_pix_amp_eu1=np.sqrt(avg_pix_amp_eu1)
        sigma_pix_amp_eu2=np.sqrt(avg_pix_amp_eu2)


        print("###############################  pixel {}  #####################".format(pixel))
        data_pixel=data_det[data_det['pixID']==pixel]
        pha_pixel=data_pixel['PHA']
        fullspec_pixel, channels_pixel = np.histogram(pha_pixel, bins=np.arange(0,1000,1))  
        mean_pixel_am=np.argmax(fullspec_pixel[range_pixel_am[0]:range_pixel_am[1]])+range_pixel_am[0]
        mean_pixel_eu1=np.argmax(fullspec_pixel[range_pixel_eu1[0]:range_pixel_eu1[1]])+range_pixel_eu1[0]
        mean_pixel_eu2=np.argmax(fullspec_pixel[range_pixel_eu2[0]:range_pixel_eu2[1]])+range_pixel_eu2[0]

        
        amp_pixel_am=np.max(fullspec_pixel[range_pixel_am[0]:range_pixel_am[1]])
        amp_pixel_eu1=np.max(fullspec_pixel[range_pixel_eu1[0]:range_pixel_eu1[1]])
        amp_pixel_eu2=np.max(fullspec_pixel[range_pixel_eu2[0]:range_pixel_eu2[1]])

        #avg_counts_pixel=np.sum(fullspec_pixel)
        #avg_count_pixel_error=np.sqrt(counts_det)
        check_am=amp_pixel_am >= avg_pix_amp_am - 2* sigma_pix_amp_am
        check_eu1=amp_pixel_eu2 >= avg_pix_amp_eu2 - 2* sigma_pix_amp_am
        check_eu2=amp_pixel_eu1 >= avg_pix_amp_eu1 - 2* sigma_pix_amp_eu1



        if check_am and check_eu1 and check_eu2:
            try:
                popt_iterfit_pixel_am, pcov_iterfit_pixel_am, _,_,mean_iterfit_pixel_am,range_iterfit_pixel_am = D.iterative_2sig1sig_fit(
                                                                                                            channels_pixel[range_pixel_am[0]:range_pixel_am[1]],
                                                                                                            fullspec_pixel[range_pixel_am[0]:range_pixel_am[1]],
                                                                                                            mean_pixel_am,iterate=False)
                errors_iterfit_pixel_am=np.sqrt(np.diag(pcov_iterfit_pixel_am))
                print("Am fitted")
            except:
                popt_iterfit_pixel_am, pcov_iterfit_pixel_am, _,_,mean_iterfit_pixel_am,range_iterfit_pixel_am = popt_iterfit_det_am, pcov_iterfit_det_am, _,_,mean_iterfit_det_am,range_iterfit_det_am
                errors_iterfit_pixel_am=np.sqrt(np.diag(pcov_iterfit_pixel_am))
                print("pixel {} cannot be fitted with Am".format(pixel))
                showfit_am=False
        
            try:
                popt_iterfit_pixel_eu1,pcov_iterfit_pixel_eu1, _,_,mean_iterfit_pixel_eu1,range_iterfit_pixel_eu1=D.iterative_2sig1sig_fit(
                                                                                                                channels_pixel[range_pixel_eu1[0]:range_pixel_eu1[1]],
                                                                                                                fullspec_pixel[range_pixel_eu1[0]:range_pixel_eu1[1]],
                                                                                                                mean_pixel_eu1,iterate=False)
                errors_iterfit_pixel_eu1=np.sqrt(np.diag(pcov_iterfit_pixel_eu1))
                print("Eu1 fitted")
            except:
                popt_iterfit_pixel_eu1,pcov_iterfit_pixel_eu1, _,_,mean_iterfit_pixel_eu1,range_iterfit_pixel_eu1=popt_iterfit_det_eu1,pcov_iterfit_det_eu1, _,_,mean_iterfit_det_eu1,range_iterfit_det_eu1
                errors_iterfit_pixel_eu1=np.sqrt(np.diag(pcov_iterfit_pixel_eu1))
                print("pixel {} cannot be fitted with Eu1".format(pixel)) 
                showfit_eu1=False

            try:
                popt_iterfit_pixel_eu2,pcov_iterfit_pixel_eu2, _,_,mean_iterfit_pixel_eu2,range_iterfit_pixel_eu2=D.iterative_2sig1sig_fit(
                                                                                                                channels_pixel[range_pixel_eu2[0]:range_pixel_eu2[1]],
                                                                                                                fullspec_pixel[range_pixel_eu2[0]:range_pixel_eu2[1]],
                                                                                                                mean_pixel_eu2,iterate=False)
                errors_iterfit_pixel_eu2=np.sqrt(np.diag(pcov_iterfit_pixel_eu2))
                print("Eu2 fitted")
            except:
                popt_iterfit_pixel_eu2,pcov_iterfit_pixel_eu2, _,_,mean_iterfit_pixel_eu2,range_iterfit_pixel_eu2=popt_iterfit_det_eu2,pcov_iterfit_det_eu2, _,_,mean_iterfit_det_eu2,range_iterfit_det_eu2
                errors_iterfit_pixel_eu2=np.sqrt(np.diag(pcov_iterfit_pixel_eu2))
                showfit_eu2=False
                print("pixel {} cannot be fitted with Eu2".format(pixel)) 

        else:
                popt_iterfit_pixel_am, pcov_iterfit_pixel_am, _,_,mean_iterfit_pixel_am,range_iterfit_pixel_am = popt_iterfit_det_am, pcov_iterfit_det_am, _,_,mean_iterfit_det_am,range_iterfit_det_am
                showfit_am=False
                popt_iterfit_pixel_eu1,pcov_iterfit_pixel_eu1, _,_,mean_iterfit_pixel_eu1,range_iterfit_pixel_eu1=popt_iterfit_det_eu1,pcov_iterfit_det_eu1, _,_,mean_iterfit_det_eu1,range_iterfit_det_eu1
                showfit_eu1=False
                popt_iterfit_pixel_eu2,pcov_iterfit_pixel_eu2, _,_,mean_iterfit_pixel_eu2,range_iterfit_pixel_eu2=popt_iterfit_det_eu2,pcov_iterfit_det_eu2, _,_,mean_iterfit_det_eu2,range_iterfit_det_eu2
                showfit_eu2=False
            


        pha_pixel_lines = [mean_iterfit_pixel_am,mean_iterfit_pixel_eu1,mean_iterfit_pixel_eu2]
        popt_pha2E_pixel,pcov_pha2E_pixel=curve_fit(D.pha2en,pha_pixel_lines, E_lines)
        errors_pha2E_pixel=np.sqrt(np.diag(pcov_pha2E_pixel))

        pixel_gain=popt_pha2E_pixel[0]
        pixel_offset=popt_pha2E_pixel[1]
        
        pix_gains[pixel]=pixel_gain
        pix_offsets[pixel]=pixel_offset
        pix_gain_errors[pixel]=errors_pha2E_pixel[0]
        pix_offsets_errors[pixel]=errors_pha2E_pixel[1]


        fig=plt.figure(tight_layout=True,figsize=(16,9))
        fig.suptitle('Pixel ID : {}'.format(pixel))  #put all imp info here with pixel
        gs = gridspec.GridSpec(1, 2,width_ratios=[3,1])
        
        ax1 = fig.add_subplot(gs[0,0])
        if showfit_am:
            plt.plot(   channels_pixel[range_iterfit_pixel_am[0]:range_iterfit_pixel_am[1]],
                        D.linegauss(channels_pixel[range_iterfit_pixel_am[0]:range_iterfit_pixel_am[1]]-int(mean_pixel_am),
                        *popt_iterfit_pixel_am),label='am fit')
        if showfit_eu1:
            plt.plot(   channels_pixel[range_iterfit_pixel_eu1[0]:range_iterfit_pixel_eu1[1]],
                        D.linegauss( channels_pixel[range_iterfit_pixel_eu1[0]:range_iterfit_pixel_eu1[1]]-int(mean_pixel_eu1),
                        *popt_iterfit_pixel_eu1),label='eu1 fit')
        if showfit_eu2:
            plt.plot(   channels_pixel[range_iterfit_pixel_eu2[0]:range_iterfit_pixel_eu2[1]],
                        D.linegauss( channels_pixel[range_iterfit_pixel_eu2[0]:range_iterfit_pixel_eu2[1]]-int(mean_pixel_eu2),
                        *popt_iterfit_pixel_eu2),label='eu2 fit')
        plt.errorbar(channels_pixel[140:600],fullspec_pixel[140:600],yerr=np.sqrt(fullspec_pixel[140:600]),alpha=0.3,color='k',label="raw data",ds='steps-mid',lw=0.8)
        plt.legend()
        plt.title("pixel : {}".format(pixel))
    
        ax1.set_xlabel("PHA")
        ax1.set_ylabel("counts")
        ax1.set_xlim([120,650])

        #CALCULATE CHI SQUARE PARAMETERES TO BE WRITTEN DOWN HERE
        chi_value_pixel_am,red_chi_sq_pixel_am=D.compute_chi_square(channels_pixel[range_iterfit_pixel_am[0]:range_iterfit_pixel_am[1]],
                                                        fullspec_pixel[range_iterfit_pixel_am[0]:range_iterfit_pixel_am[1]],
                                                        D.linegauss,*popt_iterfit_pixel_am,compute_red_chi_sq=True)
        chi_value_pixel_eu1,red_chi_sq_pixel_eu1=D.compute_chi_square(channels_pixel[range_iterfit_pixel_eu1[0]:range_iterfit_pixel_eu1[1]],
                                                        fullspec_pixel[range_iterfit_pixel_eu1[0]:range_iterfit_pixel_eu1[1]],
                                                  D.linegauss,*popt_iterfit_pixel_eu1,compute_red_chi_sq=True)
        chi_value_pixel_eu2,red_chi_sq_pixel_eu2=D.compute_chi_square(channels_pixel[range_iterfit_pixel_eu2[0]:range_iterfit_pixel_eu2[1]],
                                                        fullspec_pixel[range_iterfit_pixel_eu2[0]:range_iterfit_pixel_eu2[1]],
                                                        D.linegauss,*popt_iterfit_pixel_eu2,compute_red_chi_sq=True)
        
        #iterfit returns        : popt_iter,pcov_iter,resolution,res_error,mean_final,range_final
        #popt and pcov orders   : m,c,a,mean,sigma

        pixparameters_array=np.array([['Gain',                  '{:.4f}'.format(popt_pha2E_pixel[0]),        '{:.4f}'.format(errors_pha2E_pixel[0])],
                                      ['Offset',                '{:.2f}'.format(popt_pha2E_pixel[1]),        '{:.2f}'.format(errors_pha2E_pixel[1])],
                                      
                                      ['Chi sq am',                '{:.2f}'.format(chi_value_pixel_am),                       'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                      ['Red Chi sq am',            '{:.2f}'.format(red_chi_sq_pixel_am),                      'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                      ['Chi sq eu1',                '{:.2f}'.format(chi_value_pixel_eu1),                       'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                      ['Red Chi sq eu1',            '{:.2f}'.format(red_chi_sq_pixel_eu1),                      'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                      ['Chi sq eu2',                '{:.2f}'.format(chi_value_pixel_eu2),                       'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                      ['Red Chi sq eu2',            '{:.2f}'.format(red_chi_sq_pixel_eu2),                      'XXXX'    ],   #'{:.1e}'.format(errors_polyfit_pixel[0])],
                                         
                                      ['Am amp',                '{:.2f}'.format(popt_iterfit_pixel_am[2]),      '{:.1e}'.format(errors_iterfit_pixel_am[2])],
                                      ['Am mean',               '{:.2f}'.format(popt_iterfit_pixel_am[3]),      '{:.1e}'.format(errors_iterfit_pixel_am[3])],
                                      ['Am sigma',              '{:.2f}'.format(popt_iterfit_pixel_am[4]),      '{:.1e}'.format(errors_iterfit_pixel_am[4])],
                                      
                                      ['Eu1 amp',                '{:.2f}'.format(popt_iterfit_pixel_eu1[2]),      '{:.1e}'.format(errors_iterfit_pixel_eu1[2])],
                                      ['Eu1 mean',               '{:.2f}'.format(popt_iterfit_pixel_eu1[3]),      '{:.1e}'.format(errors_iterfit_pixel_eu1[3])],
                                      ['Eu1 sigma',              '{:.2f}'.format(popt_iterfit_pixel_eu1[4]),      '{:.1e}'.format(errors_iterfit_pixel_eu1[4])],
                                      
                                      ['Eu2 amp',                '{:.2f}'.format(popt_iterfit_pixel_eu2[2]),      '{:.1e}'.format(errors_iterfit_pixel_eu2[2])],
                                      ['Eu2 mean',               '{:.2f}'.format(popt_iterfit_pixel_eu2[3]),      '{:.1e}'.format(errors_iterfit_pixel_eu2[3])],
                                      ['Eu2 sigma',              '{:.2f}'.format(popt_iterfit_pixel_eu2[4]),      '{:.1e}'.format(errors_iterfit_pixel_eu2[4])],
                                      
                                      ],dtype=object)
    
        ax2=fig.add_subplot(gs[0,1])
        columns = ("Parameter", "value", "error")   
        table = ax2.table(cellText=pixparameters_array,loc=0,colLabels=columns)
        ax2.axis('off')
        pp_pixels.savefig()
        plt.close()

    plt.figure("gainoffset",dpi=200,figsize=(16,9))
    #plt.scatter(pix_gains,pix_offsets, c='r', marker='*',s=3,label='gains and offsets')
    plt.errorbar(pix_gains,pix_offsets,xerr=pix_gain_errors, yerr=pix_offsets_errors,
                 fmt="o",alpha=0.7)
    #plt.errorbar(pix_gains,pix_offsets,  fmt="o",label='offset error')1
    plt.xlabel('pixgains')
    plt.ylabel('pixoffsets')
    plt.title(" gain offset value graph ")
    plt.legend()
    plt.title("detector ID : {}".format(detid))
    pp_det.savefig()
    plt.close()


    #D.generate_sigmahists(pixel_sigma_array_am,pixel_sigma_array_eu1,pixel_sigma_array_eu2,pixel_chi_array,pp_det)
    pp_det.close()
    pp_pixels.close()

    pixpars_table = np.recarray((256,), dtype=[('pix_gain', np.float32), ('pix_offset', np.float64)])
    pixpars_table['pix_gain']=pix_gains
    pixpars_table['pix_offset']=pix_offsets
    print("pix table is: ", pixpars_table)



    HDUList = fits.HDUList()
    HDUList.append(fits.PrimaryHDU())
    gain_offset_HDU = fits.BinTableHDU(data=pixpars_table, name='gainoffsets', uint=True)
    gain_offset_HDU.header['det_gain'] = (det_gain,'detector gain')
    gain_offset_HDU.header['det_offs'] = (det_offset,'detector offset')
    HDUList.append(gain_offset_HDU)
    HDUList.writeto(outpath.joinpath(infile.stem[:-5]+'_detid_{}_gain_offset_ref_calculategainoffsetcode.fits.gz'.format(detid)),overwrite=True)
    print("pix gains and offset ref file saved at :",outpath.joinpath(infile.stem[:-5]+'_detid_{}_gain_offset_ref_calculategainoffsetcode.fits.gz'.format(detid)))

    


    
