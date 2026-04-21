
import numpy as np
from matplotlib import pyplot as plt
from astropy.io import fits
from scipy import ndimage
from copy import deepcopy as copy
import math
import sys
#import image
import skimage
import filter
import file_manager
import image_auto as image
import scipy.ndimage
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import warnings

def Preprocess(NIRSpec_angle,x_center_pix_coord,
    y_center_pix_coord,Yh_height,NS_path,NC_path,
    Throughputs_path,debug=False,align=True,wave_slice=None,
    NIRSpec_anomaly_handle=False,fusion_name=None):
    ratio = 3
    file_NC=[fits.open(NC_path[i]) for i in range(len(NC_path))]
    file_NS=fits.open(NS_path) 
    moving_object= file_NC[0][0].header["TARGTYPE"] == "MOVING"
    Yh_width=Yh_height
    Ym_height, Ym_width = ratio * Yh_height, ratio * Yh_width 
    interp_type = 'cubic'
    little_coord_default,Driess = False,False
    dev_coeff,anom_limit = 1e10,30
    exceptions = []
    compute_NIRCam_psf,compute_NIRSpec_psf = False,False
    sym_pad_size = Ym_height
    zero_pad_size = 0
    verbose= False

    if file_NC[0][0].header["DETECTOR"]=="MULTIPLE":
        detector="MEAN" 
    else:
        detector=file_NC[0][0].header["DETECTOR"]

    NIRCam_throughputs=[(Throughputs_path+f'/NIRCam/{detector}_{file_NC[i][0].header["FILTER"]}_system_throughput.txt') 
                        for i in range(len(NC_path))]
    NIRSpec_throughput=(Throughputs_path+f'/NIRSpec/In_Situ/comm_PCE_{file_NS[0].header["FILTER"]}_{file_NS[0].header["GRATING"]}_IFU.fits')
    center_pixel_coord=np.array([x_center_pix_coord,y_center_pix_coord])

    cube = image.NIRSpec_Image(NS_path, Yh_height, Yh_width, center_pixel_coord, NIRSpec_angle, Driess, debug)#
    wavelengths = cube.wavelengths
    NIRCam_throughput_files = [file_manager.read_throughput(NIRCam_throughputs[k]) for k in range(len(NIRCam_throughputs))]
    TP_order=[NIRCam_throughput_files[k][0].max() for k in range(len(NIRCam_throughputs))]
    k_min,k_max=np.argmin(TP_order),np.argmax(TP_order)
    Min_NC_throughput=NIRCam_throughput_files[k_min]
    Max_NC_throughput=NIRCam_throughput_files[k_max]
    if wave_slice is None:
        if (wavelengths<min(Min_NC_throughput[0][np.where(Min_NC_throughput[1]>1e-5)])).sum() > 0:
            ind_min=np.where(wavelengths<min(Min_NC_throughput[0][np.where(Min_NC_throughput[1]>1e-5)]))[0][0]
            ind_min=int(ind_min)
        else:
            ind_min=0
        if (wavelengths>max(Max_NC_throughput[0][np.where(Max_NC_throughput[1]>1e-5)])).sum()> 0:
            ind_max=np.where(wavelengths>max(Max_NC_throughput[0][np.where(Max_NC_throughput[1]>1e-5)]))[0][0]
            ind_max=int(ind_max)
        else:
            ind_max=len(wavelengths)
        ind_min,ind_max
        wave_slice=slice(ind_min,ind_max)
    Yh = cube.preprocess()
    Yh = Yh[wave_slice]
    if NIRSpec_anomaly_handle:
        filt = filter.NIRSpec_Filter(Yh, fusion_name, exceptions, dev_coeff, anom_limit, verbose)    
        Yh = filt.filter()
    hdu = fits.PrimaryHDU(data = Yh)
    Lm = np.zeros((len(NIRCam_throughputs), Yh.shape[0]))
    for k in range(len(NIRCam_throughputs)):
        NIRCam_throughput = file_manager.read_throughput(NIRCam_throughputs[k])
        i = 0
        while(wavelengths[wave_slice][i] < NIRCam_throughput[0][0]):
            i += 1
        j = -1
        while(wavelengths[wave_slice][j] > NIRCam_throughput[0][-1]):
            j -= 1
        Lm[k, i:j] = scipy.interpolate.interp1d(NIRCam_throughput[0], NIRCam_throughput[1])(wavelengths[wave_slice][i:j])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        NC_coord=[WCS(file_NC[i][1].header) for i in range(len(NC_path))]
        NS_coord=WCS(file_NS[1].header)
    Ym_0 = image.NIRCam_Image(NC_path[0], Ym_height, Ym_width, cube.center_coord, cube.angle, cube.fov, moving_object, #moving object 
               little_coord_default, Yh,Lm[0],wave_slice,cube.pix_size, ratio, debug = debug#,center = centers[0]
                             ,NS_coord=cube.NS_coord,NS_center=cube.NS_center,align=align).preprocess()#image.
    #
    Ym = np.array([filter.NIRCam_Filter(image.NIRCam_Image(NC_path[i], Ym_height, Ym_width, 
                      cube.center_coord, cube.angle, cube.fov, moving_object, #moving_object
                       little_coord_default, Yh, Lm[i], wave_slice, cube.pix_size, ratio, Ym_0,
                      debug = debug,NS_coord=cube.NS_coord,NS_center=cube.NS_center,align=align).preprocess(), verbose).filter() for i in range(len(NC_path))])#image.
    
    return Yh,Ym,wavelengths,wave_slice
    
def pick_rank(S):
    for ncomp in range(3,30):
        percent=(S[:ncomp]**2).sum()/((S[:]**2).sum())
        #print("rank",ncomp,"percent",percent)
        if percent>0.9:
            break
    return ncomp 
def calc_PCA(NS_data):
    #do PCA
    Yh_2d=np.reshape(NS_data,(NS_data.shape[0],NS_data.shape[1]*NS_data.shape[2]))
    Lh = np.ones(NS_data.shape[0])
    Yh_Lh_1_2d = np.dot(np.diag(Lh**-1), Yh_2d)
    mean = np.mean(Yh_Lh_1_2d, axis = 1) # NIRSpec mean spectrum
    U, S, V_T = scipy.linalg.svd(Yh_Lh_1_2d.T - mean, full_matrices = False)
    #U, V_T = sklearn.utils.extmath.svd_flip(U, V_T, u_based_decision = False)
    return S
