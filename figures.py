# Code by Landry Marquis
#
# If you use this code, please cite the paper: 
# Marquis L. et al., Fusion of JWST data: Demonstrating practical feasibility,
# Astronomy & Astrophysics, volume 708, 2026

import fusion
import file_manager
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches
import matplotlib.font_manager
import scipy.interpolate
import skimage.transform
import os
import imageio.v3 as iio
import skimage.transform
import matplotlib.animation as animation
from astropy.io import fits
from astropy.visualization import ZScaleInterval
from mpl_toolkits.mplot3d.axes3d import Axes3D
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
matplotlib.set_loglevel('error')


def fusion_principle():

    NGC_1512_path = "/Users/admin/Documents/Fusion_Database/2107_NGC_1512/Hubble-NGC1512.png"
    img = iio.imread(NGC_1512_path)[44:-43, 722:-722]
    img = np.array(img)/np.max(np.array(img))

    fusion_symbol_path = "Figures/Paper_1/Fusion_symbol.png"
    fusion_symbol = iio.imread(fusion_symbol_path)
    fusion_symbol = skimage.transform.resize(fusion_symbol, (200,200,4), order=3)

    waves_coeffs = 1+0*np.random.rand(12)
    waves_coeffs[0] = 1
    waves_coeffs[11] = 1
    w = 21
    f = w//5 + 1
    e = 7/3
    cmap = plt.get_cmap('turbo')
    a,b,c = img.shape
    Fusion = np.ones((int(e*a),int(e*b),c))
    nb_waves = 13
    nb_filters = 5
    coeff_lumi = 1.7
    colors = cmap(np.linspace(0.2, 0.9, nb_waves))

    for i in range(nb_waves-1,-1,-1):
        img_wave = img * colors[i,:3] * waves_coeffs[i-1]
        Fusion[i*int(e*a/w):a+i*int(e*a/w), i*int(e*b/w):b+i*int(e*b/w)] = img_wave * coeff_lumi
        
    NIRSpec = np.ones((int(e*a),int(e*b),c))
    img_sub = skimage.transform.downscale_local_mean(img, (200,200,1))
    img_NIRSpec = skimage.transform.resize(img_sub[:-1,:-1], (2001,2001,3), order=0)
    for i in range(nb_waves-1,-1,-1):
        NIRSpec[i*int(e*a/w):a+i*int(e*a/w), i*int(e*b/w):b+i*int(e*b/w)] = img_NIRSpec * colors[i,:3] * waves_coeffs[i-1] * coeff_lumi

    NIRCam = np.ones((int(e*a),int(e*b),c))
    colors = cmap(np.linspace(0.2, 0.9, nb_filters))
    for i in range(nb_filters-1,-1,-1):
        NIRCam[i*667:a+i*667, i*667:b+i*667] = img * colors[i,:3] * np.mean(np.array([waves_coeffs[3*i-1],waves_coeffs[3*i-2],waves_coeffs[3*i-3]])) * coeff_lumi

    res = np.concatenate((NIRCam/np.max(img), NIRSpec/np.max(img), Fusion/np.max(img)), axis = 1)
    res = skimage.transform.downscale_local_mean(res, (20,20,1))
    res = res[1:-2,1:-2]

    fig = plt.figure()
    fig.set_size_inches((11,4.2))
    font = matplotlib.font_manager.FontProperties(family='Helvetica', size=19)
    grid_size = (2,6)
    text_size = 20
    symbol_size = 65
    ax = plt.subplot2grid(grid_size, (0, 0), colspan=6, rowspan=2)
    ax.set_axis_off()
    ax.imshow(res, origin='lower')
    font2 = matplotlib.font_manager.FontProperties( weight='bold', size=24)
    ax.text(10,213,'a', font = font2)
    ax.text(242,213,'b', font = font2)
    ax.text(476,213,'c', font = font2)
    ax.text(412,53,'=', size = symbol_size)
    ax.text(17,-20,'Imager', size = text_size)
    ax.text(270,-20,'IFU', size = text_size)
    ax.text(491,-20,'Fusion', size = text_size, font = font)
    fig.subplots_adjust(hspace=0.2, wspace=0.2, left=0.005, bottom=0.04, right=0.998, top=1)
    plt.savefig('Figures/Paper_1/Figure_fusion_principle.eps')


def create_directory(x):
    if not os.path.exists(x):
        os.makedirs(x)


def figure_NIRCam(Orion_fusion_name, Titan_fusion_name):

    fig = plt.figure()
    fig.set_size_inches((10,4))
    font = matplotlib.font_manager.FontProperties(family='Helvetica', size=19)
    grid_size = (2,5)
    text_size = 12

    full_fov_d203_506_path = "/Users/admin/Documents/Fusion_Database/1288_Orion/NIRCam/Level3_CLEAR-F210M-B_i2d_aligned.fits"
    z = ZScaleInterval()
    full_fov_NIRCam_d203_506 = (fits.getdata(full_fov_d203_506_path).T)[3379:3589,3310:3774]
    z1,z2 = z.get_limits(full_fov_NIRCam_d203_506)
    ax_fullfov = plt.subplot2grid(grid_size, (0, 0), colspan=2, rowspan=1)
    font2 = matplotlib.font_manager.FontProperties( weight='bold', size=24)
    ax_fullfov.text(-30,-8,'a', font = font2)
    ax_fullfov.set_xticks([])
    ax_fullfov.set_yticks([])
    ax_fullfov.imshow(full_fov_NIRCam_d203_506, vmin=z1, vmax=z2)
    rect = matplotlib.patches.Rectangle((229, 95), 30, 30, angle=35,linewidth=2, edgecolor='r', facecolor='none')
    ax_fullfov.add_patch(rect)
    ax_fullfov.text(3,201,'F210M', bbox={'facecolor': 'white', 'pad': 2}, size = text_size)
    ax_fullfov.set_ylabel("d203-506", fontproperties = font)

    NIRCam_d203_506 = fits.getdata("Fusion_results/" + Orion_fusion_name + "prepro_NIRCam.fits")

    NIRCam_filters = ['F182M','F187N','F210M']
    for f in range(1,NIRCam_d203_506.shape[0]+1):
        ax_NIRCam = plt.subplot2grid(grid_size, (0, 1+f), colspan=1, rowspan=1)
        if f == 1:
            ax_NIRCam.text(-5,30.5,'b', font = font2)
        ax_NIRCam.set_xticks([])
        ax_NIRCam.set_yticks([])
        ax_NIRCam.imshow(NIRCam_d203_506[f-1], origin='lower')
        ax_NIRCam.text(0,0.7,NIRCam_filters[f-1], bbox={'facecolor': 'white', 'pad': 2}, size = text_size)

    NIRCam_Titan = fits.getdata("Fusion_results/" + Titan_fusion_name + "prepro_NIRCam.fits")
    NIRCam_filters = ['F182M','F187N','F200W','F210M','F212N']
    for f in range(1,NIRCam_Titan.shape[0]+1):
        ax_NIRCam = plt.subplot2grid(grid_size, (1, f-1), colspan=1, rowspan=1)
        if f==1:
            ax_NIRCam.set_ylabel("Titan", fontproperties = font)
            ax_NIRCam.text(-6,43,'c', font = font2)
        ax_NIRCam.set_xticks([])
        ax_NIRCam.set_yticks([])
        ax_NIRCam.imshow(NIRCam_Titan[f-1], origin='lower')
        ax_NIRCam.text(0.1,1.2,NIRCam_filters[f-1], bbox={'facecolor': 'white', 'pad': 2}, size = text_size)

    fig.subplots_adjust(hspace=0.195, wspace=0.14, left=0.03, bottom=0.007, right=0.998, top=0.91)
    # plt.subplot_tool()
    plt.savefig('Figures/Paper_1/Figure_NIRCam.eps')


def figure_NIRSpec(Orion_fusion_name, Titan_fusion_name):

    fig = plt.figure()
    font = matplotlib.font_manager.FontProperties(family='Helvetica', size=16)
    fig.set_size_inches((9.5,4.2))
    grid_size = (2,5)
    linewidth = 0.3
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)

    NIRSpec = fits.getdata("Fusion_results/" + Orion_fusion_name + "prepro_NIRSpec.fits")
    waves = np.load("Vectors_Save/" + Orion_fusion_name + "/waves.npy")

    ax_fusion_lambda1 = plt.subplot2grid(grid_size, (0,0), colspan=1, rowspan=1)
    ax_fusion_lambda1.set_xticks([])
    ax_fusion_lambda1.set_yticks([])
    ax_fusion_lambda1.set_ylabel("d203-506", fontproperties = font)
    ax_fusion_lambda2 = plt.subplot2grid(grid_size, (0,1), colspan=1, rowspan=1)
    ax_fusion_lambda2.set_xticks([])
    ax_fusion_lambda2.set_yticks([])

    ax_spectra1 = plt.subplot2grid(grid_size, (0,2), colspan=3, rowspan=1)

    lambda_1 = 813
    lambda_2 = 1167

    fusion_lambda1 = NIRSpec[lambda_1]
    fusion_lambda2 = NIRSpec[lambda_2]
    fusion_spectra1 = NIRSpec[:,5,6]
    fusion_spectra2 = NIRSpec[:,3,5]
    max_sp1 = np.max(fusion_spectra1)
    min_sp2 = np.min(fusion_spectra2)

    font2 = matplotlib.font_manager.FontProperties(weight='bold', size=20)
    ax_fusion_lambda1.text(-2,10,'a', font = font2)
    ax_fusion_lambda1.imshow(fusion_lambda1, origin = 'lower')
    ax_fusion_lambda1.text(-0.32,0.05,'$\lambda$ = ' + str(round(waves[lambda_1],3)) + ' $\mu m$', bbox={'facecolor': 'white', 'pad': 2}, size = 12)
    ax_fusion_lambda2.imshow(fusion_lambda2, origin = 'lower')
    ax_fusion_lambda2.text(-0.3,0.05,'$\lambda$ = ' + str(round(waves[lambda_2],3)) + ' $\mu m$', bbox={'facecolor': 'white', 'pad': 2}, size = 12)
    ax_fusion_lambda1.plot(5,3, 'rs', markersize=8)
    ax_fusion_lambda1.plot(6,5, 'ks', markersize=8)
    ax_fusion_lambda2.plot(5,3, 'rs', markersize=8)
    ax_fusion_lambda2.plot(6,5, 'ks', markersize=8)
    ax_spectra1.plot(waves, fusion_spectra1, linewidth=linewidth, color = 'k')
    ax_spectra1.plot(waves, fusion_spectra2, linewidth=linewidth, color = 'r')
    ax_spectra1.axvline(x=waves[lambda_1], ls=':')
    ax_spectra1.axvline(x=waves[lambda_2], ls=':')
    ax_spectra1.set_xlim((waves[0]-0.005, waves[-1]+0.005))
    ax_spectra1.set_ylim(min_sp2, max_sp1)
    ax_spectra1.yaxis.set_label_position("right")
    ax_spectra1.yaxis.set_ticks_position('right')
    ax_spectra1.set_ylabel('Intensity (MJy/sr)', fontproperties = font, rotation=270, labelpad=20)
    ax_spectra1.set_yscale('log')
    ax_spectra1.text(1.625,28000,'b', font = font2)

    ax_fusionT_lambda1 = plt.subplot2grid(grid_size, (1,0), colspan=1, rowspan=1)
    ax_fusionT_lambda1.set_xticks([])
    ax_fusionT_lambda1.set_yticks([])
    ax_fusionT_lambda1.set_ylabel("Titan", fontproperties = font)
    ax_fusionT_lambda2 = plt.subplot2grid(grid_size, (1,1), colspan=1, rowspan=1)
    ax_fusionT_lambda2.set_xticks([])
    ax_fusionT_lambda2.set_yticks([])

    ax_spectraT1 = plt.subplot2grid(grid_size, (1,2), colspan=3, rowspan=1)

    NIRSpec = fits.getdata("Fusion_results/" + Titan_fusion_name + "prepro_NIRSpec.fits")
    waves = np.load("Vectors_Save/" + Titan_fusion_name + "/waves.npy")
    lambda_1 = 813
    lambda_2 = 1034
    fusion_lambda1 = NIRSpec[lambda_1]
    fusion_lambda2 = NIRSpec[lambda_2]
    fusion_spectra1 = NIRSpec[:,8,8]
    fusion_spectra2 = NIRSpec[:,5,5]

    ax_fusionT_lambda1.text(-2.7,14,'c', font = font2)
    ax_fusionT_lambda1.imshow(fusion_lambda1, origin = 'lower')
    ax_fusionT_lambda1.text(-0.25,0.2,'$\lambda$ = '+ str(round(waves[lambda_1],3)) + ' $\mu m$', bbox={'facecolor': 'white', 'pad': 2}, size = 12)
    ax_fusionT_lambda2.imshow(fusion_lambda2, origin = 'lower')
    ax_fusionT_lambda2.text(-0.22,0.2,'$\lambda$ = '+ str(round(waves[lambda_2],3)) + ' $\mu m$', bbox={'facecolor': 'white', 'pad': 2}, size = 12)
    ax_fusionT_lambda1.plot(5,5, 'rs', markersize=5)
    ax_fusionT_lambda1.plot(8,8, 'ks', markersize=5)
    ax_fusionT_lambda2.plot(5,5, 'rs', markersize=5)
    ax_fusionT_lambda2.plot(8,8, 'ks', markersize=5)
    ax_spectraT1.text(1.625,90000,'d', font = font2)
    ax_spectraT1.plot(waves, fusion_spectra1, linewidth=linewidth, color = 'k')
    ax_spectraT1.plot(waves, fusion_spectra2, linewidth=linewidth, color = 'r')
    ax_spectraT1.axvline(x=waves[lambda_1], ls=':')
    ax_spectraT1.axvline(x=waves[lambda_2], ls=':')
    ax_spectraT1.set_xlim((waves[0]-0.005, waves[-1]+0.005))
    ax_spectraT1.yaxis.set_label_position("right")
    ax_spectraT1.yaxis.set_ticks_position('right')
    ax_spectraT1.set_ylabel('Intensity (MJy/sr)', fontproperties = font, rotation=270, labelpad=20)
    ax_spectraT1.set_yscale('log')
    ax_spectraT1.set_xlabel("Wavelength ($\mu m$)", fontproperties = font)

    plt.tight_layout()
    fig.subplots_adjust(hspace=0.221, wspace=0.167, left=0.03, bottom=0.124, right=0.926, top=0.93)
    # plt.subplot_tool()
    plt.savefig('Figures/Paper_1/Figure_NIRSpec.eps')

    return max_sp1, min_sp2


def results(Orion_fusion_name, Titan_fusion_name, max_sp1, min_sp2, mu):

    fig = plt.figure()
    font = matplotlib.font_manager.FontProperties(family='Helvetica', size=15)
    font2 = matplotlib.font_manager.FontProperties( weight='bold', size=20)
    fig.set_size_inches((9.5,4.2))
    plt.xticks(fontsize = 15)
    plt.yticks(fontsize = 15)
    linewidth = 0.3

    grid_size = (2,5)

    fusion = fits.getdata("Fusion_results/" + Orion_fusion_name + "X_0.0.fits")
    waves = np.load("Vectors_Save/" + Orion_fusion_name + "/waves.npy")

    ax_fusion_lambda1 = plt.subplot2grid(grid_size, (0,0), colspan=1, rowspan=1)
    ax_fusion_lambda1.set_xticks([])
    ax_fusion_lambda1.set_yticks([])
    ax_fusion_lambda2 = plt.subplot2grid(grid_size, (0,1), colspan=1, rowspan=1)
    ax_fusion_lambda2.set_xticks([])
    ax_fusion_lambda2.set_yticks([])

    ax_spectra1 = plt.subplot2grid(grid_size, (0,2), colspan=3, rowspan=1)

    lambda_1 = 813
    lambda_2 = 1167

    fusion_lambda1 = fusion[lambda_1]
    fusion_lambda2 = fusion[lambda_2]
    fusion_spectra1 = fusion[:,16,19]
    fusion_spectra2 = fusion[:,10,16]

    ax_fusion_lambda1.text(-4,31,'a', font = font2)
    ax_fusion_lambda1.imshow(fusion_lambda1, origin = 'lower')
    ax_fusion_lambda1.text(0.1,1.1,'$\lambda$ = '+ str(round(waves[lambda_1],3)) + ' $\mu m$', bbox={'facecolor': 'white', 'pad': 2}, size = 12)
    ax_fusion_lambda2.imshow(fusion_lambda2, origin = 'lower')
    ax_fusion_lambda2.text(0.1,1.1,'$\lambda$ = '+ str(round(waves[lambda_2],3)) + ' $\mu m$', bbox={'facecolor': 'white', 'pad': 2}, size = 12)
    ax_fusion_lambda1.plot(16,10, 'rs', markersize=5)
    ax_fusion_lambda1.plot(19,16, 'ks', markersize=5)
    ax_fusion_lambda1.set_ylabel("d203-506", fontproperties = font)
    ax_fusion_lambda2.plot(16,10, 'rs', markersize=5)
    ax_fusion_lambda2.plot(19,16, 'ks', markersize=5)
    ax_spectra1.text(1.625,28000,'b', font = font2)
    ax_spectra1.plot(waves, fusion_spectra1, linewidth=linewidth, color = 'k')
    ax_spectra1.plot(waves, fusion_spectra2, linewidth=linewidth, color = 'r')
    ax_spectra1.axvline(x=waves[lambda_1], ls=':')
    ax_spectra1.axvline(x=waves[lambda_2], ls=':')
    ax_spectra1.set_xlim((waves[0]-0.005, waves[-1]+0.005))
    ax_spectra1.set_ylim((min_sp2, max_sp1))
    ax_spectra1.yaxis.set_label_position("right")
    ax_spectra1.yaxis.set_ticks_position('right')
    ax_spectra1.set_ylabel('Intensity (MJy/sr)', fontproperties = font, rotation=270, labelpad=20)
    ax_spectra1.set_yscale('log')

    ax_fusionT_lambda1 = plt.subplot2grid(grid_size, (1,0), colspan=1, rowspan=1)
    ax_fusionT_lambda1.set_xticks([])
    ax_fusionT_lambda1.set_yticks([])
    ax_fusionT_lambda2 = plt.subplot2grid(grid_size, (1,1), colspan=1, rowspan=1)
    ax_fusionT_lambda2.set_xticks([])
    ax_fusionT_lambda2.set_yticks([])

    ax_spectraT1 = plt.subplot2grid(grid_size, (1,2), colspan=3, rowspan=1)

    fusion = fits.getdata("Fusion_results/" + Titan_fusion_name + "X_"+str(mu)+".fits")
    spatial_slice = (slice(21,63),slice(24,66))
    waves = np.load("Vectors_Save/" + Titan_fusion_name + "/waves.npy")
    lambda_1 = 813
    lambda_2 = 1034
    fusion_lambda1 = fusion[lambda_1]
    fusion_lambda2 = fusion[lambda_2]
    fusion_spectra1 = fusion[:,24,23]
    fusion_spectra2 = fusion[:,15,15]

    ax_fusionT_lambda1.text(-5.4,43,'c', font = font2)
    ax_fusionT_lambda1.imshow(fusion_lambda1, origin = 'lower')
    ax_fusionT_lambda1.text(0.3,1.6,'$\lambda$ = '+ str(round(waves[lambda_1],3)) + ' $\mu m$', bbox={'facecolor': 'white', 'pad': 2}, size = 12)
    ax_fusionT_lambda2.imshow(fusion_lambda2, origin = 'lower')
    ax_fusionT_lambda2.text(0.3,1.6,'$\lambda$ = '+ str(round(waves[lambda_2],3)) + ' $\mu m$', bbox={'facecolor': 'white', 'pad': 2}, size = 12)
    ax_fusionT_lambda1.plot(14,14, 'rs', markersize=5)
    ax_fusionT_lambda1.plot(25,25, 'ks', markersize=5)
    ax_fusionT_lambda1.set_ylabel("Titan", fontproperties = font)
    ax_fusionT_lambda2.plot(14,14, 'rs', markersize=5)
    ax_fusionT_lambda2.plot(25,25, 'ks', markersize=5)
    ax_spectraT1.text(1.625,90000,'d', font = font2)
    ax_spectraT1.plot(waves, fusion_spectra1, linewidth=linewidth, color = 'k')
    ax_spectraT1.plot(waves, fusion_spectra2, linewidth=linewidth, color = 'r')
    ax_spectraT1.axvline(x=waves[lambda_1], ls=':')
    ax_spectraT1.axvline(x=waves[lambda_2], ls=':')
    ax_spectraT1.set_xlim((waves[0]-0.005, waves[-1]+0.005))
    ax_spectraT1.yaxis.set_label_position("right")
    ax_spectraT1.yaxis.set_ticks_position('right')
    ax_spectraT1.set_ylabel('Intensity (MJy/sr)', fontproperties = font, rotation=270, labelpad=20)
    ax_spectraT1.set_yscale('log')
    ax_spectraT1.set_xlabel("Wavelength ($\mu m$)", fontproperties = font)

    fig.subplots_adjust(hspace=0.221, wspace=0.167, left=0.03, bottom=0.124, right=0.926, top=0.93)
    # plt.subplot_tool()
    plt.savefig('Figures/Paper_1/Figure_results_.eps')


def validation_NIRCam(Orion_fusion_name, Titan_fusion_name, mu):
    fig = plt.figure()
    font = matplotlib.font_manager.FontProperties(family='Helvetica', size=17)
    font2 = matplotlib.font_manager.FontProperties( weight='bold', size=21)
    fig.set_size_inches((15,4))
    grid_size = (2,8)

    fusion = fits.getdata("Fusion_results/" + Orion_fusion_name + "X_0.0.fits")
    NIRCam = fits.getdata("Fusion_results/" + Orion_fusion_name + "prepro_NIRCam.fits")
    Lm = np.load("Vectors_Save/" + Orion_fusion_name + "/Lm.npy")

    fusion_over_Lm = np.zeros_like(NIRCam)
    for i in range(len(NIRCam)):
        fusion_over_Lm[i] = np.sum((Lm[i] * fusion.T).T, axis=0)

    ax_NIRCam_f1 = plt.subplot2grid(grid_size, (0,0), colspan=1, rowspan=1)
    ax_NIRCam_f1.set_xticks([])
    ax_NIRCam_f1.set_yticks([])
    ax_NIRCam_f1.imshow(NIRCam[0], origin = 'lower')
    ax_NIRCam_f1.set_ylabel("NIRCam", fontproperties = font)
    ax_NIRCam_f1.text(-5,31,'a', font = font2)
    ax_NIRCam_f2 = plt.subplot2grid(grid_size, (0,1), colspan=1, rowspan=1)
    ax_NIRCam_f2.set_xticks([])
    ax_NIRCam_f2.set_yticks([])
    ax_NIRCam_f2.imshow(NIRCam[1], origin = 'lower')
    ax_NIRCam_f3 = plt.subplot2grid(grid_size, (0,2), colspan=1, rowspan=1)
    ax_NIRCam_f3.set_xticks([])
    ax_NIRCam_f3.set_yticks([])
    ax_NIRCam_f3.imshow(NIRCam[2], origin = 'lower')
    ax_fusion_f1 = plt.subplot2grid(grid_size, (1,0), colspan=1, rowspan=1)
    ax_fusion_f1.set_xticks([])
    ax_fusion_f1.set_yticks([])
    ax_fusion_f1.imshow(fusion_over_Lm[0], origin = 'lower')
    ax_fusion_f1.set_ylabel("Fusion", fontproperties = font)
    ax_fusion_f1.set_xlabel("F182M", fontproperties = font)
    ax_fusion_f1.text(-5,31,'c', font = font2)
    ax_fusion_f2 = plt.subplot2grid(grid_size, (1,1), colspan=1, rowspan=1)
    ax_fusion_f2.set_xticks([])
    ax_fusion_f2.set_yticks([])
    ax_fusion_f2.imshow(fusion_over_Lm[1], origin = 'lower')
    ax_fusion_f2.set_xlabel("F187N", fontproperties = font)
    ax_fusion_f3 = plt.subplot2grid(grid_size, (1,2), colspan=1, rowspan=1)
    ax_fusion_f3.set_xticks([])
    ax_fusion_f3.set_yticks([])
    ax_fusion_f3.imshow(fusion_over_Lm[2], origin = 'lower')
    ax_fusion_f3.set_xlabel("F210M", fontproperties = font)

    fusion = fits.getdata("Fusion_results/" + Titan_fusion_name + "X_"+str(mu)+".fits")
    NIRCam = fits.getdata("Fusion_results/" + Titan_fusion_name + "prepro_NIRCam.fits")
    Lm = np.load("Vectors_Save/" + Titan_fusion_name + "/Lm.npy")
    ax_NIRCam_ft3 = plt.subplot2grid(grid_size, (0,3), colspan=1, rowspan=1)
    ax_NIRCam_ft3.set_xticks([])
    ax_NIRCam_ft3.set_yticks([])
    ax_NIRCam_ft3.imshow(NIRCam[0], origin = 'lower')
    ax_NIRCam_ft3.text(-7,43,'b', font = font2)
    ax_NIRCam_ft4 = plt.subplot2grid(grid_size, (0,4), colspan=1, rowspan=1)
    ax_NIRCam_ft4.set_xticks([])
    ax_NIRCam_ft4.set_yticks([])
    ax_NIRCam_ft4.imshow(NIRCam[1], origin = 'lower')
    ax_NIRCam_ft5 = plt.subplot2grid(grid_size, (0,5), colspan=1, rowspan=1)
    ax_NIRCam_ft5.set_xticks([])
    ax_NIRCam_ft5.set_yticks([])
    ax_NIRCam_ft5.imshow(NIRCam[2], origin = 'lower')
    ax_NIRCam_ft6 = plt.subplot2grid(grid_size, (0,6), colspan=1, rowspan=1)
    ax_NIRCam_ft6.set_xticks([])
    ax_NIRCam_ft6.set_yticks([])
    ax_NIRCam_ft6.imshow(NIRCam[3], origin = 'lower')
    ax_NIRCam_ft7 = plt.subplot2grid(grid_size, (0,7), colspan=1, rowspan=1)
    ax_NIRCam_ft7.set_xticks([])
    ax_NIRCam_ft7.set_yticks([])
    ax_NIRCam_ft7.imshow(NIRCam[4], origin = 'lower')
    ax_fusion_ft3 = plt.subplot2grid(grid_size, (1,3), colspan=1, rowspan=1)
    ax_fusion_ft3.set_xticks([])
    ax_fusion_ft3.set_yticks([])
    ax_fusion_ft3.imshow(np.sum((Lm[0] * fusion.T).T, axis=0), origin = 'lower')
    ax_fusion_ft3.set_xlabel("F182M", fontproperties = font)
    ax_fusion_ft3.text(-7,43,'d', font = font2)
    ax_fusion_ft4 = plt.subplot2grid(grid_size, (1,4), colspan=1, rowspan=1)
    ax_fusion_ft4.set_xticks([])
    ax_fusion_ft4.set_yticks([])
    ax_fusion_ft4.imshow(np.sum((Lm[1] * fusion.T).T, axis=0), origin = 'lower')
    ax_fusion_ft4.set_xlabel("F187N", fontproperties = font)
    ax_fusion_ft5 = plt.subplot2grid(grid_size, (1,5), colspan=1, rowspan=1)
    ax_fusion_ft5.set_xticks([])
    ax_fusion_ft5.set_yticks([])
    ax_fusion_ft5.imshow(np.sum((Lm[2] * fusion.T).T, axis=0), origin = 'lower')
    ax_fusion_ft5.set_xlabel("F200W", fontproperties = font)
    ax_fusion_ft6 = plt.subplot2grid(grid_size, (1,6), colspan=1, rowspan=1)
    ax_fusion_ft6.set_xticks([])
    ax_fusion_ft6.set_yticks([])
    ax_fusion_ft6.imshow(np.sum((Lm[3] * fusion.T).T, axis=0), origin = 'lower')
    ax_fusion_ft6.set_xlabel("F210M", fontproperties = font)
    ax_fusion_ft7 = plt.subplot2grid(grid_size, (1,7), colspan=1, rowspan=1)
    ax_fusion_ft7.set_xticks([])
    ax_fusion_ft7.set_yticks([])
    ax_fusion_ft7.imshow(np.sum((Lm[4] * fusion.T).T, axis=0), origin = 'lower')
    ax_fusion_ft7.set_xlabel("F212N", fontproperties = font)

    fig.subplots_adjust(hspace=0, wspace=0.2, left=0.025, bottom=0.044, right=0.996, top=0.96)
    # plt.subplot_tool()
    plt.savefig('Figures/Paper_1/Figure_val_NIRCam.eps')


def validation_NIRSpec(Orion_fusion_name, Titan_fusion_name, mu):

    fig = plt.figure()
    font = matplotlib.font_manager.FontProperties(family='Helvetica', size=17)
    font2 = matplotlib.font_manager.FontProperties( weight='bold', size=20)
    fig.set_size_inches((12,4))
    grid_size = (4,12)
    linewidth = 0.3

    fusion = fits.getdata("Fusion_results/" + Orion_fusion_name + "X_0.0.fits")
    NIRSpec = fits.getdata("Fusion_results/" + Orion_fusion_name + "prepro_NIRSpec.fits")
    waves = np.load("Vectors_Save/" + Orion_fusion_name + "/waves.npy")

    ax_spectra = plt.subplot2grid(grid_size, (0,0), colspan=6, rowspan=3)
    ax_spectra.text(1.625,28000,'a', font = font2)
    ax_spectra.set_xticks([])
    ax_spectra.set_ylabel('Intensity (MJy/sr)', fontproperties = font)
    ax_spectra.set_yscale('log')
    ax_spectra.set_xlim((waves[0]-0.005, waves[-1]+0.005))
    ax_spectra.plot(waves, np.mean(fusion, axis = (1,2)), linewidth=linewidth, color = 'b', label = 'Fusion')
    ax_spectra.plot(waves, np.mean(NIRSpec, axis = (1,2)), linewidth=linewidth, color = 'orange', label = 'NIRSpec')
    ax_spectra.legend(loc='upper left')
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    ax_residuals = plt.subplot2grid(grid_size, (3,0), colspan=6, rowspan=1)
    ax_residuals.set_xlim((waves[0]-0.005, waves[-1]+0.005))
    ax_residuals.set_xlabel("Wavelength ($\mu m$)", fontproperties = font)
    ax_residuals.set_ylabel('RE (%)', fontproperties = font)
    ax_residuals.plot(waves, 100 * (np.mean(fusion, axis = (1,2)) - np.mean(NIRSpec, axis = (1,2)))/np.mean(NIRSpec, axis = (1,2)), linewidth=linewidth, color = 'deepskyblue')
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)

    fusion = fits.getdata("Fusion_results/" + Titan_fusion_name + "X_"+ str(mu) +".fits")
    NIRSpec = fits.getdata("Fusion_results/" + Titan_fusion_name + "prepro_NIRSpec.fits")
    waves = np.load("Vectors_Save/" + Titan_fusion_name + "/waves.npy")

    ax_spectra = plt.subplot2grid(grid_size, (0,6), colspan=6, rowspan=3)
    ax_spectra.text(1.625,15500,'b', font = font2)
    ax_spectra.set_xticks([])
    ax_spectra.set_yscale('log')
    ax_spectra.set_xlim((waves[0]-0.005, waves[-1]+0.005))
    ax_spectra.plot(waves, np.mean(fusion, axis = (1,2)), linewidth=linewidth, color = 'b', label = 'Fusion')
    ax_spectra.plot(waves, np.mean(NIRSpec, axis = (1,2)), linewidth=linewidth, color = 'orange', label = 'NIRSpec')
    ax_spectra.legend(loc='upper left')
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    ax_residuals = plt.subplot2grid(grid_size, (3,6), colspan=6, rowspan=1)
    ax_residuals.set_xlim((waves[0]-0.005, waves[-1]+0.005))
    residuals = 100 * (np.mean(fusion, axis = (1,2)) - np.mean(NIRSpec, axis = (1,2)))/np.mean(NIRSpec, axis = (1,2))
    ax_residuals.plot(waves, residuals, linewidth=linewidth, color = 'deepskyblue')
    ax_residuals.set_xlabel("Wavelength ($\mu m$)", fontproperties = font)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)

    plt.tight_layout()
    plt.savefig('Figures/Paper_1/Figure_val_NIRSpec.eps')


def validation_3_filters(Titan_3_f_fusion):

    fig = plt.figure()
    font = matplotlib.font_manager.FontProperties(family='Helvetica', size=17)
    fig.set_size_inches((12,4))
    font2 = matplotlib.font_manager.FontProperties(weight='bold', size=21)
    fusion = fits.getdata("Fusion_results/" + Titan_3_f_fusion + "X_0.0.fits")
    NIRCam = fits.getdata("Fusion_results/" + Titan_3_f_fusion + "prepro_NIRCam.fits")
    NIRSpec = fits.getdata("Fusion_results/" + Titan_3_f_fusion + "prepro_NIRSpec.fits")
    Lm = np.load("Vectors_Save/" + Titan_3_f_fusion + "/Lm.npy")
    waves = np.load("Vectors_Save/" + Titan_3_f_fusion + "/waves.npy")
    
    grid_size = (4,12)

    ax_NIRCam_f1 = plt.subplot2grid(grid_size, (0,0), colspan=2, rowspan=2)
    ax_NIRCam_f1.set_xticks([])
    ax_NIRCam_f1.set_yticks([])
    ax_NIRCam_f1.imshow(NIRCam[0], origin = 'lower')
    ax_NIRCam_f1.set_ylabel("NIRCam", fontproperties = font)
    ax_NIRCam_f1.text(-7,43,'a', font = font2)
    ax_NIRCam_f2 = plt.subplot2grid(grid_size, (0,2), colspan=2, rowspan=2)
    ax_NIRCam_f2.set_xticks([])
    ax_NIRCam_f2.set_yticks([])
    ax_NIRCam_f2.imshow(NIRCam[1], origin = 'lower')
    ax_NIRCam_f3 = plt.subplot2grid(grid_size, (0,4), colspan=2, rowspan=2)
    ax_NIRCam_f3.set_xticks([])
    ax_NIRCam_f3.set_yticks([])
    ax_NIRCam_f3.imshow(NIRCam[2], origin = 'lower')
    ax_fusion_f1 = plt.subplot2grid(grid_size, (2,0), colspan=2, rowspan=2)
    ax_fusion_f1.set_xticks([])
    ax_fusion_f1.set_yticks([])
    ax_fusion_f1.imshow(np.sum((Lm[0] * fusion.T).T, axis=0), origin = 'lower')
    ax_fusion_f1.set_ylabel("Fusion", fontproperties = font)
    ax_fusion_f1.set_xlabel("F182M", fontproperties = font)
    ax_fusion_f2 = plt.subplot2grid(grid_size, (2,2), colspan=2, rowspan=2)
    ax_fusion_f2.set_xticks([])
    ax_fusion_f2.set_yticks([])
    ax_fusion_f2.imshow(np.sum((Lm[1] * fusion.T).T, axis=0), origin = 'lower')
    ax_fusion_f2.set_xlabel("F200W", fontproperties = font)
    ax_fusion_f3 = plt.subplot2grid(grid_size, (2,4), colspan=2, rowspan=2)
    ax_fusion_f3.set_xticks([])
    ax_fusion_f3.set_yticks([])
    ax_fusion_f3.imshow(np.sum((Lm[2] * fusion.T).T, axis=0), origin = 'lower')
    ax_fusion_f3.set_xlabel("F210M", fontproperties = font)

    ax_spectra = plt.subplot2grid(grid_size, (0,6), colspan=6, rowspan=3)
    ax_spectra.set_xticks([])
    ax_spectra.yaxis.set_label_position("right")
    ax_spectra.yaxis.set_ticks_position('right')
    ax_spectra.set_ylabel('Intensity (MJy/sr)', fontproperties = font, rotation=270, labelpad=20)
    ax_spectra.set_yscale('log')
    ax_spectra.set_xlim((waves[0]-0.005, waves[-1]+0.005))
    ax_spectra.text(1.625,15500,'b', font = font2)
    ax_spectra.plot(waves, np.mean(fusion, axis = (1,2)), linewidth=0.2, color = 'b', label = 'Fusion')
    ax_spectra.plot(waves, np.mean(NIRSpec, axis = (1,2)), linewidth=0.2, color = 'orange', label = 'NIRSpec')
    ax_spectra.legend(loc='lower left')
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    ax_residuals = plt.subplot2grid(grid_size, (3,6), colspan=6, rowspan=1)
    ax_residuals.yaxis.set_label_position("right")
    ax_residuals.yaxis.set_ticks_position('right')
    ax_residuals.set_xlim((waves[0]-0.005, waves[-1]+0.005))
    ax_residuals.set_ylabel('Percent', fontproperties = font, rotation=270, labelpad=20)
    ax_residuals.plot(waves, 100 * (np.mean(fusion, axis = (1,2))-np.mean(NIRSpec, axis = (1,2)))/np.mean(NIRSpec, axis = (1,2)), linewidth=0.2, color = 'deepskyblue')
    ax_residuals.set_xlabel("Wavelength ($\mu m$)", fontproperties = font)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.legend()

    # plt.subplot_tool()
    fig.subplots_adjust(hspace=0.308, wspace=0.443, left=0.024, bottom=0.148, right=0.926, top=0.91)
    fig.subplots_adjust(hspace=0.15)
    plt.savefig('Figures/Paper_1/Figure_annex_marginals.eps')


def validation_unseen_filters(Titan_3_f_fusion, Titan_fusion_name):

    fig = plt.figure()
    font = matplotlib.font_manager.FontProperties(family='Helvetica', size=17)
    fig.set_size_inches((4,4))
    fusion = fits.getdata("Fusion_results/" + Titan_3_f_fusion + "X_0.0.fits")
    NIRCam = fits.getdata("Fusion_results/" + Titan_fusion_name + "prepro_NIRCam.fits")
    NIRCam_by_fusion = NIRCam_forward_model(fusion,42, 0, ['Point_Spread_Functions/Wm_F170LP.fits',
             'Point_Spread_Functions/Wh_F170LP.fits'], slice(0,1616), 3)
    Lm_cross_calibrated = np.load("Vectors_Save/" + Titan_fusion_name + "/Lm.npy")
    waves = np.load("Vectors_Save/" + Titan_3_f_fusion + "/waves.npy")

    grid_size = (2,2)

    ax_NIRCam_f1 = plt.subplot2grid(grid_size, (0,0), colspan=1, rowspan=1)
    ax_NIRCam_f1.set_xticks([])
    ax_NIRCam_f1.set_yticks([])
    ax_NIRCam_f1.imshow(NIRCam[1], origin = 'lower')
    ax_NIRCam_f1.set_ylabel("NIRCam", fontproperties = font)
    ax_NIRCam_f2 = plt.subplot2grid(grid_size, (0,1), colspan=1, rowspan=1)
    ax_NIRCam_f2.set_xticks([])
    ax_NIRCam_f2.set_yticks([])
    ax_NIRCam_f2.imshow(NIRCam[4], origin = 'lower')
    ax_fusion_f1 = plt.subplot2grid(grid_size, (1,0), colspan=1, rowspan=1)
    ax_fusion_f1.set_xticks([])
    ax_fusion_f1.set_yticks([])
    ax_fusion_f1.imshow(np.sum((Lm_cross_calibrated[1] * fusion.T).T, axis=0), origin = 'lower')
    ax_fusion_f1.set_ylabel("Fusion", fontproperties = font)
    ax_fusion_f1.set_xlabel("F187N", fontproperties = font)
    ax_fusion_f2 = plt.subplot2grid(grid_size, (1,1), colspan=1, rowspan=1)
    ax_fusion_f2.set_xticks([])
    ax_fusion_f2.set_yticks([])
    ax_fusion_f2.imshow(np.sum((Lm_cross_calibrated[4] * fusion.T).T, axis=0), origin = 'lower')
    ax_fusion_f2.set_xlabel("F212N", fontproperties = font)

    print('PSNR: ' + str(psnr(NIRCam[1]/np.max(NIRCam[1]),np.mean(len(waves)*(Lm_cross_calibrated[1] * NIRCam_by_fusion.T).T, axis=0)/np.max(NIRCam[1]), data_range = 1)))

    print('PSNR: ' + str(psnr(NIRCam[4]/np.max(NIRCam[4]),np.mean(len(waves)*(Lm_cross_calibrated[4] * NIRCam_by_fusion.T).T, axis=0)/np.max(NIRCam[4]), data_range = 1)))

    print('SSIM: ' + str(ssim(NIRCam[1]/np.max(NIRCam[1]),np.mean(len(waves)*(Lm_cross_calibrated[1] * NIRCam_by_fusion.T).T, axis=0)/np.max(NIRCam[1]), data_range = 1)))

    print('SSIM: ' + str(ssim(NIRCam[4]/np.max(NIRCam[4]),np.mean(len(waves)*(Lm_cross_calibrated[4] * NIRCam_by_fusion.T).T, axis=0)/np.max(NIRCam[4]), data_range = 1)))

    # plt.subplot_tool()
    fig.subplots_adjust(hspace=0.13, wspace=0, left=0.062, bottom=0.065, right=1, top=0.948)
    plt.savefig('Figures/Paper_1/Figure_unseen_filters.eps')


def filters_for_fusion():

    NIRCam_sw_modules = ["B1","B2","B3","B4"]
    NIRCam_lw_module = "B5"
    NIRCam_sw_filters = ["115W","162M","164N","182M","187N","200W","210M","212N"]
    NIRCam_lw_filters = ["300M","323N","335M","356W","360M","460M","466N","470N"]
    NIRCam_mid_waves = [1.12,1.62,1.64,1.82,1.87,2,2.1,2.12,3,3.23,3.35,3.56,3.6,4.6,4.66,4.7]
    NIRCam_filters = NIRCam_sw_filters + NIRCam_lw_filters
    NIRCam_W_filters = [0,5,11]
    NIRCam_M_filters = [1,3,6,8,10,12,13]
    NIRCam_N_filters = [2,4,7,9,14,15]
    NIRSpec_filters_dispersers = ["070LP_G140H", "100LP_G140H", "170LP_G235H", "290LP_G395H"]
    NIRSpec_mid_waves = [1,1.4,2.4,4]
    NIRCam_averaged_throughputs = []
    N = 20000
    waves = np.linspace(0.8,5.2,N)
    cmap = plt.get_cmap('turbo')
    colors = cmap(np.linspace(0.1, 0.95, N))*0.8

    def wave_to_pos(wave):
        pos = 0
        while waves[pos] < wave:
            pos += 1
        return pos

    def zeros_on_gaps(X):
        gaps = [[1.4078,1.4858],[2.36067,2.49153],[3.98276,4.20323]]
        X[wave_to_pos(gaps[0][0]):wave_to_pos(gaps[0][1])] = np.zeros_like(X[wave_to_pos(gaps[0][0]):wave_to_pos(gaps[0][1])])
        X[wave_to_pos(gaps[1][0]):wave_to_pos(gaps[1][1])] = np.zeros_like(X[wave_to_pos(gaps[1][0]):wave_to_pos(gaps[1][1])])
        X[wave_to_pos(gaps[2][0]):wave_to_pos(gaps[2][1])] = np.zeros_like(X[wave_to_pos(gaps[2][0]):wave_to_pos(gaps[2][1])])
        return X

    for f in NIRCam_sw_filters:
        NIRCam_averaged_throughput = np.zeros(N)
        for m in NIRCam_sw_modules:
            NIRCam_throughput = file_manager.read_throughput("Throughputs/NIRCam/NRC"+m+"_F"+f+"_system_throughput.txt")
            i = 0
            while(waves[i] < NIRCam_throughput[0][0]):
                i += 1
            j = -1
            while(waves[j] > NIRCam_throughput[0][-1]):
                j -= 1
            NIRCam_averaged_throughput[i:j] += scipy.interpolate.interp1d(NIRCam_throughput[0], NIRCam_throughput[1])(waves[i:j])

        NIRCam_averaged_throughputs.append(zeros_on_gaps(NIRCam_averaged_throughput/len(NIRCam_sw_modules)))
    
    for f in NIRCam_lw_filters:
        NIRCam_throughput = file_manager.read_throughput("Throughputs/NIRCam/NRCB5_F"+f+"_system_throughput.txt")
        i = 0
        while(waves[i] < NIRCam_throughput[0][0]):
            i += 1
        j = -1
        while(waves[j] > NIRCam_throughput[0][-1]):
            j -= 1
        NIRCam_averaged_throughput = np.zeros(N)
        NIRCam_averaged_throughput[i:j] = scipy.interpolate.interp1d(NIRCam_throughput[0], NIRCam_throughput[1])(waves[i:j])
        NIRCam_averaged_throughputs.append(zeros_on_gaps(NIRCam_averaged_throughput))

    NIRSpec_throughputs = []
    for fd in NIRSpec_filters_dispersers:
        NIRSpec_throughput = fits.getdata("Throughputs/NIRSpec/In_Situ/comm_PCE_F"+fd+"_IFU.fits")
        NIRSpec_throughput = np.array([np.array(i) for i in NIRSpec_throughput])
        i = 0
        while(waves[i] < 1e6 * NIRSpec_throughput[0][0]):
            i += 1
        j = -1
        while(waves[j] > 1e6 * NIRSpec_throughput[-1][0]):
            j -= 1
        NIRSpec_throughput_z = np.zeros(N)
        NIRSpec_throughput_z[i:j] = scipy.interpolate.interp1d(1e6 * NIRSpec_throughput[:,0], NIRSpec_throughput[:,1])(waves[i:j])
        # print(zeros_on_gaps(NIRSpec_throughput_z))
        NIRSpec_throughputs.append(zeros_on_gaps(NIRSpec_throughput_z))

    depths = [3,1.9,0.85,0]
    alpha = 0.9
    fig = plt.figure()
    fig.set_size_inches((9.5,4.2))
    ax = fig.add_subplot(projection='3d')

    gaps = [[1.4078,1.4858],[2.36067,2.49153],[3.98276,4.20323]]
    for [a,b] in gaps:
        ax.plot([a,a],[depths[0],depths[-1]],zs=0, linestyle='dashed', color='black', linewidth=1, alpha = 0.5)
        ax.plot([b,b],[depths[0],depths[-1]],zs=0, linestyle='dashed', color='black', linewidth=1, alpha = 0.5)

    s = 0
    for f in NIRSpec_throughputs:
        ax.plot_surface(waves, depths[0], np.array([f*0,f]), ccount = 1, alpha=alpha, color=colors[wave_to_pos(NIRSpec_mid_waves[s])])
        s +=1

    for i in NIRCam_W_filters:
        ax.plot_surface(waves, depths[1], np.array([NIRCam_averaged_throughputs[i]*0,NIRCam_averaged_throughputs[i]]), ccount = 1, alpha=alpha, color=colors[wave_to_pos(NIRCam_mid_waves[i])])

    for i in NIRCam_M_filters:
        ax.plot_surface(waves, depths[2], np.array([NIRCam_averaged_throughputs[i]*0,NIRCam_averaged_throughputs[i]]), ccount = 1, alpha=alpha, color=colors[wave_to_pos(NIRCam_mid_waves[i])])

    for i in NIRCam_N_filters:
        ax.plot_surface(waves, depths[3], np.array([NIRCam_averaged_throughputs[i]*0,NIRCam_averaged_throughputs[i]]), ccount = 1, alpha=alpha, color=colors[wave_to_pos(NIRCam_mid_waves[i])])

    x_scale=2.5
    y_scale=2
    z_scale=1

    scale=np.diag([x_scale, y_scale, z_scale, 1.0])
    scale=scale*(1.0/scale.max())
    scale[3,3]=1.0

    def short_proj():
        return np.dot(Axes3D.get_proj(ax), scale)

    ytickslabels = ["NIRSpec","NIRCam W", "NIRCam M","NIRCam N"]
    depths = np.array([3,1.9,0.85,0])
    ax.set_yticks(depths)
    ax.set_yticklabels(ytickslabels, rotation=17)
    
    ax.tick_params(top = True, axis='y', pad=22)
    ax.set_xlabel("Wavelength ($\mu m$)")
    ax.set_zlabel("Throughput")
    ax.get_proj=short_proj
    ax.view_init(elev=47, azim=-112)
    ax.grid(False)
    
    fig.subplots_adjust(hspace=0, wspace=0, left=0, bottom=0, right=1, top=1)
    plt.savefig('Figures/Paper_1/Figure_throughputs.eps')


def table_fusion_ranges():
    NIRSpec_filters_dispersers = ["070LP_G140H", "100LP_G140H", "170LP_G235H", "290LP_G395H"]
    NIRSpec_throughputs = []
    for fd in NIRSpec_filters_dispersers:
        NIRSpec_throughput = fits.getdata("Throughputs/NIRSpec/In_Situ/comm_PCE_F"+fd+"_IFU.fits")
        NIRSpec_throughput = np.array([np.array(i) for i in NIRSpec_throughput])
        NIRSpec_throughputs.append(NIRSpec_throughput)
        print(fd, 1e6 *NIRSpec_throughput[0][0], 1e6 *NIRSpec_throughput[-1][0])


def table_PSNR(fusion_name, mu):

    NIRCam = fits.getdata("Fusion_results/" + fusion_name + "prepro_NIRCam.fits")
    fusion = fits.getdata("Fusion_results/" + fusion_name + "X_"+str(mu)+".fits")
    waves = np.load("Vectors_Save/" + fusion_name + "/waves.npy")
    NIRCam_by_fusion = NIRCam_forward_model(fusion,30, 0, ['Point_Spread_Functions/Wm_F170LP.fits',
             'Point_Spread_Functions/Wh_F170LP.fits'], slice(0,1616), 3)
    Lm_cross_calibrated = np.load("Vectors_Save/" + fusion_name + "/Lm.npy")

    for i in range(NIRCam.shape[0]):
        
        print('PSNR: ' + str(psnr(NIRCam[i]/np.max(NIRCam[i]),np.mean(len(waves)*(Lm_cross_calibrated[i] * NIRCam_by_fusion.T).T, axis=0)/np.max(NIRCam[i]), data_range = 1)))
        print('SSIM: ' + str(ssim(NIRCam[i]/np.max(NIRCam[i]),np.mean(len(waves)*(Lm_cross_calibrated[i] * NIRCam_by_fusion.T).T, axis=0)/np.max(NIRCam[i]), data_range = 1)))
    

def NIRSpec_forward_model(X, sym_pad_size, zero_pad_size, psf_names, wave_slice, ratio):

    X_fft = np.fft.fft2(fusion.add_padding(X, sym_pad_size, zero_pad_size), norm = 'ortho')
    
    NIRSpec_psf = fits.getdata(psf_names[1])
    NIRSpec_psf = NIRSpec_psf[wave_slice] * (1 / np.reshape(np.sum(NIRSpec_psf[wave_slice], axis=(1,2)), (X.shape[0],1,1))) # normalize the sum of spreading pattern to 1
    NIRSpec_psf = np.pad(NIRSpec_psf, ((0,0),
    (int((X_fft.shape[1] - NIRSpec_psf.shape[1])/2),int((X_fft.shape[1] - NIRSpec_psf.shape[1])/2)), (int((X_fft.shape[2] - NIRSpec_psf.shape[2])/2),int((X_fft.shape[2] - NIRSpec_psf.shape[2])/2))), mode = 'constant', constant_values = 0)
    NIRSpec_otf = np.fft.fftshift(NIRSpec_psf)
    NIRSpec_otf = np.fft.fft2(NIRSpec_otf)

    X_observed_by_NIRSpec = np.real(np.fft.ifft2(X_fft * NIRSpec_otf, norm = 'ortho'))[:,sym_pad_size+zero_pad_size : -sym_pad_size-zero_pad_size, sym_pad_size+zero_pad_size : -sym_pad_size-zero_pad_size]

    return X_observed_by_NIRSpec


def NIRCam_forward_model(X, sym_pad_size, zero_pad_size, psf_names, wave_slice, ratio):

    X_fft = np.fft.fft2(fusion.add_padding(X, sym_pad_size, zero_pad_size), norm = 'ortho')
    
    NIRCam_psf = fits.getdata(psf_names[0])
    NIRCam_psf = NIRCam_psf[wave_slice] * (1 / np.reshape(np.sum(NIRCam_psf[wave_slice], axis=(1,2)), (X.shape[0],1,1))) # normalize the sum of spreading pattern to 1
    NIRCam_psf = np.pad(NIRCam_psf, ((0,0),
    (int((X_fft.shape[1] - NIRCam_psf.shape[1])/2),int((X_fft.shape[1] - NIRCam_psf.shape[1])/2)), (int((X_fft.shape[2] - NIRCam_psf.shape[2])/2),int((X_fft.shape[2] - NIRCam_psf.shape[2])/2))), mode = 'constant', constant_values = 0)
    NIRCam_otf = np.fft.fftshift(NIRCam_psf)
    NIRCam_otf = np.fft.fft2(NIRCam_otf)

    X_observed_by_NIRCam = np.real(np.fft.ifft2(X_fft * NIRCam_otf, norm = 'ortho'))[:,sym_pad_size+zero_pad_size : -sym_pad_size-zero_pad_size, sym_pad_size+zero_pad_size : -sym_pad_size-zero_pad_size]

    return X_observed_by_NIRCam


def gif_fusion(fusion_name):

    Fusion = fits.getdata("Fusion_results/" + fusion_name + "X_0.0.fits")
    NIRCam = fits.getdata("Fusion_results/" + fusion_name + "prepro_NIRCam.fits")
    NIRSpec = fits.getdata("Fusion_results/" + fusion_name + "prepro_NIRSpec.fits")
    waves = np.load("Vectors_Save/" + fusion_name + "/waves.npy")
    Lm = np.load("Vectors_Save/" + fusion_name + "/Lm.npy")

    NIRCam_clipped = NIRCam
    for f in range(0,Lm.shape[0]):
        NIRCam_clipped[f] = NIRCam[f] / np.max(NIRCam[f]) 

    NIRCam2 = np.zeros((NIRSpec.shape[0], NIRCam.shape[1], NIRCam.shape[2]))
    for i in range(NIRSpec.shape[0]):
        for f in range(Lm.shape[0]):
            NIRCam2[i] += Lm[f,i] * NIRCam_clipped[f]

    fig, (ax1,ax2,ax3) = plt.subplots(1, 3)

    prevalent_NIRCam = np.zeros_like(waves)
    for i in range(len(waves)):
        fopt = 0
        for f in range(1,Lm.shape[0]):
            if Lm[f,i] > Lm[fopt,i]:
                fopt = f
        prevalent_NIRCam[i] = int(fopt)

    r = 100
    im = ax1.imshow(NIRSpec[r], animated=True, origin = 'lower', vmin=0, vmax=np.max(NIRSpec))
    im2 = ax2.imshow(Fusion[r], animated=True, origin = 'lower')
    im3 = ax3.imshow(NIRCam_clipped[0], animated=True, origin = 'lower')
    text = ax2.text(4.2, -10, '$\lambda = $'+str(np.round(waves[r],3))+' $\mu m$', fontsize=12, weight='bold')

    font = matplotlib.font_manager.FontProperties(family='Helvetica', size=17)
    ax1.set_xticks([])
    ax1.set_yticks([])
    ax1.set_xlabel('NIRSpec', fontproperties = font)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_xlabel('Fusion', fontproperties = font)
    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_xlabel('NIRCam', fontproperties = font)

    a = 10
    def update(i):
        
        im.set_array(NIRSpec[a*i+r])
        im.set_clim(vmin=np.min(NIRSpec[a*i+r]), vmax= np.max(NIRSpec[a*i+r]))
        im2.set_array(Fusion[a*i+r])
        im2.set_clim(vmin=np.min(Fusion[a*i+r]), vmax= np.max(Fusion[a*i+r]))
        im3.set_array(NIRCam_clipped[int(prevalent_NIRCam[a*i+r])])
        text.set_text('$\lambda = $'+str(np.round(waves[a*i+r],3))+' $\mu m$') 
        plt.pause(0.01)

        return im, im2, im3

    animation_fig = animation.FuncAnimation(fig, update, frames=int((NIRSpec.shape[0]-r)/a), interval=400, blit=True, repeat = False,)
    # plt.subplot_tool()
    fig.subplots_adjust(hspace=0.26, wspace=0.23, left=0.02, bottom=0, right=0.98, top=1)
    plt.show()
    writermp4 = animation.FFMpegWriter(fps=2) 
    animation_fig.save("Figures/Others/"+fusion_name+"2.mp4", writer=writermp4)


# def fusion_principle_Titan_final(Titan_fusion_name, mu):
    # img_fusion_1 = fits.getdata("Fusion_results/" + Titan_fusion_name + "X_"+str(mu)+".fits")[1000][3:-6,3:-6]
    # img+= 0
    # img_fusion_2 = fits.getdata("Fusion_results/" + Titan_fusion_name + "prepro_NIRCam.fits")[3][3:-6,3:-6]
    # img_fusion_2 = fits.getdata("Fusion_results/" + Titan_fusion_name + "X_"+str(mu)+".fits")[812][3:-6,3:-6]
    # img_fusion_3 = fits.getdata("Fusion_results/" + Titan_fusion_name + "X_"+str(mu)+".fits")[182][3:-6,3:-6]
    # img_fusion = np.array([img_fusion_1, img_fusion_2, img_fusion_3]).T
    # img_fusion = np.array(img_fusion)/np.max(np.array(img_fusion))

    # cmap = plt.get_cmap('turbo')
    # nb_waves = 13
    # colors = cmap(np.linspace(0.2, 0.9, nb_waves))
    # coeff_lumi = 3
    #2*colors[2,:1],colors[6,:1],colors[10,:1]
    # img_fusion = img_fusion * np.array([0.5,0.5,0.5]).T * coeff_lumi
    # img_fusion = np.array(img_fusion)/np.max(np.array(img_fusion))

    # import skimage.io
    # skimage.io.imsave('img_fusion.png',img_fusion)
    # from PIL import Image
    # img = Image.fromarray((img_fusion_1* 255).astype(np.uint8))
    # img.save('img_fusion.png')
    # plt.imsave(img_fusion,'img_fusion_plt.png')
    # img_fusion_NIRSpec = skimage.transform.downscale_local_mean(img_fusion_1, (3,3))
    # img_fusion_1 = skimage.transform.resize(img_fusion_NIRSpec, img_fusion_1.shape, order=0)
    # plt.imshow(img_fusion_2, origin = 'lower', cmap='gray')
    # plt.show()


def paper_figures():

    Orion_fusion_name = "1288_d203_506_F170LP_166_230um_3_filters_"
    Titan_fusion_name = "1251_Titan_F170LP_166_230um_5_filters_"
    Titan_3_f_fusion = "1251_Titan_F170LP_166_230um_3_filters_validation_"
    mu = 0.0
    # Main text
    fusion_principle()
    figure_NIRCam(Orion_fusion_name, Titan_fusion_name)
    max_sp1, min_sp2 = figure_NIRSpec(Orion_fusion_name, Titan_fusion_name)
    results(Orion_fusion_name, Titan_fusion_name, max_sp1, min_sp2, mu)
    validation_NIRCam(Orion_fusion_name, Titan_fusion_name, mu)
    validation_NIRSpec(Orion_fusion_name, Titan_fusion_name, mu)
    # Annexes
    filters_for_fusion()
    table_fusion_ranges()
    table_PSNR(Orion_fusion_name, mu)
    table_PSNR(Titan_fusion_name, mu)
    validation_3_filters(Titan_3_f_fusion)
    validation_unseen_filters(Titan_3_f_fusion, Titan_fusion_name)
    plt.show()
