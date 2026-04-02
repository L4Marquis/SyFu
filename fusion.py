# Code by Landry Marquis based on https://github.com/cguilloteau/Fast-fusion-of-astronomical-images
#
# If you use this code, please cite the paper: 
# Marquis L. et al., Fusion of JWST data: Demonstrating practical feasibility,
# Astronomy & Astrophysics, volume 708, 2026

import numpy as np
import numpy.matlib
import scipy.sparse
import scipy.sparse.linalg
import scipy.linalg
import scipy.interpolate
import skimage.transform
import sklearn.utils.extmath
import file_manager
import tools
import PSF
import image
import filter
import time
from astropy.io import fits
from matplotlib import pyplot as plt


def compute_linear_system(Ym, Yh, waves, wave_slice, NIRCam_throughputs, NIRSpec_throughput, fusion_name, ratio = 3, compute_NIRCam_psf = False, compute_NIRSpec_psf = False, psf_names = ['M_.fits', 'H_.fits'], nb_comp = 4, sym_pad_size = 35, zero_pad_size = 1, epsilon = 10**(-2), calibrate_NIRCam_on_NIRSpec = True, verbose = False, debug = False):
    ''' Computes the matrices defining the linear system to solve.

    Inputs: - Ym, the preprocessed multispectral NIRCam image
            - Yh, the preprocessed hyperspectral NIRSpec image

    Outputs: - Am and bm: matrices describing Ym data fidelity
             - Ah and bh: matrices describing Yh data fidelity
             - Ar: matrix of Sobolev regularization
             - Z_fft: initial solution for the optimization
    '''

    waves = waves[wave_slice]
    nb_waves, NIRSpec_height, NIRSpec_width = Yh.shape
    nb_filters, NIRCam_height, NIRCam_width = Ym.shape

    Lm = np.zeros((nb_filters, nb_waves))

    if debug :
        print('Lm shape', Lm.shape)
    
    for k in range(nb_filters):
        NIRCam_throughput = file_manager.read_throughput(NIRCam_throughputs[k])
        i = 0
        while(waves[i] < NIRCam_throughput[0][0]):
            i += 1
        j = -1
        while(waves[j] > NIRCam_throughput[0][-1]):
            j -= 1
        Lm[k, i:j] = scipy.interpolate.interp1d(NIRCam_throughput[0], NIRCam_throughput[1])(waves[i:j])

    Lh = np.ones_like(waves)
    sigma_m_2 = np.mean(Ym)
    sigma_h_2 = np.mean(Yh)

    if verbose:
        lowpass = np.zeros_like(Yh)
        for i in range(nb_waves):
            lowpass[i,:,:] = scipy.ndimage.gaussian_filter(Yh[i,:,:], sigma = 2)
        gauss_highpass = Yh - lowpass
        sigma_gauss_h = 1.4826 * np.mean(scipy.stats.median_abs_deviation(gauss_highpass, axis=(1,2)))
        print('NIRSpec noise variance :')
        print('Poisson :', sigma_h_2)
        print('Gaussian :', sigma_gauss_h)
        lowpass = np.zeros_like(Ym)
        for i in range(nb_filters):
            lowpass[i,:,:] = scipy.ndimage.gaussian_filter(Ym[i,:,:], sigma = 2)
        gauss_highpass = Ym - lowpass
        sigma_gauss_m = 1.4826 * np.mean(scipy.stats.median_abs_deviation(gauss_highpass, axis=(1,2)))
        print('NIRCam noise variance :')
        print('Poisson :', sigma_m_2)
        print('Gaussian :',sigma_gauss_m)
        print('Ratio sigma_gaus_m / sigma_gauss_h :',sigma_gauss_m/sigma_gauss_h)

    Yh_2d = np.reshape(Yh, (nb_waves, NIRSpec_height * NIRSpec_width))
    Yh_Lh_1_2d = np.dot(np.diag(Lh**-1), Yh_2d)
    mean = np.mean(Yh_Lh_1_2d, axis = 1) # NIRSpec mean spectrum
    
    if debug:
        print('Median of mean : ', np.median(mean))
    
    U, S, V_T = scipy.linalg.svd(Yh_Lh_1_2d.T - mean, full_matrices = False)
    U, V_T = sklearn.utils.extmath.svd_flip(U, V_T, u_based_decision = False)
   
    preserved_information = sum(np.square(S[:nb_comp]) / sum(np.square(S)))
    if verbose:
        print('Variance of each components:', np.square(S[:nb_comp]))
        print('Remaining variance:', np.sum(S[nb_comp:]))
        print('Information (%) in each components:', np.square(S[:nb_comp]) / sum(np.square(S)))
        print('Total information (%):', preserved_information)
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        ax.set(xlabel = 'Number', ylabel = 'Value')
        ax.set_title("Eigenvalues log scale")
        ax.plot(np.arange(len(S)), np.log(np.square(S)))
        fig.savefig("Fusion_Results/"+fusion_name+"eigenvalues.png", bbox_inches = 'tight')
        plt.close(fig)
        fig, ax = plt.subplots(nrows = 1, ncols = 1)
        ax.set(xlabel = 'Number', ylabel = 'Value')
        ax.set_title("20 highest eigenvalues log scale")
        ax.plot(np.arange(20), np.log(np.square(S[:20])))
        fig.savefig("Fusion_Results/"+fusion_name+"eigenvalues20.png", bbox_inches = 'tight')
        plt.close(fig)
        V_test_T = V_T[:20]
        V_test = V_test_T.T
        Z_test = ((Yh_Lh_1_2d.T - mean) @ V_test).T
        Z_test = np.reshape(Z_test, (20, NIRSpec_height, NIRSpec_width))
        Z_test = skimage.transform.resize(Z_test, (20, NIRCam_height, NIRCam_width), order = 3, mode = 'symmetric')
        file_manager.save_as_fits(Z_test, "Fusion_Results/"+fusion_name+"Z_test.fits")

    S = S[:nb_comp]
    U = U[:, :nb_comp]
    V_T = V_T[:nb_comp]
    V = V_T.T
    Z = ((Yh_Lh_1_2d.T - mean) @ V).T
    Z = np.reshape(Z, (nb_comp, NIRSpec_height, NIRSpec_width))
    Z = skimage.transform.resize(Z, (nb_comp, NIRCam_height, NIRCam_width), order = 3, mode = 'symmetric')

    mean_Z = ((Yh_Lh_1_2d.T) @ V).T
    mean_Z = np.reshape(mean_Z, (nb_comp, NIRSpec_height, NIRSpec_width))
    mean_Z = skimage.transform.resize(mean_Z, (nb_comp, NIRCam_height, NIRCam_width), order = 3, mode = 'symmetric')
    mean_Z = np.mean(np.reshape(mean_Z, (nb_comp, -1)), axis = 0)
    mean_Z = np.tile(mean_Z, nb_comp)

    if verbose:
        file_manager.save_as_fits(Z, "Fusion_Results/Z_init.fits")
        X = np.reshape(((np.reshape(Z, (nb_comp, NIRCam_height * NIRCam_width)).T @ V_T) + mean).T, (nb_waves, NIRCam_height, NIRCam_width))
        file_manager.save_as_fits(X, "Fusion_Results/X_init.fits")
    if debug :
        mean_X = np.mean(X, axis = (1,2))
        f, a = plt.subplots(nrows = 1, ncols=1)
        a.plot(mean_X - mean, 'b')
        print(np.mean(mean_X / mean))
        plt.show()

    Z_pad = add_padding(Z, sym_pad_size, zero_pad_size)
    Ym_pad = add_padding(Ym, sym_pad_size, zero_pad_size)
    mean_Ym_pad = np.mean(Ym_pad, axis = (1,2))
    Ym_fft = np.fft.fft2(Ym_pad, norm = 'ortho')

    Ym_fft_shape = Ym_fft.shape
    Ym_pad_heigth = Ym_fft_shape[1]
    Ym_pad_width = Ym_fft_shape[2]
    spatial_size = Ym_pad_heigth * Ym_pad_width
    subspace_size = spatial_size * nb_comp
    if debug:
        print('Spatial size :', spatial_size)

    Yh_pad_size = int((sym_pad_size) // ratio)
    Yh_pad_shape = ((0, 0), (Yh_pad_size, Yh_pad_size), (Yh_pad_size, Yh_pad_size))
    Yh_pad = np.pad(Yh, Yh_pad_shape, 'symmetric')
    Yh_pad_shape2 = ((0, 0), (zero_pad_size, zero_pad_size), (zero_pad_size, zero_pad_size))
    Yh_pad = np.pad(Yh_pad, Yh_pad_shape2, 'constant', constant_values = 0)
    mean_Yh_pad = np.mean(Yh_pad, axis = (1,2))
    Yh_fft =  np.fft.fft2(Yh_pad, norm = 'ortho')

    Yh_pad_heigth = Yh_fft.shape[1]
    Yh_pad_width = Yh_fft.shape[2]

    Z_fft = np.fft.fft2(Z_pad, norm = 'ortho')

    mean_fft = np.zeros((nb_waves, spatial_size), complex)
    mean_fft[:, 0] = mean * np.sqrt(spatial_size) # sqrt comes from norm = ortho 

    if debug:
        print('Regarde Ym fft :', Ym_fft[:,0,0][0:4])
        print('Regarde mean Ym pad :', np.sqrt(spatial_size) * mean_Ym_pad[0:4])
        print('Regarde :', np.sqrt(spatial_size) * mean[0:4])
        print('Regarde mean_fft :', mean_fft[0:4, 0])
        print('Regarde Yh fft :', Yh_fft[:,0,0][0:4])
        print('Regarde mean Yh pad :', np.sqrt(spatial_size) * mean_Yh_pad[0:4])
        print('Regarde ici :', (Yh_fft[:,0,0] - mean_fft[:, 0])[0:4])
        print('Regarde la :', (Yh_fft[:,0,0] / mean_fft[:, 0])[0:4])
    # see https://numpy.org/doc/stable/reference/routines.fft.html

    if compute_NIRCam_psf:
        NIRCam_psf = PSF.PSF_NIRCam(psf_names[0], waves * 1e-6, NIRCam_height).compute()
    else:
        NIRCam_psf = fits.getdata(psf_names[0])
    NIRCam_psf = NIRCam_psf[wave_slice] * (1 / np.reshape(np.sum(NIRCam_psf[wave_slice], axis=(1,2)), (nb_waves,1,1))) # normalize the sum of spreading pattern to 1
    NIRCam_psf = np.pad(NIRCam_psf, ((0,0),
    (int((Ym_pad_heigth - NIRCam_psf.shape[1])/2),int((Ym_pad_heigth - NIRCam_psf.shape[1])/2)), (int((Ym_pad_width - NIRCam_psf.shape[2])/2),int((Ym_pad_width - NIRCam_psf.shape[2])/2))), mode = 'constant', constant_values = 0)
    NIRCam_otf = np.fft.fftshift(NIRCam_psf)
    NIRCam_otf = np.fft.fft2(NIRCam_otf)
    NIRCam_otf = np.array(NIRCam_otf)
    M_ = NIRCam_otf[:nb_waves, 0, 0]

    mean_m_fft = mean_fft.copy()
    mean_m_fft[:, 0] = mean_fft[:, 0] * M_

    # Normalization of the throughput

    calibration_ratios = np.zeros(nb_filters)
    for i in range(nb_filters):
        calibration_ratios[i] = np.real((Lm @ mean_m_fft)[i,0]/Ym_fft[i,0,0])/np.sum(Lm[i])

    with open("Fusion_Information/Information_"+ fusion_name +".txt", "w") as file:
        file.write("Information preserved : "+str(preserved_information)+" % \n")
        for i in range(len(calibration_ratios)) :
            file.write("Calibration ratio filter "+str(i)+" : "+str(calibration_ratios[i])+"\n")

    if calibrate_NIRCam_on_NIRSpec:
        for i in range(nb_filters):
            Lm[i] /= np.real((Lm @ mean_m_fft)[i,0]/Ym_fft[i,0,0])
            if debug:
                print('Coeff de normalisation Lm'+str(i), (Lm @ mean_m_fft)[i,0]/Ym_fft[i,0,0])
    else:
        for i in range(nb_filters):
            Lm[i] /= np.sum(Lm[i])

    Ym = (np.reshape(Ym_fft, (nb_filters, spatial_size)) - (Lm @ mean_m_fft))

    if debug:
        print('Ym_fft mean', np.mean(Ym_fft[:,0,0]))
        print('mean_m_fft mean', mean_m_fft[0,0])
        print('mean Lm mean_m_fft', np.mean(Lm @ mean_m_fft))
        print('Ym mean', Ym[0,0])
        print('Regarde Ym fft :', Ym_fft[:,0,0][0:4])
        print('Regarde Wm fft :', NIRCam_otf[:,0,0][0:4])
        print('Regarde mean_m_fft :', (mean_m_fft)[:,0][0:4])
        print('Regarde Lm :', np.sum(Lm))
        print('Regarde Lm0 :', np.sum(Lm[0]))
        print('Regarde Lm1 :', np.sum(Lm[1]))
        print('Regarde Lm mean_m_fft :', np.dot(Lm, mean_m_fft)[:,0][0:4])
        print('Regarde Lm mean_m_fft / Ym :', (np.dot(Lm, mean_m_fft)[:,0][0:4]) / (Ym_fft[:,0,0][0:4]))
        print('Regarde max Lm mean_m_fft :', np.max(Lm @ mean_m_fft))
        print('Regarde Ym mean :', Ym[:,0][0:4])

    if compute_NIRSpec_psf:
        NIRSpec_psf = PSF.PSF_NIRSpec(psf_names[1], waves * 1e-6, NIRSpec_height).compute()
    else:
        NIRSpec_psf = fits.getdata(psf_names[1])
    NIRSpec_psf = NIRSpec_psf[wave_slice] * (1 / np.reshape(np.sum(NIRSpec_psf[wave_slice], axis=(1,2)), (nb_waves,1,1))) # normalize the sum of spreading pattern to 1
    NIRSpec_psf = np.pad(NIRSpec_psf, ((0,0),
    (int((Ym_pad_heigth - NIRSpec_psf.shape[1])/2),int((Ym_pad_heigth - NIRSpec_psf.shape[1])/2)), (int((Ym_pad_width - NIRSpec_psf.shape[2])/2),int((Ym_pad_width - NIRSpec_psf.shape[2])/2))), mode = 'constant', constant_values = 0)
    NIRSpec_otf = np.fft.fftshift(NIRSpec_psf)
    NIRSpec_otf = np.fft.fft2(NIRSpec_otf)
    NIRSpec_otf = np.array(NIRSpec_otf)
    H_ = NIRSpec_otf[:nb_waves, 0, 0]

    mean_h_fft = mean_fft.copy()
    mean_h_fft[:, 0] = mean_fft[:, 0] * H_

    Yh_fft_2d = np.reshape(Yh_fft, (nb_waves, Yh_pad_heigth * Yh_pad_width))
    Yh = (Yh_fft_2d - np.dot(np.diag(Lh), tools.aliasing(mean_h_fft, (nb_waves, Ym_pad_heigth,Ym_pad_width), ratio)))

    if debug:
        print('Yh_fft mean', np.mean(Yh_fft[:,0,0]))
        print('mean_h_fft mean', mean_h_fft[0,0])
        print('Yh mean', Yh[0,0])
        print('Regarde Yh fft :', Yh_fft[:,0,0][0:4])
        print('Regarde Wh fft :', NIRSpec_otf[:,0,0][0:4])
        print('Regarde mean_h_fft S :', tools.aliasing(mean_h_fft, (nb_waves, Ym_pad_heigth,Ym_pad_width), ratio)[:,0][0:4])
        print('Regarde Lh mean_h_fft S :', (np.dot(np.diag(Lh), tools.aliasing(mean_h_fft, (nb_waves, Ym_pad_heigth,Ym_pad_width), ratio)))[:,0][0:4])
        print('Regarde Yh mean :', Yh[:,0][0:4])

    D = grad_operator_2D(Ym_pad_heigth, Ym_pad_width)

    weights = compute_weights_for_Sobolev(epsilon, Ym, D, Lm, V, V_T, Ym_fft_shape, nb_comp)

    Ym = np.reshape(Ym, np.prod(Ym.shape))
    Yh = np.reshape(Yh, np.prod(Yh.shape))
    Z_fft = np.reshape(Z_fft, np.prod(Z_fft.shape))

    row = np.arange(spatial_size)
    row = numpy.matlib.repmat(row, 1, nb_comp**2)[0]

    for i in range(nb_comp):
        row[i * subspace_size : (i+1) * subspace_size] += i * spatial_size

    column = np.arange(subspace_size)
    column = numpy.matlib.repmat(column, 1, nb_comp)[0]

    Lm_Wm_V = np.zeros((nb_filters, nb_comp, spatial_size), complex)

    otf_2d_shape = (nb_waves, spatial_size)

    NIRCam_otf_2d = np.reshape(NIRCam_otf, otf_2d_shape)
    NIRSpec_otf_2d = np.reshape(NIRSpec_otf, otf_2d_shape)
    
    # Start of A computation see Claire Guilloteau's thesis page 90
    for m in range(nb_filters):
        for i in range(nb_comp):
            sum_h = np.zeros(spatial_size, complex)
            for l in range(nb_waves):
                sum_h += NIRCam_otf_2d[l] * Lm[m, l] * V[l, i]
            Lm_Wm_V[m, i] = sum_h

    mat = np.reshape(np.reshape(np.arange(nb_comp**2), (nb_comp, nb_comp)).T, nb_comp**2)
    data = np.zeros((nb_comp**2, spatial_size), complex)

    for i in range(nb_comp):
        for j in range(nb_comp - i):
            temp = np.zeros(spatial_size, complex)
            for m in range(nb_filters):
                temp += np.conj(Lm_Wm_V[m, j+i]) * Lm_Wm_V[m, i]
            if j == 0:
                data[i * (nb_comp + 1)] = temp
            else:
                index = j + i * (nb_comp + 1)
                data[mat[index]] = np.conj(temp)
                data[index] = temp

    data = np.reshape(data, (nb_comp**2 * spatial_size))
    
    gamma_m = 1
    gamma_h = 1
    Am = 0.5 * gamma_m * scipy.sparse.coo_matrix((data, (row, column)), 
                                           (subspace_size, subspace_size),
                                           complex)
    upsilon = compute_identity_block_diagonal_matrix(Ym_pad_heigth, Ym_pad_width, ratio)
    Lh_V = np.dot(np.diag(Lh), V)

    row = numpy.matlib.repmat(upsilon.row, 1, nb_comp**2)[0]
    for i in range(nb_comp):
        row[i * subspace_size * ratio**2 : (i+1) * subspace_size * ratio**2] += i * spatial_size
    
    if debug:
        print(subspace_size*ratio**2)
        print(spatial_size*ratio**2)
        print(spatial_size)
        print(row[subspace_size * ratio**2 - 6: subspace_size * ratio**2 + 6])
        print(row[spatial_size * ratio**2 -6: spatial_size * ratio**2+6])
        print('taille upsilon', (upsilon.col).shape)

    column = np.zeros(subspace_size * ratio**2)
    for i in range(nb_comp):
        column[i * spatial_size * ratio**2 : (i+1) * spatial_size * ratio**2] = (upsilon.col + i *
                                                                                 spatial_size)
    column = numpy.matlib.repmat(column, 1, nb_comp)[0]

    if debug:
        print(column[spatial_size * ratio**2 -6: spatial_size * ratio**2+6])

    data = np.zeros((nb_comp**2 * spatial_size * ratio**2), complex)
    for i in range(nb_comp**2):
        temp = np.zeros(spatial_size * ratio**2, complex)
        for l in range(nb_waves):
            g = NIRSpec_otf_2d[l]
            g_upsilon_g = ratio**(-2) * np.conj(g[upsilon.row]) * g[upsilon.col] #  
            ii = i//nb_comp
            ij = i%nb_comp
            temp += (Lh_V[l, ii] * g_upsilon_g * Lh_V[l, ij])
        data[i * spatial_size * ratio**2 : (i+1) * spatial_size * ratio**2] = temp
    # Unoptimized function that compute twice the number of coeffs that it should, but correct a long standing bug.

    Ah = 0.5 * gamma_h * scipy.sparse.coo_matrix((data, (row, column)), 
                                           (subspace_size, subspace_size),
                                           complex)
    
    Ar = compute_A_Sobolev_regularization(nb_comp, Ym_pad_heigth, Ym_pad_width, D)

    Ym_2d = np.reshape(Ym, (nb_filters, spatial_size))
    Yh_2d = np.reshape(Yh, (nb_waves, spatial_size // ratio**2))

    bnc = np.dot(Lm.T, Ym_2d)

    for l in range(nb_waves):
        otf_band_conj = np.conj(NIRCam_otf[l])
        otf_band_conj = np.reshape(otf_band_conj, spatial_size)
        bnc[l] *= otf_band_conj
        
    bnc = np.dot(V_T, bnc)

    Yh_ = tools.aliasing_adjoint(Yh_2d, (nb_waves, Ym_pad_heigth, Ym_pad_width), ratio)

    for l in range(nb_waves):
        otf_band_conj = np.conj(NIRSpec_otf[l])
        otf_band_conj = np.reshape(otf_band_conj, spatial_size)
        Yh_[l] *= otf_band_conj

    bns = np.dot(np.dot(np.diag(Lh), V).T, Yh_)

    bm = np.reshape(- gamma_m * bnc, np.prod(bnc.shape))
    bh = np.reshape(- gamma_h * bns, np.prod(bns.shape))

    cm = 0.5 * gamma_m * np.dot(np.conj(Ym).T, Ym)
    ch = 0.5 * gamma_h * np.dot(np.conj(Yh).T, Yh)

    if debug :
        print('Lm shape 2: ', Lm.shape)

    return Ah, Am, Ar, bh, bm, ch, cm, D, gamma_h, gamma_m, Lh, Lm, mean_Z, mean, V_T, V, S, wave_slice, waves, weights, Ym_pad_heigth, Ym_pad_width, Z_fft 
    


def grad_operator_2D(heigth, width):

    Dx = np.zeros((heigth, width))
    Dy = np.zeros((heigth, width))

    Dx[0, 0] = 1
    Dx[0, 1] = -1
    Dy[0, 0] = 1
    Dy[1, 0] = -1

    Dx = np.fft.fft2(Dx)
    Dy = np.fft.fft2(Dy)

    D = (Dx, Dy)

    return D


def compute_weights_for_Sobolev(epsilon, Ym, D, Lm, P, P_tilde, Ym_shape, nb_components):
    # see Claire Guilloteau's thesis 4.2.2

    Ym_D_x = np.fft.ifft2(Ym.reshape(Ym_shape) * D[0], axes = (1, 2), norm = 'ortho')
    Ym_D_y = np.fft.ifft2(Ym.reshape(Ym_shape) * D[1], axes = (1, 2), norm = 'ortho')

    sigma_x_hat_2 = (np.linalg.norm(Ym_D_x, ord = 2, axis = 0) / np.trace(np.dot(Lm, Lm.T)), 
                     np.linalg.norm(Ym_D_y, ord = 2, axis = 0) / np.trace(np.dot(Lm, Lm.T)))

    P_tilde_P = np.diag(P_tilde @ P)

    sigma_maj_z_hat = np.zeros((2, nb_components, sigma_x_hat_2[0].shape[0], 
                                sigma_x_hat_2[0].shape[1]))

    for i in range(nb_components):
        sigma_maj_z_hat[0][i] = sigma_x_hat_2[0] * P_tilde_P[i]
        sigma_maj_z_hat[1][i] = sigma_x_hat_2[1] * P_tilde_P[i]

    weights = (0.5 * (1 / (sigma_maj_z_hat[0] + epsilon) + 1 / (sigma_maj_z_hat[1] + epsilon)), 
               0.5 * (1 / (sigma_maj_z_hat[0] + epsilon) + 1 / (sigma_maj_z_hat[1] + epsilon)))
    weights = (weights - np.min(weights)) / np.max(weights - np.min(weights))

    return weights


def compute_identity_block_diagonal_matrix(Ym_padding_heigth, Ym_padding_width, ratio):

    identity = scipy.sparse.identity(Ym_padding_width // ratio)
    identity_2 = identity.copy()

    for i in range(ratio - 1):
        identity_2 = scipy.sparse.hstack((identity_2, identity))

    identity_3 = identity_2.copy()

    for i in range(ratio - 1):
        identity_3 = scipy.sparse.vstack((identity_3, identity_2))

    identity_4 = identity_3.copy()

    for i in range(Ym_padding_heigth // ratio - 1):
        identity_4 = scipy.sparse.block_diag((identity_4, identity_3))

    identity_5 = identity_4.copy()

    for i in range(ratio - 1):
        identity_5 = scipy.sparse.hstack((identity_5, identity_4))

    identity_6 = identity_5.copy()

    for i in range(ratio - 1):
        identity_6 = scipy.sparse.vstack((identity_6, identity_5))

    return identity_6


def compute_A_Sobolev_regularization(nb_components, Ym_heigth, Ym_width, D):

    size = nb_components * Ym_heigth * Ym_width

    Dx, Dy = D

    Dx = np.reshape(Dx, np.prod(Dx.shape))
    Dy = np.reshape(Dy, np.prod(Dy.shape))
    D = np.conj(Dy) * Dy + np.conj(Dx) * Dx

    row = np.arange(0, size, 1)
    col = row.copy()
    data = np.reshape(numpy.matlib.repmat(D, nb_components, 1), size)

    Ar = scipy.sparse.coo_matrix((data, (row, col)), (size, size), complex)

    return Ar


def fusion(fusion_name, NIRCam_path, NIRCam_test_path, NIRSpec_path, Ym_height, Ym_width, Yh_height, Yh_width, center_pixel_coord, NIRSpec_angle, moving_object, angle_type, little_coord_default, dev_coeff, anom_limit, exceptions, wave_slice, NIRCam_throughputs, NIRCam_test_throughputs, NIRSpec_throughput, ratio, compute_NIRCam_psf, compute_NIRSpec_psf, psf_names, epsilon, calibrate_NIRCam_on_NIRSpec, mu, beta_m, beta_h, max_iter, nb_comp, sym_pad_size, zero_pad_size, obj_func_file_name = None, Ym_fidel_file_name = None, Yh_fidel_file_name = None, regul_file_name = None, save_vectors = False, ready_to_use_vectors = False, NIRSpec_anomaly_handle = True, Hermissianize_Ah = False, verbose = False, Driess = False, debug = False):
    ''' Computes and saves the solution of the NIRCam/NIRSpec fusion problem.

    Output: - z, the solution of the fusion problem in the subspace
    '''

    if ready_to_use_vectors and not save_vectors:
        Ym = fits.getdata("Fusion_Results/"+fusion_name+"prepro_NIRCam.fits")
        if len(NIRCam_test_path) > 0:
            Ym_tests = fits.getdata("Fusion_Results/"+fusion_name+"prepro_NIRCam_test.fits")
        Yh = fits.getdata("Fusion_Results/"+fusion_name+"prepro_NIRSpec.fits")
        Ah, Am, Ar, bh, bm, ch, cm, D, gamma_h, gamma_m, Lh, Lm, mean_Z_fft, mean, P_tilde, P, S, wave_slice, waves, weights, Ym_heigth, Ym_width, z_0 = file_manager.load_preprocessed_vectors(fusion_name)
        vectors = Ah, Am, Ar, bh, bm, ch, cm, D, gamma_h, gamma_m, Lh, Lm, mean_Z_fft, mean, P_tilde, P, S, wave_slice, waves, weights, Ym_heigth, Ym_width, z_0
        wave_slice = slice(*eval(str(wave_slice).replace('slice','')))

    else:
        cube = image.NIRSpec_Image(NIRSpec_path, Yh_height, Yh_width, center_pixel_coord, NIRSpec_angle, Driess, debug)
        wavelengths = cube.wavelengths
        Yh = cube.preprocess()
        if NIRSpec_anomaly_handle:
            filt = filter.NIRSpec_Filter(Yh, fusion_name, exceptions, dev_coeff, anom_limit, verbose)
            Yh = filt.filter()
        Yh = Yh[wave_slice]

        hdu = fits.PrimaryHDU(data = Yh)
        hdu.writeto("Fusion_Results/"+fusion_name+"prepro_NIRSpec.fits", overwrite = True)

        Lm = np.zeros((len(NIRCam_throughputs), Yh.shape[0]))

        if debug :
            print((len(NIRCam_throughputs), Yh.shape[0]))

        for k in range(len(NIRCam_throughputs)):
            NIRCam_throughput = file_manager.read_throughput(NIRCam_throughputs[k])
            i = 0
            while(wavelengths[wave_slice][i] < NIRCam_throughput[0][0]):
                i += 1
            j = -1
            while(wavelengths[wave_slice][j] > NIRCam_throughput[0][-1]):
                j -= 1
            
            if debug :
                print(i,j)

            Lm[k, i:j] = scipy.interpolate.interp1d(NIRCam_throughput[0], NIRCam_throughput[1])(wavelengths[wave_slice][i:j])
        
        Lm_tests = np.zeros((len(NIRCam_test_throughputs), Yh.shape[0]))
        for k in range(len(NIRCam_test_throughputs)):
            NIRCam_test_throughput = file_manager.read_throughput(NIRCam_test_throughputs[k])
            i = 0
            while(wavelengths[wave_slice][i] < NIRCam_test_throughput[0][0]):
                i += 1
            j = -1
            while(wavelengths[wave_slice][j] > NIRCam_test_throughput[0][-1]):
                j -= 1
            Lm_tests[k, i:j] = scipy.interpolate.interp1d(NIRCam_test_throughput[0], NIRCam_test_throughput[1])(wavelengths[wave_slice][i:j])

        if moving_object:
            Ym_0 = image.NIRCam_Image(NIRCam_path[0], Ym_height, Ym_width, cube.center_coord, cube.angle, cube.fov, moving_object, angle_type, little_coord_default, Yh, Lm[0], wave_slice, cube.pix_size, ratio, debug = debug).preprocess()
            Ym = np.array([filter.NIRCam_Filter(image.NIRCam_Image(NIRCam_path[i], Ym_height, Ym_width, cube.center_coord, cube.angle, cube.fov, moving_object, angle_type, little_coord_default, Yh, Lm[i], wave_slice, cube.pix_size, ratio, Ym_0, debug = debug).preprocess(), verbose).filter() for i in range(len(NIRCam_path))])
            Ym_tests = np.array([filter.NIRCam_Filter(image.NIRCam_Image(NIRCam_test_path[i], Ym_height, Ym_width, cube.center_coord, cube.angle, cube.fov, moving_object, angle_type, little_coord_default, Yh, Lm_tests[i], wave_slice, cube.pix_size, ratio, Ym_0, debug = debug).preprocess(), verbose).filter() for i in range(len(NIRCam_test_path))])

            if debug :
                print(Ym_tests.shape)

        else:
            centers = np.array([[3543,3495],[3601,3540],[3543,3495]]) # raw coordinates because d203-506 data are not aligned
            Ym = np.array([filter.NIRCam_Filter(image.NIRCam_Image(NIRCam_path[i], Ym_height, Ym_width, cube.center_coord, cube.angle, cube.fov, moving_object, angle_type, little_coord_default, Yh, Lm[i], wave_slice, cube.pix_size, ratio, debug = debug, center = centers[i]).preprocess(), verbose).filter() for i in range(len(NIRCam_path))])

        hdu = fits.PrimaryHDU(data = Ym)
        hdu.writeto("Fusion_Results/"+fusion_name+"prepro_NIRCam.fits", overwrite = True)
        
        if len(NIRCam_test_path) > 0:
            hdu = fits.PrimaryHDU(data = Ym_tests)
            hdu.writeto("Fusion_Results/"+fusion_name+"prepro_NIRCam_test.fits", overwrite = True)
        
        if verbose:
            sttra = time.time()
        Ah, Am, Ar, bh, bm, ch, cm, D, gamma_h, gamma_m, Lh, Lm, mean_Z_fft, mean, P_tilde, P, S, wave_slice, waves, weights, Ym_heigth, Ym_width, z_0  = compute_linear_system(Ym, Yh, wavelengths, wave_slice, NIRCam_throughputs, NIRSpec_throughput, fusion_name, ratio, compute_NIRCam_psf, compute_NIRSpec_psf, psf_names, nb_comp, sym_pad_size, zero_pad_size, epsilon, calibrate_NIRCam_on_NIRSpec, verbose, debug)
        if verbose:
            print("Computationnal time : ",time.time() - sttra," s")
        if Hermissianize_Ah:
            while (scipy.sparse.linalg.norm((np.conj(Ah).T - Ah), ord = 'fro') > 1e-25):
                Ah = (np.conj(Ah).T + Ah)/2
        vectors = Ah, Am, Ar, bh, bm, ch, cm, D, gamma_h, gamma_m, Lh, Lm, mean_Z_fft, mean, P_tilde, P, S, wave_slice, waves, weights, Ym_heigth, Ym_width, z_0
    
    if len(NIRCam_test_path) > 0:
        Lm_tests = np.zeros((len(NIRCam_test_throughputs), waves.shape[0]))
        for k in range(len(NIRCam_test_throughputs)):
            NIRCam_test_throughput = file_manager.read_throughput(NIRCam_test_throughputs[k])
            i = 0
            while(waves[i] < NIRCam_test_throughput[0][0]):
                i += 1
            j = -1
            while(waves[j] > NIRCam_test_throughput[0][-1]):
                j -= 1
            Lm_tests[k, i:j] = scipy.interpolate.interp1d(NIRCam_test_throughput[0], NIRCam_test_throughput[1])(waves[i:j])

    if Hermissianize_Ah:
        while (scipy.sparse.linalg.norm((np.conj(Ah).T - Ah), ord = 'fro') > 1e-25):
            Ah = (np.conj(Ah).T + Ah)/2

    if save_vectors:
        file_manager.save_preprocessed_vectors(vectors, 3, fusion_name)
        
    if verbose:
        print("Ah is hermitian right ? ", scipy.sparse.linalg.norm((np.conj(Ah).T - Ah), ord = 'fro') < 1e-14)

    objective_function_values, Ym_fidelity_values, Yh_fidelity_values, regularization_values = [],[],[],[]
    
    if verbose:
        sttra3 = time.time()   
    Amcsr = Am.tocsr()
    Ahcsr = Ah.tocsr()
    Arcsr = Ar.tocsr()
    z_final_fft, exit_code = scipy.sparse.linalg.cg(2*(beta_m * Amcsr + beta_h * Ahcsr + mu * Arcsr), -(beta_m * bm + beta_h * bh), maxiter=max_iter)
    if verbose:
        print("Computationnal time optim : ",time.time() - sttra3," s") 

    z = postprocess(z_final_fft, nb_comp, Ym_heigth, Ym_width, sym_pad_size, zero_pad_size)
    X = ((z.T @ P_tilde) + mean).T

    save_fusion_results(fusion_name, z, X, objective_function_values, Ym_fidelity_values, Yh_fidelity_values, regularization_values, mu, f_obj_file_name = obj_func_file_name, Ym_fid_file_name = Ym_fidel_file_name, Yh_fid_file_name = Yh_fidel_file_name, regul_file_name = regul_file_name, verbose = verbose)
    
    return z


def save_fusion_results(fusion_name, z, X, objective_function_values, Ym_fidelity_values,
                        Yh_fidelity_values, regularization_values, mu, save_z = True, 
                        save_x = True, save_f_obj = True, f_obj_file_name = None, save_Ym_fid = True, Ym_fid_file_name = None, save_Yh_fid = True, Yh_fid_file_name = None, save_regul = True, regul_file_name = None, verbose = False):

    if save_z:
        file_name = 'z_' + str(round(mu, 1)) + '.fits'
        file_manager.save_as_fits(z, "Fusion_Results/" +fusion_name+ file_name)

    if save_x:
        file_name = 'X_' + str(round(mu, 1)) + '.fits'
        file_manager.save_as_fits(X, "Fusion_Results/" +fusion_name+ file_name)

    if save_f_obj:
        save_values_list(fusion_name, objective_function_values, mu, f_obj_file_name)

    if save_Ym_fid:
        save_values_list(fusion_name, Ym_fidelity_values, mu, Ym_fid_file_name)

    if save_Yh_fid:
        save_values_list(fusion_name, Yh_fidelity_values, mu, Yh_fid_file_name)
    
    if save_regul:
        save_values_list(fusion_name, regularization_values, mu, regul_file_name)


def save_values_list(fusion_name, values_list, mu, file_name):

    file_name = file_name + str(round(mu, 1)) + '.npy'
    np.save("Fusion_Results/" + fusion_name+file_name, values_list)


def postprocess(z, nb_components, Ym_heigth, Ym_width, sym_pad_size, zero_pad_size):
    
    z = np.reshape(z, (nb_components, Ym_heigth, Ym_width))
    z = np.real(np.fft.ifft2(z, norm = 'ortho'))
    size = sym_pad_size + zero_pad_size
    z = z[:, size:-size, size:-size] 

    return z


def compute_psf_from_list(point_spread_function_list):

    point_spread_function = np.array(point_spread_function_list)
    k, l, m, n = point_spread_function.shape
    point_spread_function = np.reshape(point_spread_function, (k, m, n))

    return point_spread_function


def add_padding(X, sym_pad_size, zero_pad_size):

    shape_length = len(X.shape)

    if shape_length <= 2:

        symmetrical_padding_shape = [(sym_pad_size, sym_pad_size) for i in range(shape_length)]
        zero_padding_shape = [(zero_pad_size, zero_pad_size) for i in range(shape_length)]
    
    else :

        symmetrical_padding_shape = [(0,0)] + [(sym_pad_size, sym_pad_size) for i in range(shape_length -1)]
        zero_padding_shape = [(0,0)] + [(zero_pad_size, zero_pad_size) for i in range(shape_length -1)]
    
    X_symmetrical_padding = np.pad(X, symmetrical_padding_shape, mode = 'symmetric')
    X_full_padding = np.pad(X_symmetrical_padding, zero_padding_shape, mode = 'constant', constant_values = 0)

    return X_full_padding
