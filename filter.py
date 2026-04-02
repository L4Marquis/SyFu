# Code by Landry Marquis
#
# If you use this code, please cite the paper: 
# Marquis L. et al., Fusion of JWST data: Demonstrating practical feasibility,
# Astronomy & Astrophysics, volume 708, 2026

import numpy as np
import copy
import skimage.transform
import scipy.stats
import matplotlib.pyplot as plt
import os
import shutil


class Filter:

    def __init__(self, name, tensor, verbose):
        self.name = name
        self.tensor = tensor
        self.tot_anom = 0
        self.verbose = verbose


class NIRCam_Filter(Filter):
    
    def __init__(self, Ym, verbose):
        super().__init__('NIRCam_filter', Ym, verbose)
        self.med = np.median(self.tensor, axis = (0,1))
        self.pad_size = 2
        
    def filter(self):
        res = copy.deepcopy(self.tensor)
        anomalies = [(j,k) for j in range(self.tensor.shape[0]) for k in range(self.tensor.shape[1]) if self.tensor[j,k] < 0]
        for (j,k) in anomalies:
            res[j,k] = self.med
        res_pad = np.pad(res, ((self.pad_size,self.pad_size),(self.pad_size,self.pad_size)), 'constant', constant_values=((self.med,self.med),(self.med,self.med)))
        for (j,k) in anomalies:
            res[j,k] = np.median((res_pad[j+1+self.pad_size,k+self.pad_size],
                                  res_pad[j+1+self.pad_size,k+1+self.pad_size],
                                  res_pad[j+self.pad_size,k+1+self.pad_size],
                                  res_pad[j+self.pad_size,k-1+self.pad_size], 
                                  res_pad[j+1+self.pad_size,k-1+self.pad_size], 
                                  res_pad[j-1+self.pad_size,k-1+self.pad_size], 
                                  res_pad[j-1+self.pad_size,k+self.pad_size], 
                                  res_pad[j-1+self.pad_size,k+1+self.pad_size], 
                                  res_pad[j+2+self.pad_size,k+self.pad_size], 
                                  res_pad[j-2+self.pad_size,k+self.pad_size], 
                                  res_pad[j+self.pad_size,k+2+self.pad_size], 
                                  res_pad[j+self.pad_size,k-2+self.pad_size]))
        return res


class NIRSpec_Filter(Filter):

    def __init__(self, Yh, fusion_name, exceptions, dev_coeff = 6, anom_limit = 9, verbose = False, window_width = 8):
        super().__init__('NIRSpec_filter', Yh, verbose)
        self.med = np.median(self.tensor, axis = (1,2))
        self.MAD = scipy.stats.median_abs_deviation(self.tensor, axis = (1,2))
        self.dev_coeff = dev_coeff
        self.anom_limit = anom_limit
        self.pad_size = 2
        self.abnormal_waves = 0
        self.anom = np.zeros_like(self.med)
        self.small_anom = 0
        self.huge_anom = 0
        self.huge_anom_waves = []
        self.huge_anom_clusters = []
        self.window_width = window_width
        self.linewidth = 0.5
        self.fusion_name = fusion_name
        self.exceptions = exceptions

    def filter(self):

        if not os.path.exists("Anomalies/" + self.fusion_name):
            os.makedirs("Anomalies/" + self.fusion_name)
        else:
            shutil.rmtree("Anomalies/" + self.fusion_name)
            os.makedirs("Anomalies/" + self.fusion_name)

        res = copy.deepcopy(self.tensor)
        res = np.where(res < 0, 0, res)
        for i in range(self.tensor.shape[0]):
            if i not in self.exceptions:
                anomalies = [(j,k) for j in range(self.tensor.shape[1]) for k in range(self.tensor.shape[2]) if ((self.tensor[i,j,k] - self.med[i]) > self.dev_coeff * self.MAD[i] or (self.tensor[i,j,k] - self.med[i]) < - self.dev_coeff * self.MAD[i])]
                n = len(anomalies)
                self.anom[i] = n
                self.tot_anom += n
                if n > 0 :
                    self.abnormal_waves += 1
                    if self.verbose:
                        plt.imsave('Anomalies/' + self.fusion_name + '/anomalie' + str(i+1) + '.png', skimage.transform.resize(self.tensor[i], (270,270), order = 0, mode = 'symmetric'), origin = 'lower')
                    u = np.zeros_like(self.tensor[i])
                    for (j,k) in anomalies:
                        u[j,k] = 1
                    if self.verbose:
                        plt.imsave('Anomalies/' + self.fusion_name + '/anomalie' + str(i+1) + 'mask.png', skimage.transform.resize(u, (270,270), order = 0, mode = 'symmetric'), origin = 'lower')
                    if n < self.anom_limit :
                        self.small_anom += 1
                        for (j,k) in anomalies:
                            res[i,j,k] = self.med[i]
                        res_pad = np.pad(res[i], ((self.pad_size,self.pad_size),(self.pad_size,self.pad_size)), 'constant', constant_values=((self.med[i],self.med[i]),(self.med[i],self.med[i])))
                        for (j,k) in anomalies:
                            res[i,j,k] = np.mean((res_pad[j+1+self.pad_size,k+self.pad_size],
                                                res_pad[j-1+self.pad_size,k+self.pad_size],
                                                res_pad[j+self.pad_size,k+1+self.pad_size],
                                                res_pad[j+self.pad_size,k-1+self.pad_size]))
                        if self.verbose:
                            plt.imsave('Anomalies/' + self.fusion_name + '/anomalie' + str(i+1) + 'corrected.png', skimage.transform.resize(res[i], (270,270), order = 0, mode = 'symmetric'), origin = 'lower')
                    else :
                        self.huge_anom += 1
                        self.huge_anom_waves.append(i)
        i = 0
        while i in range(len(self.huge_anom_waves)):
            j = i
            linked_anom = [self.huge_anom_waves[j]]
            if j+1 < len(self.huge_anom_waves):
                while self.huge_anom_waves[j+1] == self.huge_anom_waves[j] + 1:
                    linked_anom.append(self.huge_anom_waves[j+1])
                    if j+2 < len(self.huge_anom_waves):
                        j += 1
                    else:
                        break
            i = j + 1
            self.huge_anom_clusters.append(linked_anom)

        for linked_anom in self.huge_anom_clusters:
            j = min(linked_anom)
            i = max(linked_anom)
            if self.verbose:
                anom_spec = np.mean(res[j-self.window_width:i+self.window_width, : ,: ], axis = (1,2))
                fig, ax = plt.subplots(nrows = 1, ncols = 1)
                ax.set(xlabel = 'Wavelength (um)', ylabel = 'Value (MJy/sr)')
                ax.set_title("NIRSpec huge anomaly spectrum "+ str(j)+ " to "+ str(i))
                ax.plot(np.arange(j-self.window_width,i+self.window_width), anom_spec, "-r", label = 'original', linewidth = self.linewidth)
            for k in linked_anom:
                res[k] = np.sqrt(((k-j+1)/(i-j+2)) * np.square(res[j-1]) + ((i-k+1)/(i-j+2)) * np.square(res[i+1]))
                if self.verbose:
                    plt.imsave('Anomalies/' + self.fusion_name + '/anomalie' + str(k+1) + 'corrected.png', skimage.transform.resize(res[k], (270,270), order = 0, mode = 'symmetric'), origin = 'lower')
            if self.verbose:
                anom_spec = np.mean(res[j-self.window_width:i+self.window_width, : ,: ], axis = (1,2))
                ax.plot(np.arange(j-self.window_width,i+self.window_width), anom_spec, "-b", label = 'corrected', linewidth = self.linewidth)
                fig.savefig("Anomalies/" + self.fusion_name + "/NIRSpec_huge_anomaly_spectrum_" + str(j) + "_to_" + str(i) + ".png", bbox_inches = 'tight')
                plt.close(fig)
        return res
