# Code by Landry Marquis
#
# If you use this code, please cite the paper: 
# Marquis L. et al., Fusion of JWST data: Demonstrating practical feasibility,
# Astronomy & Astrophysics, volume 708, 2026

import stpsf
import numpy as np
import tools
import fusion
import file_manager
import os
# os.environ["WEBBPSF_PATH"] = "/Users/admin/Documents/webbpsf-data"


class PSF:

    def __init__(self, name, wave, size, overspl, subspl):
        self.name = name
        self.wave = wave
        self.size = size
        self.overspl = overspl
        self.subspl = subspl
        self.subspl_wave = tools.subsample_list_uniformly(self.wave, self.subspl)
        self.instrument = None

    def calc(self, wave):
        subspl_psf_hdu = self.instrument.calc_datacube(wave[0:9999], fov_pixels = self.size, oversample = self.overspl)
        subspl_psf_list = subspl_psf_hdu[0].data

        for i in range(len(self.wave) // 9999):
            subspl_psf_hdu = self.instrument.calc_datacube(wave[9999*(i+1):9999*(i+2)], fov_pixels = self.size, oversample = self.overspl)
            subspl_psf_list = np.concatenate((subspl_psf_list, subspl_psf_hdu[0].data), axis = 0)

        return subspl_psf_list

    def postprocess(self, list):
        psf_list = tools.overspl_by_copy(list, self.subspl)
        psf = fusion.compute_psf_from_list(psf_list)
        file_manager.save_as_fits(psf, self.name)
        return psf
    
    
class PSF_NIRCam(PSF):

    def __init__(self, name, wave, size, overspl = 1, subspl = 1):
        super().__init__(name, wave, size, overspl, subspl)
        self.instrument = stpsf.NIRCam()
    
    def compute(self):
        short_wave = [i for i in self.subspl_wave if i < 2.35 * 1e-6]
        long_wave = [i for i in self.subspl_wave if i >= 2.35 * 1e-6]

        if len(short_wave) > 0 and len(long_wave) > 0:
            short_subspl_psf_list = super().calc(short_wave)
            self.instrument.filter = 'F300M' # trick to get long wavelengths mode activated
            long_subspl_psf_list = super().calc(long_wave)
            subspl_psf_list = np.concatenate((short_subspl_psf_list, long_subspl_psf_list), axis = 0)
        elif len(short_wave) > 0:
            subspl_psf_list = super().calc(short_wave)
        elif len(long_wave) > 0:
            self.instrument.filter = 'F300M'
            subspl_psf_list = super().calc(long_wave)
        else:
            return ValueError("Wave must not be empty.")
        return super().postprocess(subspl_psf_list)


class PSF_NIRSpec(PSF):

    def __init__(self, name, wave, size, overspl = 1, subspl = 1):
        super().__init__(name, wave, size, overspl, subspl)
        self.instrument = stpsf.NIRSpec()
        self.instrument.image_mask = None
    
    def compute(self):
        subspl_psf_list = super().calc(self.subspl_wave)
        return super().postprocess(subspl_psf_list)
