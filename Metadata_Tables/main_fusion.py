# Code by Landry Marquis
#
# If you use this code, please cite the paper: 
# Marquis L. et al., Fusion of JWST data: Demonstrating practical feasibility,
# Astronomy & Astrophysics, volume 708, 2026

import time
import figures


def main():

    if verbose:
        start = time.time()

    fuse(fusion_name, NIRCam_path, NIRCam_test_path, NIRSpec_path, Ym_height, Ym_width, Yh_height, Yh_width, center_pixel_coord, NIRSpec_angle, moving_object, angle_type, little_coord_default, dev_coeff, anom_limit, exceptions, wave_slice, NIRCam_throughputs, NIRCam_test_throughputs, NIRSpec_throughput, ratio, compute_NIRCam_psf, compute_NIRSpec_psf, psf_names, epsilon, calibrate_NIRCam_on_NIRSpec, mu, beta_m, beta_h, max_iter, nb_comp, sym_pad_size, zero_pad_size, obj_func_file_name, Ym_fidel_file_name, Yh_fidel_file_name, regul_file_name, save_vectors, ready_to_use_vectors, NIRSpec_anomaly_handle, Hermissianize_Ah, verbose, Driess, debug)

    if verbose:
        print("Computationnal time : ",time.time() - start," s")

from Configurations.config_1288_proplyd_G235H_F170LP_3_filters import *
main()
from Configurations.config_1251_G235H_F170LP_5_filters import *
main()
from Configurations.config_1251_G235H_F170LP_3_filters_validation import *
main()
figures.paper_figures()
