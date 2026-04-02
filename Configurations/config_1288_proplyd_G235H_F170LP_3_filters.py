import fusion


######### Input data information #########

NIRCam_path = ["/Users/admin/Documents/Fusion_Database/1288_Orion/NIRCam/Level3_CLEAR-F182M-B_i2d_aligned.fits",
               "/Users/admin/Documents/Fusion_Database/1288_Orion/NIRCam/Level3_CLEAR-F187N-B_i2d_aligned.fits",
               "/Users/admin/Documents/Fusion_Database/1288_Orion/NIRCam/Level3_CLEAR-F210M-B_i2d_aligned.fits"]
NIRSpec_path = "/Users/admin/Documents/Fusion_Database/1288_Orion/NIRSpec/jw01288-c1005_t014_nirspec_g235h-f170lp_s3d.fits"
ratio = 3 # integer ratio approximating the pixel size ratio between NIRCam and NIRSpec
center_pixel_coord = [86.5, 176.5]
Yh_height, Yh_width = 10, 10 # rectangular size (in NIRSpec pixels) of the fusion fov, it must be even if center pix coord is .5 and odd if .0
Ym_height, Ym_width = ratio * Yh_height, ratio * Yh_width
wave_slice = slice(0, 1616) # the NIRSpec wavelengths on which fusion is performed 

######### Alignment #########

NIRSpec_angle = 45
moving_object = False
angle_type = 'Orion'
interp_type = 'cubic'
little_coord_default = False
Driess = False

######### Point Spread Functions #########

compute_NIRCam_psf = False
compute_NIRSpec_psf = False
psf_names = ['Point_Spread_Functions/Wm_F170LP.fits',
             'Point_Spread_Functions/Wh_F170LP.fits']
sym_pad_size = ratio * Yh_height
zero_pad_size = 0

######### Throughputs #########

NIRCam_throughputs = ["Throughputs/NIRCam/NRCB2_F182M_system_throughput.txt",
                      "Throughputs/NIRCam/NRCB2_F187N_system_throughput.txt",
                      "Throughputs/NIRCam/NRCB2_F210M_system_throughput.txt"]
NIRSpec_throughput = "Throughputs/NIRSpec/In_Situ/comm_PCE_F170LP_G235H_IFU.fits"
NIRCam_test_path = []
NIRCam_test_throughputs = []

######### Fusion preprocess #########

fusion_name = "1288_d203_506_F170LP_166_230um_3_filters_"
fuse = fusion.fusion
calibrate_NIRCam_on_NIRSpec = True
nb_comp = 3 # number of remaining components after PCA
epsilon = 10**(-2)

######### Optimization #########

save_vectors = True
ready_to_use_vectors = not save_vectors
mu = 0.08
max_iter = 1000
Hermissianize_Ah = False
beta_m = 1000
beta_h = 1

######### Verbose #########

verbose = False
debug = False

######### Other #########

NIRSpec_anomaly_handle = False
dev_coeff = 12
anom_limit = 30
exceptions = []
obj_func_file_name = "obj_gradient_descent_"
Ym_fidel_file_name = "Ym_fidelity_"
Yh_fidel_file_name = "Yh_fidelity_"
regul_file_name = "regularization_"
