import fusion


######### Input data information #########

NIRCam_path = [
               "/Users/admin/Documents/Fusion_Database/1251_Titan/NIRCam/jw01251-o005_t002_nircam_clear-f182m-sub160p_i2d.fits",
               "/Users/admin/Documents/Fusion_Database/1251_Titan/NIRCam/jw01251-o005_t002_nircam_clear-f200w-sub160p_i2d.fits",
               "/Users/admin/Documents/Fusion_Database/1251_Titan/NIRCam/jw01251-o005_t002_nircam_clear-f210m-sub160p_i2d.fits"]
NIRSpec_path = "/Users/admin/Documents/Fusion_Database/1251_Titan/NIRSpec/jw01251-o004_t002_nirspec_g235h-f170lp_s3d.fits"
ratio = 3
center_pixel_coord = [27.5, 25.5]
Yh_height, Yh_width = 14, 14 # must be even if center pix coord is .5 and odd if .0
Ym_height, Ym_width = ratio * Yh_height, ratio * Yh_width
wave_slice = slice(0, 1616)

######### Alignment #########

NIRSpec_angle = 45 #angle error
moving_object = True
angle_type = 'Titan'
interp_type = 'cubic'
little_coord_default = False
Driess = False

######### Point Spread Functions #########

compute_NIRCam_psf = False
compute_NIRSpec_psf = False
psf_names = ['Point_Spread_Functions/Wm_F170LP.fits',
             'Point_Spread_Functions/Wh_F170LP.fits']
sym_pad_size = Ym_height
zero_pad_size = 0

######### Throughputs #########

NIRCam_throughputs = ["Throughputs/NIRCam/NRCB1_F182M_system_throughput.txt",
                      "Throughputs/NIRCam/NRCB1_F200W_system_throughput.txt",
                      "Throughputs/NIRCam/NRCB1_F210M_system_throughput.txt"]
NIRSpec_throughput = "Throughputs/NIRSpec/In_Situ/comm_PCE_F170LP_G235H_IFU.fits"
NIRCam_test_path = [
                    "/Users/admin/Documents/Fusion_Database/1251_Titan/NIRCam/jw01251-o005_t002_nircam_clear-f187n-sub160p_i2d.fits",
                    "/Users/admin/Documents/Fusion_Database/1251_Titan/NIRCam/jw01251-o005_t002_nircam_clear-f212n-sub160p_i2d.fits"]
NIRCam_test_throughputs = ["Throughputs/NIRCam/NRCB1_F187N_system_throughput.txt",
                           "Throughputs/NIRCam/NRCB1_F212N_system_throughput.txt"]

######### Fusion preprocess #########

fusion_name = "1251_Titan_F170LP_166_230um_3_filters_validation_"
fuse = fusion.fusion
calibrate_NIRCam_on_NIRSpec = True
nb_comp = 3
epsilon = 10**(-2)

######### Optimization #########

save_vectors = True
ready_to_use_vectors = not save_vectors
mu = 0.2
max_iter = 1000
Hermissianize_Ah = False
beta_m = 10000
beta_h = 1

######### Verbose #########

verbose = False
debug = False

######### Other #########

NIRSpec_anomaly_handle = False
dev_coeff = 1e10
anom_limit = 30
exceptions = []
obj_func_file_name = "obj_gradient_descent_"
Ym_fidel_file_name = "Ym_fidelity_"
Yh_fidel_file_name = "Yh_fidelity_"
regul_file_name = "regularization_"
