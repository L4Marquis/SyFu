import pandas
import numpy as np
from tqdm.notebook import tqdm
from matplotlib import pyplot as plt
from ipywidgets import interact
import astropy.units as u
from astropy.coordinates import SkyCoord
import matplotlib.patches as patches
from scipy import ndimage
from copy import deepcopy as copy
import math
import scipy

matching_filters={}
matching_filters["NIRCAM"]=(["F115W","F115W","F162M","F164N","F182M",
                            "F187N","F200W","F210M","F212N","F300M",
                            "F323N","F335M","F356W","F360M","F460M",
                            "F466N","F470N"])
matching_filters["NIRSPEC"]=(["F070LP;G140H","F100LP;G140H","F100LP;G140H","F100LP;G140H","F170LP;G235H",
                              "F170LP;G235H","F170LP;G235H","F170LP;G235H","F170LP;G235H","F290LP;G395H",
                              "F290LP;G395H","F290LP;G395H","F290LP;G395H","F290LP;G395H","F290LP;G395H",
                              "F290LP;G395H","F290LP;G395H"])

MF=pandas.DataFrame(data=matching_filters)

def check_filters(NC_filter,NS_filter):
    subDF=MF[MF["NIRCAM"]==NC_filter]
    res=subDF[subDF["NIRSPEC"]==NS_filter]["NIRSPEC"]
    if len(res)==0:
        return False
    else:
        return True

def triangle_area(xA,yA,xB,yB,xC,yC):
    AB=np.sqrt((xA-xB)**2+(yA-yB)**2)
    BC=np.sqrt((xC-xB)**2+(yC-yB)**2)
    CA=np.sqrt((xA-xC)**2+(yA-yC)**2)
    s=.5*(AB+BC+CA)
    return np.sqrt(s*(s-AB)*(s-BC)*(s-CA))
#test: this should return 0.5
triangle_area(3,2,3,1,4,1)


def triangle_from_sides(AB,BC,CA):
    s=.5*(AB+BC+CA)
    return np.sqrt(s*(s-AB)*(s-BC)*(s-CA))

def polygon_area(xA,yA,xB,yB,xC,yC,xD,yD):
    #could make this quicker if I knew the polygons were 
    #somewhat regular... but best be on the safe side. 
    side1=triangle_area(xA,yA,xB,yB,xC,yC)
    side2=triangle_area(xA,yA,xC,yC,xD,yD)
    return side1+side2

def sky_distance(RA_A,dec_A,RA_B,dec_B):
    c1 = SkyCoord(RA_A*u.deg, dec_A*u.deg, distance=1,frame='icrs')
    c2 = SkyCoord(RA_B*u.deg, dec_B*u.deg, distance=1, frame='icrs')
    return c1.separation_3d(c2).value
    
def coord_to_float(s_region_string):
    coord_list=s_region_string.split() 
    return [float(coord_list[i]) for i in range(1,9)]

def mean_point(s_region_string):
    coord_list=s_region_string.split() 
    RA_A,dec_A,RA_B,dec_B,RA_C,dec_C,RA_D,dec_D=[float(coord_list[i]) for i in range(1,9)]
    RA_M=np.mean([RA_A,RA_B,RA_C,RA_D])
    dec_M=np.mean([dec_A,dec_B,dec_C,dec_D])
    return RA_M,dec_M
    
def check_sky_region(s_region_NIRCam,s_region_NIRSpec):
    RA_M,dec_M=mean_point(s_region_NIRSpec)
    RA_A,dec_A,RA_B,dec_B,RA_C,dec_C,RA_D,dec_D=coord_to_float(s_region_NIRCam)
    # Area of polygon
    AB=sky_distance(RA_A,dec_A,RA_B,dec_B)
    BC=sky_distance(RA_B,dec_B,RA_C,dec_C)
    CD=sky_distance(RA_C,dec_C,RA_D,dec_D)
    DA=sky_distance(RA_D,dec_D,RA_A,dec_A)
    AC=sky_distance(RA_A,dec_A,RA_C,dec_C)
    A_pol=triangle_from_sides(AB,BC,AC)+triangle_from_sides(AC,CD,DA)

    #Area of triangles with M
    MA=sky_distance(RA_M,dec_M,RA_A,dec_A)
    MB=sky_distance(RA_M,dec_M,RA_B,dec_B)
    MC=sky_distance(RA_M,dec_M,RA_C,dec_C)
    MD=sky_distance(RA_M,dec_M,RA_D,dec_D)
    MBA=triangle_from_sides(MB,MA,AB)
    MBC=triangle_from_sides(MB,MC,BC)
    MCD=triangle_from_sides(MC,MD,CD)
    MAD=triangle_from_sides(MA,MD,DA)
    Total=MBA+MBC+MCD+MAD
    if round(Total,10)<=round(A_pol,10):
        return True
    else:
        return False
def check_sky_region_better(s_region_NIRCam,RA_M,dec_M):
    #RA_M,dec_M=mean_point(s_region_NIRSpec)
    RA_A,dec_A,RA_B,dec_B,RA_C,dec_C,RA_D,dec_D=coord_to_float(s_region_NIRCam)
    # Area of polygon
    AB=sky_distance(RA_A,dec_A,RA_B,dec_B)
    BC=sky_distance(RA_B,dec_B,RA_C,dec_C)
    CD=sky_distance(RA_C,dec_C,RA_D,dec_D)
    DA=sky_distance(RA_D,dec_D,RA_A,dec_A)
    AC=sky_distance(RA_A,dec_A,RA_C,dec_C)
    A_pol=triangle_from_sides(AB,BC,AC)+triangle_from_sides(AC,CD,DA)

    #Area of triangles with M
    MA=sky_distance(RA_M,dec_M,RA_A,dec_A)
    MB=sky_distance(RA_M,dec_M,RA_B,dec_B)
    MC=sky_distance(RA_M,dec_M,RA_C,dec_C)
    MD=sky_distance(RA_M,dec_M,RA_D,dec_D)
    MBA=triangle_from_sides(MB,MA,AB)
    MBC=triangle_from_sides(MB,MC,BC)
    MCD=triangle_from_sides(MC,MD,CD)
    MAD=triangle_from_sides(MA,MD,DA)
    Total=MBA+MBC+MCD+MAD
    if round(Total,10)<=round(A_pol,10):
        return True
    else:
        return False
    
def get_pairs(csv_file):
	csvFile = pandas.read_csv(csv_file,comment="#")
	NIRCam_df=csvFile[csvFile["instrument_name"]=="NIRCAM/IMAGE"]
	NIRSpec_df=csvFile[csvFile["instrument_name"]=="NIRSPEC/IFU"]
	Matches=[]
	Info=[]
	for i_NC in tqdm(range(len(NIRCam_df))):
		for j_NS in range(len(NIRSpec_df)):
			if check_filters(NIRCam_df.iloc[i_NC]["filters"],NIRSpec_df.iloc[j_NS]["filters"]):
				if type(NIRCam_df.iloc[i_NC]["s_region"]) is str:
					if type(NIRSpec_df.iloc[j_NS]["s_region"]) is str:
						ra_m,dec_m=NIRSpec_df.iloc[j_NS]["s_ra"],NIRSpec_df.iloc[j_NS]["s_dec"]
						if check_sky_region_better(NIRCam_df.iloc[i_NC]["s_region"],ra_m,dec_m):
						#if check_sky_region(NIRCam_df.iloc[i_NC]["s_region"],NIRSpec_df.iloc[j_NS]["s_region"]):
							Matches.append((i_NC,j_NS))
							Info.append(fr"NIRCam Target: {NIRCam_df.iloc[i_NC]['target_name']} "+
								fr"NIRSpec Target: {NIRSpec_df.iloc[j_NS]['target_name']}"+
								fr"NIRCam Filter: {NIRSpec_df.iloc[j_NS]['target_name']}"+
								fr"NIRSpec Filter: {NIRSpec_df.iloc[j_NS]["filters"]}")
	Matches_np=np.asarray(Matches)
	return Matches_np,Info,NIRCam_df,NIRSpec_df



def organize(Matches_np,NIRCam_df,NIRSpec_df):
	col_list=['instrument_name','filters','target_name', 's_ra', 's_dec', 't_exptime',
       'wavelength_region', 
       'target_classification', 'obs_title', 't_obs_release',
        'proposal_pi', 'proposal_id',
       'objID','obs_id']
	Unique_NC=set(Matches_np[:,0])
	Unique_NS=set(Matches_np[:,1])
	Groups={}
	for i in range(len(Matches_np)):
		NC_t=NIRCam_df.iloc[Matches_np[i,0]]["target_name"]
		NSfilter=NIRSpec_df.iloc[Matches_np[i,1]]["filters"]
		for NS in Unique_NS:
			if NS==Matches_np[i,1]:
				NS_t=NIRSpec_df.iloc[Matches_np[i,1]]["target_name"]
				filt=NSfilter[:6]
				if NC_t not in Groups:
					Groups[NC_t]={}
				if NS_t not in Groups[NC_t]:
					Groups[NC_t][NS_t]={}
				if filt not in Groups[NC_t][NS_t]:
					Groups[NC_t][NS_t][filt]=[[],int(Matches_np[i,1])]
				if int(Matches_np[i,0]) not in Groups[NC_t][NS_t][filt][0]:
					Groups[NC_t][NS_t][filt][0].append(int(Matches_np[i,0]))
	Groups_DF={}
	for NC_t in Groups.keys():
		Indices_NS=[]
		DF_list=[]
		for NS_t in Groups[NC_t].keys():
			for filt in  Groups[NC_t][NS_t].keys():
				NS_ind=Groups[NC_t][NS_t][filt][1]
				Indices_NS.append(NS_ind)
				DF_list.append(pandas.DataFrame(NIRSpec_df.iloc[[NS_ind]])[col_list])
				NS_RA=NIRCam_df.iloc[NS_ind]["s_ra"]
				NS_dec=NIRCam_df.iloc[NS_ind]["s_dec"]
				Indices_NC=[]
				for NC_ind in Groups[NC_t][NS_t][filt][0]: 
					Indices_NC.append(NC_ind)
					RA,dec=NIRCam_df.iloc[NC_ind]["s_ra"],NIRCam_df.iloc[NC_ind]["s_dec"]
				DF_list.append(pandas.DataFrame(NIRCam_df.iloc[Indices_NC])[col_list])
		Groups_DF[NC_t]=pandas.concat(DF_list).drop_duplicates()
	return Groups_DF




def cut_rotate_NS(image,height,width,angle,center_pix_coord):
	semi_height = (height - 1)/2
	semi_width=(width - 1)/2
	image_cut = image[:,round(center_pix_coord[1] - (semi_width + round(0.7*semi_width))):
		round(center_pix_coord[1] + (semi_width + round(0.7*semi_width) + 1)),
		round(center_pix_coord[0] - (semi_height + round(0.7*semi_height))):
		round(center_pix_coord[0] + (semi_height + round(0.7*semi_height) + 1))]
	clean_image = image_cut.copy()
	clean_image[np.isnan(image_cut)] = 0
	rotated_image = ndimage.rotate(clean_image, axes = (1,2),
		angle =angle, reshape = False)
	cropped_image=rotated_image[:, round((rotated_image.shape[1]-1)/2 - semi_width) :
		round((rotated_image.shape[1]-1)/2 + semi_width + 1), 
 		round((rotated_image.shape[2]-1)/2 - semi_height) :
		round((rotated_image.shape[2]-1)/2 + semi_height + 1)]
	return cropped_image

def get_NIRCam(NIRSpec_path,NIRCam_path,NIRCam_throughputs,
	Yh_height,Yh_width,center_pixel_coord, NIRSpec_angle,wave_slice,k,moving_object=False):
	cube = image.NIRSpec_Image(NIRSpec_path, Yh_height, Yh_width, center_pixel_coord, NIRSpec_angle)
	NS_cube=cube.preprocess()
	wavelengths=cube.wavelengths
	NIRCam_throughput = image.read_throughput(NIRCam_throughputs[k])
	Lm = np.zeros((len(NIRCam_throughputs), NS_cube[wave_slice].shape[0]))
	for k in range(len(NIRCam_throughputs)):
		i = 0
		while(wavelengths[wave_slice][i] < NIRCam_throughput[0][0]):
			i += 1
		j = -1
		while(wavelengths[wave_slice][j] > NIRCam_throughput[0][-1]):
			j -= 1 
		Lm[k, i:j] = scipy.interpolate.interp1d(NIRCam_throughput[0], NIRCam_throughput[1])(wavelengths[wave_slice][i:j])
    
	ratio=3
	Ym_height=ratio * Yh_height
	Ym_width=ratio * Yh_width
	NC_image=image.NIRCam_Image(NIRCam_path, Ym_height, Ym_width, 
		cube.center_coord, cube.angle,cube.fov,cube.pix_size,moving_object=moving_object,
		Yh=NS_cube[wave_slice], Lm_i=Lm[0],wave_slice=wave_slice).preprocess()
	return NC_image,NS_cube

