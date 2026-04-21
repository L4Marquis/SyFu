import math
import scipy.ndimage
import skimage.transform
import numpy as np
from astropy.io import fits
import scipy.interpolate
import skimage.registration
import skimage.transform
import matplotlib.pyplot as plt
from astropy.visualization import ZScaleInterval
from astropy.wcs import WCS
from astropy.coordinates import SkyCoord
import warnings 
class Image:

    def __init__(self, path):
        self.path = path
        self.image = fits.open(path)[1].data # this is a fix
        self.header1 = fits.getheader(path, 1)
        self.header0 = fits.getheader(path)
        self.header1 = fits.getheader(path, 1)
        self.old_center_pix_coord = np.array((self.header1['CRPIX1'], 
                                              self.header1['CRPIX2']))
        self.old_center_coord = np.array((self.header1['CRVAL1'], 
                                          self.header1['CRVAL2']))
        self.increments = np.array((self.header1['CDELT1'], 
                                    self.header1['CDELT2']))
       

class NIRCam_Image(Image):

    def __init__(self, path, height, width, center_coord, NIRSpec_angle, NIRSpec_fov, moving_object, 
        little_coord_default, Yh, Lm_i, wave_slice, NIRSpec_pix_size, ratio, reference_image = None, 
        interp_type = 'cubic', debug = False, center = [3543,3495],NS_coord=None,NS_center=None,align=True):
        super().__init__(path)
        self.instrument = 'NIRCam'
        self.height = height
        self.semi_height = (height - 1)/2
        self.width = width
        self.semi_width = (width - 1)/2
        self.NIRSpec_fov = NIRSpec_fov
        self.rota_matrix = np.array([[self.header1['PC1_1'], 
                                      self.header1['PC1_2']],
                                     [self.header1['PC2_1'],
                                      self.header1['PC2_2']]])
        ((a,b),(c,d)) = self.rota_matrix**(-1)
        self.inv_rota_matrix = (1 / (a**2 + b**2 + c**2 + d**2)) * self.rota_matrix**(-1)
        self.center_coord = center_coord
        self.center_pix_coord = (self.old_center_pix_coord + 
                                 np.dot(self.rota_matrix, self.increments**(-1) * 
                                        (self.center_coord - self.old_center_coord)))
        self.center_pix_coord = np.array([self.center_pix_coord[1], 
                                          self.center_pix_coord[0]])
        self.align=align
            #self.angle = (theta+ NIRSpec_angle - 180) # angle for Titan
        def correct_theta(PC1_1,PC1_2):
            theta_test=math.asin(-PC1_2)
            A=round(np.abs(theta_test),3)
            B=round(np.abs(math.acos(-PC1_1)),3)
            if A!=B:
                return (-theta_test+math.pi)*180/math.pi
            else:
                return theta_test*180/math.pi
        theta=correct_theta(self.header1['PC1_1'],self.header1['PC1_2'])
        if debug:
            print("theta",theta)
        #print("ANGLE",180 /math.asin(-self.header1['PC1_2']))

        #theta=180 / math.pi *math.asin(-self.header1['PC1_2'])

        self.angle = (theta + NIRSpec_angle) # angle for Orion

 
        self.round_center_pix_coord = ((round(self.center_pix_coord[0]), 
                                        round(self.center_pix_coord[1])))
    

        self.moving_object = moving_object
        self.little_coord_default = little_coord_default
        self.Yh = Yh
        self.Lm_i = Lm_i[wave_slice]
        self.wave_slice = wave_slice
        self.NIRSpec_pix_size = NIRSpec_pix_size
        if NS_coord is not None:
            self.NS_coord=NS_coord
        if NS_center is not None:
            self.NS_center=NS_center
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.NC_coord=WCS(self.header1)
        self.ratio = ratio
        self.reference_image = reference_image
        self.interp_type = interp_type
        self.debug = debug
        if self.debug:
            print("right file")
        if self.debug :
            print("Rotation matrix:", self.rota_matrix)
            print("Transposed:", self.rota_matrix.T)
            print('Old NIRSpec centre:',self.center_coord)
            print('Old NIRCam centre:',self.old_center_coord)
            print('New NIRCam centre:',self.old_center_coord)
            print("Diff between old and new: ", (self.increments**(-1) * 
                                            (self.center_coord - self.old_center_coord)))
            print("Angle: ",self.angle)

    def preprocess(self):
        if not self.moving_object :
            clean_image=self.image.copy()
            clean_image[np.where(np.isnan(self.image))]=0
            x_center_pix_coord,y_center_pix_coord=self.center_pix_coord[1],self.center_pix_coord[0]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                xc,yc=self.NC_coord.world_to_array_index(self.NS_center)
            coeff = self.NIRSpec_pix_size / (self.ratio * self.header1['CDELT1'])

            x_min_size=int(2*(self.Yh.shape[1])*self.NIRSpec_pix_size/(self.header1['CDELT1']))#
            #y_min_size=int(2*(self.Yh.shape[2])*self.NIRSpec_pix_size/(self.header1['CDELT1']))#math.sqrt(2)*
            #x_min_size=min(x_min_size,y_min_size)
            #x_min_size=2*int(x_min_size/2)
            #print(x_min_size)
            y_min_size=x_min_size
            if self.debug:
                print("xc, yc: ",xc, yc)
                print("x_min_size, y_min_size: ",x_min_size, y_min_size)
                print("cut x ",xc-x_min_size//2,xc+x_min_size//2)
                print("cut y ",yc-y_min_size//2,yc+y_min_size//2)
            clean_image=clean_image[int(xc-x_min_size/2):int(xc+x_min_size/2),
            int(yc-y_min_size/2):int(yc+y_min_size/2)]
            
            rotated_image = scipy.ndimage.rotate(clean_image#[b:-b,b:-b]
                , angle = self.angle, reshape = True, mode ='mirror')
            if self.debug:
                plt.imshow(rotated_image)
                plt.title("rotated_image")
                plt.show()
                #rotated_image = np.where(np.isnan(rotated_image), np.ones_like(rotated_image), rotated_image)
                print("prior reshape: ", rotated_image.shape)
            rotated_image = skimage.transform.resize(rotated_image, (int(rotated_image.shape[0]/coeff),int(rotated_image.shape[1]/coeff)), order = 3)
            if self.debug:
                print("after reshape: ",rotated_image.shape)
            NIRSpec_F = np.sum(((self.Lm_i) * ((self.Yh[self.wave_slice])).T).T, axis = 0)
            upsampled_hyper = scipy.ndimage.zoom(NIRSpec_F, self.ratio, order = 0)
            if self.reference_image is not None:
                upsampled_hyper = self.reference_image
            size = upsampled_hyper.shape[1]

            center = [int(rotated_image.shape[0]/2), int(rotated_image.shape[1]/2)]
            
            aligned_image = rotated_image[int(center[0] - size // 2) : int(center[0] + size / 2),
                              int(center[1] - size / 2) : int(center[1] + size / 2)]

            gross_cut_image = rotated_image[center[0] - 2*int(size / 2) : 
                                center[0] + 2*int(size / 2) , 
                                center[1] - 2*int(size / 2)  : 
                                center[1] + 2*int(size / 2) ]
            if self.debug:
                print("size upsampled: ",size)
                print("center: ",center)
                plt.imshow(aligned_image)
                plt.title("aligned_image (before loop)")
                plt.show()
                print("aligned_image shape: ",aligned_image.shape)
                
                print("gross_cut_image shape: ",gross_cut_image.shape)
                plt.imshow(gross_cut_image)
                plt.title("gross_cut_image (before loop)")
                plt.show()
            #gross_cut_image = rotated_image[center[0] - int(size / 2) - b : center[0] + int(size / 2) + b, center[1] - int(size / 2) - b : center[1] + int(size / 2) + b]
            if self.debug :
                f, a = plt.subplots(nrows = 1, ncols=2)
                a[0].imshow(aligned_image)
                a[0].set_title(f"aligned_image_before")
                a[1].imshow(upsampled_hyper)
                a[1].set_title(f"upsampled_hyper")
                plt.show()

            if self.align:
                for i in range(5):
                    if self.debug :
                        print("aligned_image",aligned_image.shape, "upsampled_hyper", upsampled_hyper.shape)
                    shift, error, diffphase = skimage.registration.phase_cross_correlation(aligned_image, upsampled_hyper, upsample_factor = 100,
                                                                        disambiguate=False,overlap_ratio=0.8)

                    if self.debug :
                        print("shift",shift, "error", error, "diff",diffphase)

                    gross_cut_image = np.real(np.fft.ifft2(scipy.ndimage.fourier_shift(np.fft.fft2(gross_cut_image), -shift)))
                    aligned_image = gross_cut_image[size//2:-size//2,size//2:-size//2]
                    if self.debug :
                        f, a = plt.subplots(nrows = 1, ncols=2)
                        a[0].imshow(aligned_image)
                        a[0].set_title(f"aligned_image_{i}")
                        a[1].imshow(upsampled_hyper)
                        a[1].set_title(f"upsampled_hyper")
                        plt.show()

                shift, error, diffphase = skimage.registration.phase_cross_correlation(aligned_image, upsampled_hyper, upsample_factor = 100,
                                                                    disambiguate=False,overlap_ratio=0.8)

                if self.debug :
                    print("shift",shift, "error", error, "diff",diffphase)

            self.image = aligned_image
            
        else:
            #b = 30
            #b = 0
            clean_image=self.image.copy()
            clean_image[np.where(np.isnan(self.image))]=1
            rotated_image = scipy.ndimage.rotate(clean_image#[b:-b,b:-b]
                , angle = self.angle, reshape = True, mode ='mirror')
            if self.debug:
                plt.imshow(rotated_image)
                plt.title("rotated_image")
                plt.show()
            #rotated_image = np.where(np.isnan(rotated_image), np.ones_like(rotated_image), rotated_image)
            coeff = self.NIRSpec_pix_size / (self.ratio * self.header1['CDELT1'])
            rotated_image = skimage.transform.resize(rotated_image, (int(rotated_image.shape[0]/coeff),int(rotated_image.shape[1]/coeff)), order = 3)

            NIRSpec_F = np.sum(((self.Lm_i) * ((self.Yh[self.wave_slice])).T).T, axis = 0)
            upsampled_hyper = scipy.ndimage.zoom(NIRSpec_F, self.ratio, order = 0)
            if self.reference_image is not None:
                upsampled_hyper = self.reference_image
            size = upsampled_hyper.shape[1]
            center = [int(rotated_image.shape[0]/2), int(rotated_image.shape[1]/2)]
            aligned_image = rotated_image[center[0] - int(size / 2) : center[0] + int(size / 2),
                              center[1] - int(size / 2) : center[1] + int(size / 2)]
            gross_cut_image = rotated_image[center[0] - int(size / 2) : center[0] + int(size / 2) , center[1] - int(size / 2)  : center[1] + int(size / 2) ]
            #gross_cut_image = rotated_image[center[0] - int(size / 2) - b : center[0] + int(size / 2) + b, center[1] - int(size / 2) - b : center[1] + int(size / 2) + b]
            if self.align:
                for i in range(5):
                    shift, error, diffphase = skimage.registration.phase_cross_correlation(aligned_image, upsampled_hyper, upsample_factor = 100)

                    if self.debug :
                        print("shift",shift, "error", error, "diff",diffphase)

                    gross_cut_image = np.real(np.fft.ifft2(scipy.ndimage.fourier_shift(np.fft.fft2(gross_cut_image), -shift)))
                    aligned_image = gross_cut_image#[size//2:-size//2,size//2:-size//2]
                    if self.debug :
                        plt.imshow(aligned_image)
                        plt.title(f"aligned_image_{i}")
                        plt.show()
                if self.debug :
                    f, a = plt.subplots(nrows = 1, ncols=2)
                    a[0].imshow(aligned_image)
                    a[1].imshow(upsampled_hyper)
                    plt.show()

                shift, error, diffphase = skimage.registration.phase_cross_correlation(aligned_image, upsampled_hyper, upsample_factor = 100)

                if self.debug :
                    print("shift",shift, "error", error, "diff",diffphase)

            self.image = aligned_image

        self.image = np.where(np.isnan(self.image), np.zeros_like(self.image), self.image)
        
        if self.debug :
            plt.figure()
            print('NIRCam')
            plt.imshow(self.image, origin = 'lower')
            plt.show()

        return self.image


class NIRSpec_Image(Image):

    def __init__(self, path, height, width, center_pix_coord, angle, Driess, debug):
        super().__init__(path)
        self.increments = np.array((-self.increments[0], 
                                    self.increments[1]))
        if Driess:
            self.wavelengths = fits.getdata(path, "WAVE")
        else:
            self.wavelengths = np.array([i * self.header1['CDELT3'] + (self.header1['WAVSTART'] * 1e6) for i in range(self.header1['NAXIS3'])])
        self.instrument = 'NIRSpec'
        self.center_pix_coord = np.array(center_pix_coord)
        self.center_coord = (self.old_center_coord - self.increments * 
                             (self.old_center_pix_coord - self.center_pix_coord))
        self.angle = angle
        self.pix_size = self.header1['CDELT1']
        self.height = height
        self.semi_height = (height - 1)/2
        self.width = width
        self.semi_width = (width - 1)/2
        self.debug = debug
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.NS_coord=WCS(self.header1)
            self.NS_center=self.NS_coord.array_index_to_world(0,center_pix_coord[1],center_pix_coord[0])[0]
        if self.debug :
            print("NIRSpec center pix coord :", self.center_pix_coord)
            print("NIRSpec center coord :", self.center_coord)

    def preprocess(self):
        image_cut = self.image[:,round(self.center_pix_coord[1] - (self.semi_width + round(0.7*self.semi_width))):
                               round(self.center_pix_coord[1] + (self.semi_width + round(0.7*self.semi_width) + 1)),
                               round(self.center_pix_coord[0] - (self.semi_height + round(0.7*self.semi_height))):
                               round(self.center_pix_coord[0] + (self.semi_height + round(0.7*self.semi_height) + 1))]
        
        if self.debug :
            print(round(self.center_pix_coord[1] - (self.semi_width + round(0.7*self.semi_width))))
            print(round(self.center_pix_coord[1] + (self.semi_width + round(0.7*self.semi_width) + 1)))
            print(round(self.center_pix_coord[0] - (self.semi_height + round(0.7*self.semi_height))))
            print(round(self.center_pix_coord[0] + (self.semi_height + round(0.7*self.semi_height) + 1)))
            print(image_cut.shape)
            print('NIRSpec')
            plt.figure()
            plt.imshow(np.nansum(image_cut[:,:,:],0))
            plt.show()

        self.clean_image = image_cut.copy()
        self.clean_image[np.isnan(image_cut)] = 0
        self.rotated_image = scipy.ndimage.rotate(self.clean_image, axes = (1,2),
                                                  angle = self.angle, reshape = False) #small error
        
        if self.debug :
            print(round((self.rotated_image.shape[1]-1)/2 - self.semi_width))
            print(round((self.rotated_image.shape[1]-1)/2 + self.semi_width + 1))
            print(round((self.rotated_image.shape[2]-1)/2 - self.semi_height))
            print(round((self.rotated_image.shape[2]-1)/2 + self.semi_height + 1))

        self.image = self.rotated_image[:, round((self.rotated_image.shape[1]-1)/2 - self.semi_width) :
                                        round((self.rotated_image.shape[1]-1)/2 + self.semi_width + 1), 
                                        round((self.rotated_image.shape[2]-1)/2 - self.semi_height) :
                                        round((self.rotated_image.shape[2]-1)/2 + self.semi_height + 1)]
        
        if self.debug :
            print(self.image.shape)
            print(self.rotated_image.shape)
            print(self.clean_image.shape)
            
        sizes = np.array([self.height, self.width])
        self.fov = [self.center_coord - (sizes * self.increments) / 2,
                    self.center_coord + (sizes * self.increments) / 2]
        
        if self.debug :
            print(self.center_coord)
            print(self.fov)
            print('NIRSpec')
            plt.figure()
            plt.imshow(np.nansum(self.image[:,:,:],0))
            plt.show()

        return self.image


