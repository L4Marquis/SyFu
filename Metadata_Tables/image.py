# Code by Landry Marquis
#
# If you use this code, please cite the paper: 
# Marquis L. et al., Fusion of JWST data: Demonstrating practical feasibility,
# Astronomy & Astrophysics, volume 708, 2026

import math
import scipy.ndimage
import skimage.transform
import numpy as np
import scipy.interpolate
import skimage.registration
import skimage.transform
import matplotlib.pyplot as plt
from astropy.io import fits
from astropy.visualization import ZScaleInterval


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

    def __init__(self, path, height, width, center_coord, NIRSpec_angle, NIRSpec_fov, moving_object, angle_type, little_coord_default, Yh, Lm_i, wave_slice, NIRSpec_pix_size, ratio, reference_image = None, interp_type = 'cubic', debug = False, center = [3543,3495]):
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
        if angle_type == 'Orion':
            self.center_pix_coord = np.array(center)
 
        self.round_center_pix_coord = ((round(self.center_pix_coord[0]), 
                                        round(self.center_pix_coord[1])))
        self.angle = ((180 / math.pi * math.acos(self.header1['PC1_1'])/1) + 
                      NIRSpec_angle - 180) # angle for Titan
        if angle_type == 'Orion':
            self.angle = (180 - (180 / math.pi * math.acos(self.header1['PC1_1'])) + 
                        NIRSpec_angle) # angle for Orion

        self.moving_object = moving_object
        self.little_coord_default = little_coord_default
        self.Yh = Yh
        self.Lm_i = Lm_i[wave_slice]
        self.wave_slice = wave_slice
        self.NIRSpec_pix_size = NIRSpec_pix_size
        self.ratio = ratio
        self.reference_image = reference_image
        self.interp_type = interp_type
        self.debug = debug
        if self.debug :
            print("Rotation matrix:", self.rota_matrix)
            print('Old NIRSpec centre:',self.center_coord)
            print('Old NIRCam centre:',self.old_center_coord)
            print('New NIRCam centre:',self.old_center_coord)
            print("Diff between old and new: ", (self.increments**(-1) * 
                                            (self.center_coord - self.old_center_coord)))
            print("Angle: ",self.angle)

    def preprocess(self):
        if not self.moving_object :
            thin_cut_height, thin_cut_width = self.semi_height + 8, self.semi_width + 8
            gross_cut_height = thin_cut_height + round(thin_cut_height)
            gross_cut_width = thin_cut_width + round(thin_cut_width)
            fov_pix_height, fov_pix_width = complex(0, self.height), complex(0, self.width)

            self.center_NIRCam_grid = (self.center_coord + np.dot(self.inv_rota_matrix, (self.round_center_pix_coord - self.center_pix_coord)) * self.increments)
            if self.debug:
                print("Image shape:",self.image.shape)
            gross_cut = self.image[round(self.round_center_pix_coord[0] - gross_cut_height) : 
                                round(self.round_center_pix_coord[0] + gross_cut_height) +1,
                                round(self.round_center_pix_coord[1] - gross_cut_width) : 
                                round(self.round_center_pix_coord[1] + gross_cut_width) +1]
            rotated_image = scipy.ndimage.rotate(gross_cut, angle = self.angle, reshape = False)
            #small error
            center_pix = np.array([(rotated_image.shape[0]-1) / 2, 
                                (rotated_image.shape[1]-1) / 2]) #small error
            self.thin_cut = rotated_image[round(center_pix[0] - thin_cut_width) : 
                                            round(center_pix[0] + thin_cut_width) +1,
                                            round(center_pix[1] - thin_cut_height) : 
                                            round(center_pix[1] + thin_cut_height) +1]
            
            sizes = np.array([self.thin_cut.shape[1]/2, self.thin_cut.shape[0]/2])
            self.old_fov = [self.center_NIRCam_grid - (sizes * self.increments),
                            self.center_NIRCam_grid + (sizes * self.increments)]
            
            self.new_fov = self.NIRSpec_fov

            if self.debug :
                print(self.center_pix_coord)
                print(thin_cut_height, thin_cut_width)
                print(gross_cut_height, gross_cut_width)
                print('gross cut shape', gross_cut.shape)
                print(center_pix)
                print('old fov',self.old_fov)
                print('new fov',self.new_fov)
                print(self.old_fov[0][0]-self.old_fov[1][0])
                print(self.new_fov[0][0]-self.new_fov[1][0])
                print(self.old_fov[0][1]-self.old_fov[1][1])
                print(self.new_fov[0][1]-self.new_fov[1][1])
                z = ZScaleInterval()
                z1,z2 = z.get_limits(gross_cut)
                plt.figure()
                print('Gross cut')
                plt.imshow(gross_cut,  origin = 'lower')
                plt.show()
                z = ZScaleInterval()
                z1,z2 = z.get_limits(self.thin_cut)
                plt.figure()
                print('Thin cut')
                plt.imshow(self.thin_cut, origin = 'lower')
                plt.show()
                print('thin cut shape', self.thin_cut.shape)

            min_x = float(min(self.new_fov[0][1], self.new_fov[1][1]))
            max_x = float(max(self.new_fov[0][1], self.new_fov[1][1]))
            min_y = float(min(self.new_fov[0][0], self.new_fov[1][0]))
            max_y = float(max(self.new_fov[0][0], self.new_fov[1][0]))
            old_min_x = float(min(self.old_fov[0][1], self.old_fov[1][1]))
            old_max_x = float(max(self.old_fov[0][1], self.old_fov[1][1]))
            old_min_y = float(min(self.old_fov[0][0], self.old_fov[1][0]))
            old_max_y = float(max(self.old_fov[0][0], self.old_fov[1][0]))

            if self.little_coord_default:
                max_pix_diff = 11
                max_x += max_pix_diff * ((max_x - min_x) / self.height)
                min_x -= max_pix_diff * ((max_x - min_x) / self.height)
                max_y += max_pix_diff * ((max_y - min_y) / self.width)
                min_y -= max_pix_diff * ((max_y - min_y) / self.width)
                fov_pix_height, fov_pix_width = complex(0, self.height+2 * max_pix_diff), complex(0, self.width+2 * max_pix_diff)

            if self.debug :
                print(min_x > old_min_x)
                print(min_y > old_min_y)
                print(max_x < old_max_x)
                print(max_y < old_max_y)

            grid_x, grid_y = np.mgrid[min_x: max_x: fov_pix_height, min_y: max_y: fov_pix_width]
            self.grids = (np.reshape(grid_x, np.prod(grid_x.shape)), 
                        np.reshape(grid_y, np.prod(grid_y.shape)))
            old_grid_x, old_grid_y = np.mgrid[old_min_x: old_max_x: complex(0, self.thin_cut.shape[1]), old_min_y: old_max_y: complex(0, self.thin_cut.shape[0])]
            old_grid_x, old_grid_y = np.reshape(old_grid_x, np.prod(old_grid_x.shape), order='F'), np.reshape(old_grid_y, np.prod(old_grid_y.shape), order='F')
            self.points = np.array([[i,j] for i,j in zip(old_grid_x, old_grid_y)])
            values = self.thin_cut.reshape(np.prod(self.thin_cut.shape))

            if self.debug :
                print(self.points.shape)
                print(values.shape)

            self.image = scipy.interpolate.griddata(self.points, values, (grid_x, grid_y), method = self.interp_type).T

            for i in range(self.image.shape[0]):
                for j in range(self.image.shape[1]):
                    if self.image[i,j] == np.inf :
                        self.image[i,j] = (self.image[i+1,j] + self.image[i,j+1]+ self.image[i-1,j] + self.image[i,j-1])/4

            if self.little_coord_default:
                NIRSpec_F = np.sum(((self.Lm_i) * ((self.Yh[self.wave_slice])).T).T, axis = 0)
                upsampled_hyper = np.pad(scipy.ndimage.zoom(NIRSpec_F, self.ratio, order = 0),((max_pix_diff,max_pix_diff),(max_pix_diff,max_pix_diff)), mode='symmetric')
                self.image = np.where(np.isnan(self.image), np.zeros_like(self.image), self.image)
                upsampled_hyper = np.where(np.isnan(upsampled_hyper), np.zeros_like(upsampled_hyper), upsampled_hyper)
                shift, error, diffphase = skimage.registration.phase_cross_correlation(self.image
                    , upsampled_hyper, upsample_factor = 100)
                
                if self.debug :
                    print(upsampled_hyper.shape)
                    print(self.image.shape)
                    print(shift, error, diffphase)

                gross_cut_image = np.real(np.fft.ifft2(scipy.ndimage.fourier_shift(np.fft.fft2(self.image), -shift)))
                self.image = gross_cut_image[max_pix_diff:-max_pix_diff,max_pix_diff:-max_pix_diff]
        
        else:
            b = 30
            rotated_image = scipy.ndimage.rotate(self.image[b:-b,b:-b], angle = self.angle, reshape = True, mode ='mirror')
            rotated_image = np.where(np.isnan(rotated_image), np.ones_like(rotated_image), rotated_image)
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
            gross_cut_image = rotated_image[center[0] - int(size / 2) - b : center[0] + int(size / 2) + b, center[1] - int(size / 2) - b : center[1] + int(size / 2) + b]
            for i in range(5):
                shift, error, diffphase = skimage.registration.phase_cross_correlation(aligned_image, upsampled_hyper, upsample_factor = 100)

                if self.debug :
                    print(shift, error, diffphase)

                gross_cut_image = np.real(np.fft.ifft2(scipy.ndimage.fourier_shift(np.fft.fft2(gross_cut_image), -shift)))
                aligned_image = gross_cut_image[b:-b,b:-b]
            
            if self.debug :
                f, a = plt.subplots(nrows = 1, ncols=2)
                a[0].imshow(aligned_image)
                a[1].imshow(upsampled_hyper)
                plt.show()

            shift, error, diffphase = skimage.registration.phase_cross_correlation(aligned_image, upsampled_hyper, upsample_factor = 100)

            if self.debug :
                print(shift, error, diffphase)

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
            plt.imshow(image_cut[600,:,:],origin = 'lower')
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
            plt.imshow(self.image[600,:,:], origin = 'lower')
            plt.show()

        return self.image
