"""
The module contains the class Volume, which is used to store the  FBP volume data 
and the loader also randomly crops and rotates the croped volume to generate the input target pairs
"""


import torch
import numpy as np
from icecream.utils.utils import generate_all_cube_symmetries_torch ,crop_volumes,crop_volumes_mask
from icecream.utils.mask_util import make_mask
from icecream.utils.utils import upsample_fourier_rfft2x

from skimage.filters import window as skimage_window
class singleVolume:
    def __init__(self, volume_1,volume_2, 
                 crop_size,
                 wedge, 
                 use_flips = False,
                 mask=None,
                 mask_frac = 0.3,
                 normalize_crops = False,
                 device = 'cpu',
                 min_distance = 0.5,
                 window_type = 'boxcar',
                 upsample_volume = False):
        """
        Tensors of size (n1,n2,n3) are stored in a list 
        
        """
        self.volume_1 = volume_1
        self.volume_2 = volume_2
        self.mask = mask
        self.crop_size = crop_size
        self.upsample_volume = upsample_volume
        self.device = device

        self.wedge = wedge
        self.mask_frac = mask_frac
        self.use_flips = use_flips
        self.normalize_crops = normalize_crops
        self.min_distance = min_distance

        crop_size = self.crop_size
        if self.upsample_volume:
            crop_size *= 2

        self.window = torch.tensor(skimage_window(window_type, (crop_size,crop_size, crop_size)),
                                   dtype=torch.float32, 
                                   device=self.device)[None]
        





        self.generate_rotation_indeces()



        if self.upsample_volume:
            print('Upsampling the volume')
            self.volume_1 = upsample_fourier_rfft2x(self.volume_1)
            self.volume_2 = upsample_fourier_rfft2x(self.volume_2)

            if self.mask is not None:
                vol_avg = (self.volume_1 + self.volume_2) / 2
                mask = make_mask(vol_avg.cpu().numpy(),mask_boundary = None, side = 5, density_percentage=50., std_percentage=50)
                self.mask = torch.tensor(mask, dtype=torch.float32, device=self.volume_1.device)



    def upsample_operator(self, volume):
        volume_fft = torch.fft.fftshift(torch.fft.fftn(volume, dim=(-3, -2, -1)))

        n1, n2, n3 = volume.shape

        volume_fft_upsampled = torch.zeros((n1 * 2, n2 * 2, n3 * 2), dtype=volume_fft.dtype)

        volume_fft_upsampled[n1-n1//2:n1+n1//2,
                              n2-n2//2:n2+n2//2,
                              n3-n3//2:n3+n3//2] = volume_fft
        volume_upsampled = torch.fft.ifftn(torch.fft.ifftshift(volume_fft_upsampled), dim=(-3, -2, -1)).real.to(volume.dtype)
        # Clear the memory
        del volume_fft, volume_fft_upsampled
        return volume_upsampled



    def generate_rotation_indeces(self):
        """
        Generates the wedge data from the volume data
        """
        random_cube = torch.rand(self.crop_size,self.crop_size,self.crop_size).to(self.device)


        _,_, k_sets, distances = generate_all_cube_symmetries_torch(random_cube,self.wedge,use_flips = self.use_flips, min_distance=self.min_distance)

        self.k_sets = k_sets
        self.distances = distances


    def get_random_crop(self, n_crops):

        if self.mask is not None:
            return self.get_random_crop_mask(n_crops)
        else:
            return self.get_random_crop_no_mask(n_crops)
        

    def cropwisze_normalize(self, crops_1, crops_2):
        crops_1_mean = crops_1.mean(dim=(-1,-2,-3), keepdim=True)
        crops_1_std = crops_1.std(dim=(-1,-2,-3), keepdim=True) + 1e-8

        crops_1 = (crops_1 - crops_1_mean) / crops_1_std
        crops_2_mean = crops_2.mean(dim=(-1,-2,-3), keepdim=True)
        crops_2_std = crops_2.std(dim=(-1,-2,-3), keepdim=True) + 1e-8
        crops_2 = (crops_2 - crops_2_mean) / crops_2_std

        return crops_1, crops_2


    
    def get_random_crop_no_mask(self, n_crops):



        crops_1, crops_2 = crop_volumes(self.volume_1, self.volume_2,self.crop_size,n_crops)

        crops_1 = torch.stack(crops_1)
        crops_2 = torch.stack(crops_2)

        if self.normalize_crops:
            crops_1, crops_2 = self.cropwisze_normalize(crops_1, crops_2)


        data_dict = {'input_1': crops_1, 
                     'input_2':  crops_2
        }

        return data_dict
    
    def downsample_crop(self, crops):

        b,n1,n2,n3 = crops.shape

        crops_fft = torch.fft.fftshift(torch.fft.fftn(crops, dim=(-3, -2, -1)))

        crops_fft = crops_fft[:,n1//2-n1//4:n1//2+n1//4,
                               n2//2-n2//4:n2//2+n2//4,
                               n3//2-n3//4:n3//2+n3//4]
        
        crops_downsampled = torch.fft.ifftn(torch.fft.ifftshift(crops_fft), dim=(-3, -2, -1)).real
        return crops_downsampled

    def get_random_crop_mask(self, n_crops):

        if self.upsample_volume:
            crop_size = self.crop_size * 2
        else:
            crop_size = self.crop_size
        crops_1, crops_2 = crop_volumes_mask(self.volume_1, self.volume_2,self.mask, self.mask_frac,crop_size,n_crops)





        crops_1 = torch.stack(crops_1).to(self.device)
        crops_2 = torch.stack(crops_2).to(self.device)

        # Apply the window to the crops
        crops_1 = crops_1 * self.window
        crops_2 = crops_2 * self.window

        # Downsample the crops if upsample_volume is True
        if self.upsample_volume:
            crops_1 = self.downsample_crop(crops_1)
            crops_2 = self.downsample_crop(crops_2)
        if self.normalize_crops:
            crops_1, crops_2 = self.cropwisze_normalize(crops_1, crops_2)

        data_dict = {'input_1': crops_1, 
                     'input_2':  crops_2
        }

        return data_dict

    





        
    



