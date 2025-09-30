"""
The loader class for the subvolumes
"""


"""
The module contains the class Volume, which is used to store the  FBP volume data 
and the loader also randomly crops and rotates the croped volume to generate the input target pairs
"""


import torch
from torch.utils.data import Dataset   
import numpy as np
from utils import generate_all_cube_symmetries_torch ,crop_volumes, generate_random_rotate


class subvolumes(Dataset):
    def __init__(self, volume, crop_size,wedge, use_flips = False):
        """
        Tensors of size (n1,n2,n3) are stored in a list 
        
        """
        self.volume = volume
        self.crop_size = crop_size

        if isinstance(volume, list):
            self.device = volume[0].device
        else:
            self.device = volume.device
        self.wedge = wedge
        self.use_flips = use_flips

        self.generate_rotation_indeces()

    def __len__(self):
        """
        Returns the number of crops
        """
        return len(self.volume)
    

    def __getitem__(self, index):

        subvolumes = self.volume[index]

        rot_crop, rot_wedge = generate_random_rotate(subvolumes,self.wedge,self.k_sets)

        #print(rot_crop.device)

        rotated_crop_fft = torch.fft.fftshift(torch.fft.fftn(rot_crop, dim =(-3,-2,-1) ), dim=(-3,-2,-1))
        rotated_crop_fft = rotated_crop_fft * self.wedge
        rotated_crop_wedge = torch.fft.ifftn(torch.fft.ifftshift(rotated_crop_fft, dim=(-3,-2,-1)), dim=(-3,-2,-1)).real

        data_dict = {'input': rotated_crop_wedge, 
                'target':  rot_crop, 
                'wedge': rot_wedge,
                'init_crop': subvolumes}

        

        return data_dict


    def generate_rotation_indeces(self):
        """
        Generates the wedge data from the volume data
        """
        random_cube = torch.rand(self.crop_size,self.crop_size,self.crop_size).to(self.device)


        _,_, k_sets = generate_all_cube_symmetries_torch(random_cube,self.wedge,use_flips = self.use_flips)

        self.k_sets = k_sets


    # def get_random_crop(self, n_crops):

    #     # if self.volume is a list of volumes, randomly select one of them
    #     if isinstance(self.volume, list):
    #         volume_choosen = self.volume[np.random.randint(0,len(self.volume))]
    #     else:
    #         volume_choosen = self.volume

    #     crops = crop_volumes(volume_choosen,self.crop_size,n_crops)

    #     # Randomly rotate the crops

    #     rotated_crops = []
    #     wedge_rotated = []

    #     for crop in crops:
    #         rot_crop, rot_wedge = generate_random_rotate(crop,self.wedge,self.k_sets)
    #         rotated_crops.append(rot_crop)
    #         wedge_rotated.append(rot_wedge)

    #     crops = torch.stack(crops)
    #     wedge_rotated = torch.stack(wedge_rotated)
    #     rotated_crops = torch.stack(rotated_crops)


    #     rotated_crops_fft = torch.fft.fftshift(torch.fft.fftn(rotated_crops, dim =(-3,-2,-1) ), dim=(-3,-2,-1))
    #     rotated_crops_fft = rotated_crops_fft * self.wedge[None]
    #     rotated_crops_wedge = torch.fft.ifftn(torch.fft.ifftshift(rotated_crops_fft, dim=(-3,-2,-1)), dim=(-3,-2,-1)).real



    #     data_dict = {'input': rotated_crops_wedge, 
    #                  'target':  rotated_crops, 
    #                  'wedge': wedge_rotated,
    #                  'init_crop': crops}

        

    #     return data_dict
