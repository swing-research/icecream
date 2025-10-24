
import torch
import numpy as np
from torch.utils.data import Dataset
from icecream.utils.utils import generate_all_cube_symmetries_torch, crop_volumes, crop_volumes_mask

class MultiVolume(Dataset):
    def __init__(self, volume_1_set,volume_2_set, 
                 crop_size,
                 wedge_set, 
                 wedge_eq_set = None,
                 use_flips = False,
                 mask_set=None,
                 mask_frac = 0.3,
                 n_crops = 2,
                 normalize_crops = False,
                 device = 'cpu'):
        """
        Structure to keep track of all useful information regarding the tomograms.
        """
        self.volume_1_set = volume_1_set
        self.volume_2_set = volume_2_set
        self.mask_set = mask_set
        self.crop_size = crop_size
        self.device = device
        self.wedge_set = wedge_set
        self.wedge_eq_set = wedge_eq_set
        self.mask_frac = mask_frac
        self.use_flips = use_flips
        self.normalize_crops = normalize_crops
        self.n_crops = n_crops

        self.generate_rotation_indeces()

    def generate_rotation_indeces(self):
        """
        Generates the wedge data from the volume data
        """
        random_cube = torch.rand(self.crop_size,self.crop_size,self.crop_size).to(self.device)
        wedge = self.wedge_set[0] 
        if wedge.device is not self.device:
            wedge = wedge.to(self.device)
        _, _, k_sets, _ = generate_all_cube_symmetries_torch(random_cube,wedge,use_flips = self.use_flips)
        self.k_sets = k_sets

    def __len__(self):
        return len(self.volume_1_set)

    def __getitem__(self, idx):
        """
        Returns a random crop of the volume data
        """
        data_dict = self.get_random_crop(idx, self.n_crops)
        wedge_chosen = self.wedge_set[idx]
        data_dict['wedge'] = wedge_chosen
        if self.wedge_eq_set is not None:
            wedge_eq_chosen = self.wedge_eq_set[idx]
            data_dict['wedge_eq'] = wedge_eq_chosen
        data_dict['idx'] = idx
        return data_dict

    def get_random_crop(self, index, n_crops):
        if self.mask_set is not None:
            return self.get_random_crop_mask(index,n_crops)
        else:
            return self.get_random_crop_no_mask(index,n_crops)

    def cropwisze_normalize(self, crops_1, crops_2):
        crops_1_mean = crops_1.mean(dim=(-1,-2,-3), keepdim=True)
        crops_1_std = crops_1.std(dim=(-1,-2,-3), keepdim=True) + 1e-8
        crops_1 = (crops_1 - crops_1_mean) / crops_1_std
        crops_2_mean = crops_2.mean(dim=(-1,-2,-3), keepdim=True)
        crops_2_std = crops_2.std(dim=(-1,-2,-3), keepdim=True) + 1e-8
        crops_2 = (crops_2 - crops_2_mean) / crops_2_std
        return crops_1, crops_2

    def get_random_crop_no_mask(self, index, n_crops):
        crops_1, crops_2 = crop_volumes(self.volume_1_set[index], self.volume_2_set[index],
                                        self.crop_size,n_crops)
        crops_1 = torch.stack(crops_1)
        crops_2 = torch.stack(crops_2)
        if self.normalize_crops:
            crops_1, crops_2 = self.cropwisze_normalize(crops_1, crops_2)
        data_dict = {'input_1': crops_1, 
                     'input_2':  crops_2
        }
        return data_dict

    def get_random_crop_mask(self,index, n_crops):
        crops_1, crops_2 = crop_volumes_mask(self.volume_1_set[index], 
                                                self.volume_2_set[index],
                                                self.mask_set[index],
                                                self.mask_frac,
                                                self.crop_size,
                                                n_crops)
        crops_1 = torch.stack(crops_1)
        crops_2 = torch.stack(crops_2)
        if self.normalize_crops:
            crops_1, crops_2 = self.cropwisze_normalize(crops_1, crops_2)
        data_dict = {'input_1': crops_1, 
                     'input_2':  crops_2
        }
        return data_dict