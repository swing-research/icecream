from torch.utils.data import Dataset
from tqdm import tqdm
import torch
import os


class ddw_volumes(Dataset):
    def __init__(self, path_1,path_2, preload =False):

        self.path_1 = path_1
        self.path_2 = path_2
        self.preload = preload

        self.vol_names = os.listdir(path_1)

        if self.preload:
            self.load_volumes()

    def __len__(self):
        return len(self.vol_names)


    def load_volumes(self):
        self.volume_1 = []
        self.volume_2 = []

        for vname in tqdm(self.vol_names):
            vol_1 = torch.load(os.path.join(self.path_1, vname)).moveaxis(0,2)
            vol_2 = torch.load(os.path.join(self.path_2, vname)).moveaxis(0,2)
            self.volume_1.append(vol_1)
            self.volume_2.append(vol_2)
            

    def __getitem__(self,idx):

        if self.preload:

            vol_1 = self.volume_1[idx]#.clone()
            vol_2 = self.volume_2[idx]#.clone()

        else:
            vol_name = self.vol_names[idx]
    
            vol_1 = torch.load(os.path.join(self.path_1, vol_name)).moveaxis(0,2)
            vol_2 = torch.load(os.path.join(self.path_2, vol_name)).moveaxis(0,2)

        data = {'vol1':vol_1, 'vol2':vol_2}
        return data
        