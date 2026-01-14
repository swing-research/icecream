import mrcfile
import numpy as np
import torch    
from icecream.utils.mask_util import make_mask
def normalize_volume(vol):
    return (vol - vol.mean()) / (vol.std() + 1e-8)

def load_volume(vol_path):
    """
    Load a single volume from the given path to the cpu.
    Args:
        vol_path (str): Path to the volume file.
    Returns:
        torch.Tensor: Loaded volume as a tensor.
    """
    vol = mrcfile.open(vol_path).data
    vol = np.moveaxis(vol, 0, 2).astype(np.float32)
    vol_t = torch.tensor(vol, dtype=torch.float32, device='cpu')
    return normalize_volume(vol_t)



def load_data( vol_paths_1, 
              vol_paths_2, 
              vol_mask_path=None, 
              use_mask=False,
              load_device=False,
              device='cpu',
              mask_frac=0.3, 
              mask_tomo_side=5, 
              mask_tomo_density_perc=50.,
              mask_tomo_std_perc=50.):
    """
    Load data from the given volume paths.
    Args:
        vol_paths_1 (list): List of paths to the first set of volumes.
        vol_paths_2 (list): List of paths to the second set of volumes.
        vol_mask_path (str, optional): Path to the volume mask. Defaults to None.
    """
    print(vol_paths_1)
    print(vol_paths_2)

    n_volumes = len(vol_paths_1)

    if len(vol_paths_1) != len(vol_paths_2) and len(vol_paths_2) != 0:
        raise ValueError("The number of volume paths for vol_paths_1 and vol_paths_2 must be the same.")

    # Load and store all the volume in a list on the CPU. Probably sub-optimal but enough at the moment.
    vol_1_set = []
    vol_2_set = []
    vol_mask_set = []
    for i in range(len(vol_paths_1)):
        vol_1_t = load_volume(vol_paths_1[i])
        if len(vol_paths_2) != 0:
            vol_2_t = load_volume(vol_paths_2[i])
            print(f"Loaded volumes: \n {vol_paths_1[i]}\n and\n {vol_paths_2[i]}")
            print(f"They have shape (x,y,z): {list(vol_1_t.shape)} and {list(vol_2_t.shape)}.")
        else:
            print(f"Loaded volume: \n {vol_paths_1[i]}")
            print(f"It has shape (x,y,z): {list(vol_1_t.shape)}.")

        if vol_mask_path is not None:
            print(f"Loading tomogram mask: \n {vol_mask_path[i]}")
            vol_mask = mrcfile.open(vol_mask_path[i]).data
            vol_mask = np.moveaxis(vol_mask, 0, 2).astype(np.float32)
            vol_mask_t = torch.tensor(vol_mask, dtype=torch.float32, device='cpu')
        else:
            if use_mask:
                if len(vol_paths_2) != 0:
                    vol_avg = ((vol_1_t + vol_2_t) / 2).numpy()
                else:
                    vol_avg = ((vol_1_t) / 2).numpy()
                vol_mask = make_mask(vol_avg, mask_boundary=None, side=mask_tomo_side, density_percentage=mask_tomo_density_perc, std_percentage=mask_tomo_std_perc)
                vol_mask_t = torch.tensor(vol_mask, dtype=torch.float32, device='cpu')
            else:
                vol_mask_t = None
                mask_frac = 0.0
        if load_device:
            vol_1_set.append(vol_1_t.to(device))
            if len(vol_paths_2) != 0:
                vol_2_set.append(vol_2_t.to(device))
        else:
            vol_1_set.append(vol_1_t.cpu().share_memory_())
            if len(vol_paths_2) != 0:
                vol_2_set.append(vol_2_t.cpu().share_memory_())
        if vol_mask_t is not None:
            vol_mask_set.append(vol_mask_t.cpu().share_memory_())

    return vol_1_set, vol_2_set, vol_mask_set,mask_frac