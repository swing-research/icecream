import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from icecream.utils.utils import get_measurement
from skimage.filters import window as skimage_window

from tqdm import tqdm
import numpy as np
def compute_padd(N, filt_size, stride):
    w = (N - filt_size) // stride +1
    N_rec = (w-1)*stride + filt_size
    padd = N_rec - N
    return padd%filt_size



def generate_crops(vol_1, vol_2, size, stride):

    device = vol_1.device

    pad_i = compute_padd(vol_1.shape[0],size,stride)
    pad_j = compute_padd(vol_1.shape[1],size,stride)
    pad_k = compute_padd(vol_1.shape[2],size,stride)

    vol_1_pad = torch.nn.functional.pad(vol_1, (0,pad_k,0,pad_j,0,pad_i))
    vol_2_pad = torch.nn.functional.pad(vol_2, (0,pad_k,0,pad_j,0,pad_i))

    N1 = vol_1_pad.shape[0]
    N2 = vol_1_pad.shape[1]
    N3 = vol_1_pad.shape[2]


    crops_1 = []
    crops_2 = []
    with torch.no_grad():
        for i in range(0, N1 , stride):
            for j in range(0, N2 , stride):
                for k in range(0, N3 , stride):
                    if(i+size >  N1 or j+size > N2 or k+size > N3):
                            continue
                    crop_1 = vol_1_pad[i:i+size,j:j+size,k:k+size]
                    crop_2 = vol_2_pad[i:i+size,j:j+size,k:k+size]
                    crops_1.append(crop_1)
                    crops_2.append(crop_2)
    crops_1 = torch.stack(crops_1)
    crops_2 = torch.stack(crops_2)

    return crops_1, crops_2


def crop_volumes(volume, stride,size, upsampled=False, window_inp=None):
    N1,N2,N3 = volume.shape

    crops = []
    with torch.no_grad():
        for i in range(0, N1 , stride):
            for j in range(0, N2 , stride):
                for k in range(0, N3 , stride):
                    if(i+size >  N1 or j+size > N2 or k+size > N3):
                            continue
                    crop = volume[i:i+size,j:j+size,k:k+size]
                    if window_inp is not None:
                        crop = crop * window_inp
                    if upsampled:
                        crop = downsample_crop(crop[None])[0]
                    crops.append(crop.cpu())   

    crops = torch.stack(crops).cpu()
    return crops 

def downsample_crop(crops):

    b,n1,n2,n3 = crops.shape

    crops_fft = torch.fft.fftshift(torch.fft.fftn(crops, dim=(-3, -2, -1)))

    crops_fft = crops_fft[:,n1//2-n1//4:n1//2+n1//4,
                            n2//2-n2//4:n2//2+n2//4,
                            n3//2-n3//4:n3//2+n3//4]
    
    crops_downsampled = torch.fft.ifftn(torch.fft.ifftshift(crops_fft), dim=(-3, -2, -1)).real
    return crops_downsampled



def inference(vol_input, model, size, stride, batch_size, 
              window=None, 
              window_type=None,
              run_multi=None,
              wedge= None, 
              update_missing_wedge = False,
              pre_pad = False,
              pre_pad_size = 0,
              device = None,
              upsampled_=False,
              avg_pool = False):
    """
    Function to run the inference on the volume

    run_double: The is run twice on each crops 
    update_missing_wdege: If True, only the missing wedge of the volume is updated
    """

    if wedge is not None:
        print('Using wedge')
    else:
        print('Not using wedge')

    if upsampled_:
        pre_pad_size= pre_pad_size*2

    vol_fbp = vol_input
    if pre_pad:
        vol_fbp =torch.nn.functional.pad(vol_fbp, (pre_pad_size,0,pre_pad_size,0,pre_pad_size,0))

    if device is None:
        device = vol_fbp.device
    else:
        vol_fbp = vol_fbp.to(device)

    # if window_type is not None:
    #     window_op = torch.tensor(skimage_window(window_type, (size,size,size)), dtype=torch.float32, device=device)
    # else:
    #     window_op = torch.ones((size,size,size), device=device)

    window_op = torch.ones((size,size,size), device=device)
    if upsampled_:
        size = size*2
        stride = stride*2

    # if window_type is not None:
    #     window_inp = torch.tensor(skimage_window(window_type, (size,size,size)), dtype=torch.float32, device=device)
    # else:
    #     window_inp = torch.ones((size,size,size), device=device)

    window_inp = torch.ones((size,size,size), device=device)


    pad_i = compute_padd(vol_fbp.shape[0],size,stride)
    pad_j = compute_padd(vol_fbp.shape[1],size,stride)
    pad_k = compute_padd(vol_fbp.shape[2],size,stride)


    vol_fbp_pad = torch.nn.functional.pad(vol_fbp, (0,pad_k,0,pad_j,0,pad_i))

    N1 = vol_fbp_pad.shape[0]
    N2 = vol_fbp_pad.shape[1]
    N3 = vol_fbp_pad.shape[2]
    crops = []

    # low pass filter the volume to mimic the interpolation used while training
    if avg_pool:
        vol_fbp_pad = F.avg_pool3d(vol_fbp_pad[None,None], kernel_size=3, stride=1, padding=1)[0,0]
    

    with torch.no_grad():
        crops = crop_volumes(vol_fbp_pad, stride, size, upsampled=upsampled_, window_inp = window_inp)


    crop_loader = DataLoader(crops, batch_size=batch_size, shuffle=False)


    output_crops = []

    if wedge is not None:
        wedge_complement = 1 - wedge
    with torch.no_grad():
        model.eval()
        for crop in tqdm(crop_loader,leave=False):
            crop = crop.to(device)
            output = model(crop[:,None])[:,0]
            output = get_measurement(output, wedge)
            output = model(output[:,None])[:,0]
            
            output_crop = output
            output_crops.append(output_crop.detach().cpu())
    output_crops = torch.cat(output_crops, dim=0)#.to(device)


    N1,N2,N3 = vol_input.shape

    if upsampled_:
        N1_true = N1// 2
        N2_true = N2// 2
        N3_true = N3// 2


        size = size//2
        stride = stride//2

        pre_pad_size = pre_pad_size//2
    else:
        N1_true = N1
        N2_true = N2
        N3_true = N3

    vol_est= torch.zeros((N1_true,N2_true,N3_true), device=device)
    if pre_pad:
        vol_est =torch.nn.functional.pad(vol_est, (pre_pad_size,0,pre_pad_size,0,pre_pad_size,0))


    N1_pad, N2_pad, N3_pad = vol_est.shape

    pad_i = compute_padd(vol_est.shape[0],size,stride)
    pad_j = compute_padd(vol_est.shape[1],size,stride)
    pad_k = compute_padd(vol_est.shape[2],size,stride)





    
    vol_est = torch.nn.functional.pad(vol_est, (0,pad_k,0,pad_j,0,pad_i))

    N1, N2, N3 = vol_est.shape
    mask = torch.zeros_like(vol_est)


    if window is None:
        window = torch.ones((size,size,size), device=device)

    count = 0
    for i in range(0, N1 , stride):
        for j in range(0, N2 , stride):
            for k in range(0, N3 , stride):
                if(i+size >  N1 or j+size > N2 or k+size > N3):
                        continue
                vol_est[i:i+size,j:j+size,k:k+size] += output_crops[count].to(device)*window
                mask[i:i+size,j:j+size,k:k+size] += window
                count += 1

    vol_est = vol_est / mask


    vol_est = vol_est[:N1_pad,:N2_pad,:N3_pad]

    vol_est_np = vol_est.cpu().numpy()

    if pre_pad:
        vol_est_np = vol_est_np[pre_pad_size:,pre_pad_size:,pre_pad_size:]
    vol_inp_np = None
    return vol_est_np, vol_inp_np


def initialize__prediction_window(crop_size, lnt = None):
    w = np.zeros((crop_size,crop_size,crop_size))
    mid = crop_size//2
    if lnt is None:
        lnt = crop_size//8
    w[mid -lnt:mid+lnt,mid -lnt:mid+lnt,mid -lnt:mid+lnt] = 1
    w_t = torch.tensor(w, dtype=torch.float32)     
    return w_t  


def inference_direct(vol_input, model, size, stride, batch_size, 
              window=None, 
              run_multi=None,
              wedge= None, 
              update_missing_wedge = False,
              pre_pad = False,
              pre_pad_size = 0,
              avg_pool = False):
    """
    Function to run the inference on the volume

    run_double: The is run twice on each crops 
    update_missing_wdege: If True, only the missing wedge of the volume is updated
    """

    if wedge is not None:
        print('Using wedge')
    else:
        print('Not using wedge')

    if pre_pad:
        vol_input =torch.nn.functional.pad(vol_input, (pre_pad_size,0,pre_pad_size,0,pre_pad_size,0))


    vol_fbp = vol_input
    device = vol_fbp.device
    pad_i = compute_padd(vol_fbp.shape[0],size,stride)
    pad_j = compute_padd(vol_fbp.shape[1],size,stride)
    pad_k = compute_padd(vol_fbp.shape[2],size,stride)


    vol_fbp_pad = torch.nn.functional.pad(vol_fbp, (0,pad_k,0,pad_j,0,pad_i))

    N1 = vol_fbp_pad.shape[0]
    N2 = vol_fbp_pad.shape[1]
    N3 = vol_fbp_pad.shape[2]
    

    # low pass filter the volume to mimic the interpolation used while training
    if avg_pool:
        vol_fbp_pad = F.avg_pool3d(vol_fbp_pad[None,None], kernel_size=3, stride=1, padding=1)[0,0]
    
    if wedge is not None:
        wedge_complement = 1 - wedge

    vol_est= torch.zeros_like(vol_fbp_pad)
    vol_inp = torch.zeros_like(vol_fbp_pad)
    mask = torch.zeros_like(vol_fbp_pad)
    model.eval()
    if window is None:
        window = torch.ones((size,size,size), device=device)
    with torch.no_grad():
        for i in tqdm(range(0, N1 , stride)):
            for j in tqdm(range(0, N2 , stride), leave=False):
                crops = []
                for k in range(0, N3 , stride):
                    if(i+size >  N1 or j+size > N2 or k+size > N3):
                            continue
                    crop = vol_fbp_pad[i:i+size,j:j+size,k:k+size]
                    crops.append(crop.cpu())


                if len(crops) == 0:
                    continue

                crops = torch.stack(crops).cpu()
                crop_loader = DataLoader(crops, batch_size=batch_size, shuffle=False)

                output_crops = []  
                for crop in tqdm(crop_loader,leave=False):
                    crop = crop.to(device)

                    if run_multi is None: 
                        if update_missing_wedge is False:
                            if wedge is not None:
                                crop = get_measurement(crop, wedge)
                        output = model(crop[:,None])[:,0]

                        if update_missing_wedge:
                            op_missing_wedge = get_measurement(output, wedge_complement)
                            output = crop + op_missing_wedge
                    else:
                        if update_missing_wedge is False:
                            if wedge is not None:
                                crop = get_measurement(crop, wedge)

                        op_temp = model(crop[:,None])

                        if update_missing_wedge:
                            op_missing_wedge = get_measurement(op_temp[:,0], wedge_complement)
                            op_temp = crop[:,None] + op_missing_wedge[:,None]
                        for i in range(run_multi):
                            #print('running)')
                            if update_missing_wedge is False:
                                if wedge is not None:
                                    crop = get_measurement(crop, wedge)
                            
                            op_temp = model(op_temp)

                            if update_missing_wedge:
                                op_missing_wedge = get_measurement(op_temp[:,0], wedge_complement)
                                op_temp = crop[:,None] + op_missing_wedge[:,None]

                        output = op_temp[:,0]
                    output_crop = output
                    output_crops.append(output_crop.detach().cpu())
                output_crops = torch.cat(output_crops, dim=0)#.to(device)
                count = 0
                for k in range(0, N3 , stride):
                    if(i+size >  N1 or j+size > N2 or k+size > N3):
                            continue
                    vol_est[i:i+size,j:j+size,k:k+size] += output_crops[count].to(device)*window
                    vol_inp[i:i+size,j:j+size,k:k+size] += crops[count].to(device)*window
                    mask[i:i+size,j:j+size,k:k+size] += window
                    count += 1


    mask[mask == 0] = 1  # Avoid division by zero
    vol_est = vol_est / mask
    vol_inp = vol_inp / mask


    vol_est = vol_est[:vol_fbp.shape[0],:vol_fbp.shape[1],:vol_fbp.shape[2]]
    vol_inp = vol_inp[:vol_fbp.shape[0],:vol_fbp.shape[1],:vol_fbp.shape[2]]

    vol_est_np = vol_est.cpu().numpy()
    vol_inp_np = vol_inp.cpu().numpy()

    if pre_pad:
        vol_est_np = vol_est_np[pre_pad_size:,pre_pad_size:,pre_pad_size:]
        vol_inp_np = vol_inp_np[pre_pad_size:,pre_pad_size:,pre_pad_size:]

    return vol_est_np, vol_inp_np

def inference_rots(vol_input, model, size, stride, batch_size, 
              window=None, 
              run_multi=None,
              wedge= None, 
              k_index_set = None,
              pre_pad = False,
              pre_pad_size = 0):
    """
    Function to run the inference on the volume

    run_double: The is run twice on each crops 
    """

    if wedge is not None:
        print('Using wedge')
    else:
        print('Not using wedge')

    if pre_pad:
        vol_input =torch.nn.functional.pad(vol_input, (pre_pad_size,0,pre_pad_size,0,pre_pad_size,0))


    vol_fbp = vol_input
    device = vol_fbp.device
    pad_i = compute_padd(vol_fbp.shape[0],size,stride)
    pad_j = compute_padd(vol_fbp.shape[1],size,stride)
    pad_k = compute_padd(vol_fbp.shape[2],size,stride)


    vol_fbp_pad = torch.nn.functional.pad(vol_fbp, (0,pad_k,0,pad_j,0,pad_i))

    N1 = vol_fbp_pad.shape[0]
    N2 = vol_fbp_pad.shape[1]
    N3 = vol_fbp_pad.shape[2]
    crops = []
    with torch.no_grad():
        for i in range(0, N1 , stride):
            for j in range(0, N2 , stride):
                for k in range(0, N3 , stride):
                    if(i+size >  N1 or j+size > N2 or k+size > N3):
                            continue
                    crop = vol_fbp_pad[i:i+size,j:j+size,k:k+size]
                    crops.append(crop)

    crops = torch.stack(crops).cpu()


    crop_loader = DataLoader(crops, batch_size=batch_size, shuffle=False)


    output_crops_avg = []
    output_crops_std = []
    with torch.no_grad():
        model.eval()
        for crop in tqdm(crop_loader,leave=False):
            crop = crop.to(device)

            if run_multi is None: 

                if wedge is not None:
                    crop = get_measurement(crop, wedge)

                if k_index_set is not None:
                    rots_op = []
                    for k_index in k_index_set:
                        crop_rot = generate_choosen_rotation_batch(crop, k_index)
                        output = model(crop_rot[:,None])[:,0].detach()
                        output = generate_choosen_rotation_batch_inv(output, k_index)
                        rots_op.append(output)
                    rots_op = torch.stack(rots_op, dim=0)
                    output_avg = rots_op.mean(dim=0).cpu()
                    output_std = rots_op.std(dim=0).cpu()
                else:
                    output = model(crop[:,None])[:,0]
                    output_avg = output.cpu()
                    output_std = torch.zeros_like(output_avg)
            else:
                if wedge is not None:
                    crop = get_measurement(crop, wedge)

                rots_op = []
                for k_index in k_index_set:
                    crop_rot = generate_choosen_rotation_batch(crop, k_index)
                    op_temp = model(crop_rot[:,None])
                    for i in range(run_multi):
                        op_temp = model(op_temp)

                    output = op_temp[:,0].detach()
                    output = generate_choosen_rotation_batch_inv(output, k_index)
                    rots_op.append(output)
                
                rots_op = torch.stack(rots_op, dim=0)
                output_avg = rots_op.mean(dim=0).cpu()
                output_std = rots_op.std(dim=0).cpu()


            
            output_crops_avg.append(output_avg)
            output_crops_std.append(output_std)
    output_crops_avg = torch.cat(output_crops_avg, dim=0)#.to(device)
    output_crops_std = torch.cat(output_crops_std, dim=0)#.to(device)


    vol_est= torch.zeros_like(vol_fbp_pad)
    vol_est_std = torch.zeros_like(vol_fbp_pad)
    mask = torch.zeros_like(vol_fbp_pad)
    #vol_inp = torch.zeros_like(vol_fbp_pad)


    if window is None:
        window = torch.ones((size,size,size), device=device)

    count = 0
    for i in range(0, N1 , stride):
        for j in range(0, N2 , stride):
            for k in range(0, N3 , stride):
                if(i+size >  N1 or j+size > N2 or k+size > N3):
                        continue
                vol_est[i:i+size,j:j+size,k:k+size] += output_crops_avg[count].to(device)*window
                vol_est_std[i:i+size,j:j+size,k:k+size] += output_crops_std[count].to(device)*window
                mask[i:i+size,j:j+size,k:k+size] += window
                #vol_inp[i:i+size,j:j+size,k:k+size] += crops[count].to(device)*window
                count += 1

    vol_est = vol_est / mask
    vol_est_std = vol_est_std / mask


    vol_est = vol_est[:vol_fbp.shape[0],:vol_fbp.shape[1],:vol_fbp.shape[2]]
    vol_est_std = vol_est_std[:vol_fbp.shape[0],:vol_fbp.shape[1],:vol_fbp.shape[2]]

    vol_est_np = vol_est.cpu().numpy()
    vol_est_std_np = vol_est_std.cpu().numpy()

    if pre_pad:
        vol_est_np = vol_est_np[pre_pad_size:,pre_pad_size:,pre_pad_size:]
        vol_est_std_np = vol_est_std_np[pre_pad_size:,pre_pad_size:,pre_pad_size:]

    return vol_est_np, vol_est_std_np

def generate_choosen_rotation_batch(vol, k_index = 0):
    rotated = torch.zeros_like(vol)
    for i in range(vol.shape[0]):
        rotated[i] = generate_choosen_rotations(vol[i], k_index)
    return rotated

def generate_choosen_rotation_batch_inv(vol, k_index = 0):
    rotated = torch.zeros_like(vol)
    for i in range(vol.shape[0]):
        rotated[i] = generate_choosen_inv_rotations(vol[i], k_index)
    return rotated


def generate_choosen_rotations(vol, k_index = 0):

    kset = [[0, 0, 1, -1],
            [0, 0, 3, -1],
            [0, 1, 0, -1],
            [0, 1, 1, -1],
            [0, 1, 2, -1],
            [0, 1, 3, -1],
            [0, 2, 1, -1],
            [0, 2, 3, -1],
            [0, 3, 0, -1],
            [0, 3, 1, -1],
            [0, 3, 2, -1],
            [0, 3, 3, -1],
            [1, 0, 0, -1],
            [1, 0, 1, -1],
            [1, 0, 2, -1],
            [1, 0, 3, -1],
            [1, 2, 0, -1],
            [1, 2, 1, -1],
            [1, 2, 2, -1],
            [1, 2, 3, -1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 2],
            [0, 0, 3, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 2],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 2],
            [0, 1, 2, 1],
            [0, 1, 3, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 2],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 0, 1, 2],
            [1, 0, 2, 1],
            [1, 0, 3, 1],
            ]

    k = kset[k_index]

    kx, ky, kz, axis = k

    rotated = torch.rot90(vol, k=kx, dims=(1, 2))
    rotated = torch.rot90(rotated, k=ky, dims=(0, 2))
    rotated = torch.rot90(rotated, k=kz, dims=(0, 1))




    if axis != -1:
        rotated = torch.flip(rotated, [axis])


    return rotated

def generate_choosen_inv_rotations(vol, k_index = 0):

    kset = [[0, 0, 1, -1],
            [0, 0, 3, -1],
            [0, 1, 0, -1],
            [0, 1, 1, -1],
            [0, 1, 2, -1],
            [0, 1, 3, -1],
            [0, 2, 1, -1],
            [0, 2, 3, -1],
            [0, 3, 0, -1],
            [0, 3, 1, -1],
            [0, 3, 2, -1],
            [0, 3, 3, -1],
            [1, 0, 0, -1],
            [1, 0, 1, -1],
            [1, 0, 2, -1],
            [1, 0, 3, -1],
            [1, 2, 0, -1],
            [1, 2, 1, -1],
            [1, 2, 2, -1],
            [1, 2, 3, -1],
            [0, 0, 1, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 2],
            [0, 0, 3, 1],
            [0, 1, 0, 0],
            [0, 1, 0, 1],
            [0, 1, 0, 2],
            [0, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 1, 1, 2],
            [0, 1, 2, 1],
            [0, 1, 3, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 2],
            [1, 0, 1, 0],
            [1, 0, 1, 1],
            [1, 0, 1, 2],
            [1, 0, 2, 1],
            [1, 0, 3, 1],
            ]

    k = kset[k_index]

    kx, ky, kz, axis = k
    if axis != -1:
        rotated = torch.flip(vol, [axis])
    else:
        rotated = vol.clone()


    rotated = torch.rot90(rotated, k=-kz, dims=(0, 1))
    rotated = torch.rot90(rotated, k=-ky, dims=(0, 2))
    rotated = torch.rot90(rotated, k=-kx, dims=(1, 2))
    
    







    return rotated
