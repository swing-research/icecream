import torch
import torch.nn.functional as F
from icecream.utils.utils import get_measurement
from torch.utils.data import DataLoader
from tqdm import tqdm


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

def crop_volumes(volume, stride, size, upsampled=False, window_inp=None):
    N1,N2,N3 = volume.shape

    n1 = (N1 - size) // stride + 1
    n2 = (N2 - size) // stride + 1
    n3 = (N3 - size) // stride + 1
    num_crops = n1 * n2 * n3

    if window_inp is not None:
        window_inp = window_inp.to(device=volume.device, dtype=volume.dtype)

    out = torch.empty(
            (num_crops, size, size, size),
            dtype=volume.dtype,
            device='cpu',
        )
    idx = 0
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
                    crop = crop.to('cpu')
                    out[idx] = crop
                    idx += 1
    # remove the extra allocated space
    crops = out[:idx]
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
              wedge= None,
              pre_pad = False,
              pre_pad_size = 0,
              device = None,
              upsampled_= False,
              avg_pool = False):
    """
    Function to run the inference on the volume

    run_double: The is run twice on each crops 
    update_missing_wdege: If True, only the missing wedge of the volume is updated
    """

    #ipdb.set_trace()
    if wedge is not None:
        print('Using wedge')
    else:
        print('Not using wedge')
    if upsampled_:
        pre_pad_size= pre_pad_size*2
    vol_fbp = vol_input.to(torch.float16)
    if pre_pad:
        vol_fbp =torch.nn.functional.pad(vol_fbp, (pre_pad_size,0,pre_pad_size,0,pre_pad_size,0))
    if device is None:
        device = vol_fbp.device
    else:
        vol_fbp = vol_fbp.to(device)
        if wedge is not None:
            wedge = wedge.to(device)
    if upsampled_:
        size = size*2
        stride = stride*2
    window_inp = torch.ones((size,size,size), device=device)
    pad_i = compute_padd(vol_fbp.shape[0],size,stride)
    pad_j = compute_padd(vol_fbp.shape[1],size,stride)
    pad_k = compute_padd(vol_fbp.shape[2],size,stride)
    vol_fbp_pad = torch.nn.functional.pad(vol_fbp, (0,pad_k,0,pad_j,0,pad_i))

    # low pass filter the volume to mimic the interpolation used while training
    if avg_pool:
        vol_fbp_pad = F.avg_pool3d(vol_fbp_pad[None,None], kernel_size=3, stride=1, padding=1)[0,0]
    with torch.no_grad():
        crops = crop_volumes(vol_fbp_pad, stride, size, upsampled=upsampled_, window_inp = window_inp)
    crop_loader = DataLoader(crops, batch_size=batch_size, shuffle=False)
    output_crops = torch.zeros_like(crops)

    with torch.no_grad():
        model.eval()
        idx = 0
        for crop in tqdm(crop_loader,leave=False):
            crop = crop.to(device)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(crop[:,None])[:,0]
            # Convert to float32
            output = output.float()
            output = get_measurement(output, wedge)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(output[:,None])[:,0]
            # Convert to float32
            output_crop = output

            batch_size_curr = output_crop.shape[0]
            output_crops[idx:idx+batch_size_curr] = output_crop.detach().cpu()
            idx += batch_size_curr

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

    vol_est= torch.zeros((N1_true,N2_true,N3_true))
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
        window = torch.ones((size,size,size))
    else:
        window = window.cpu()
    count = 0
    for i in range(0, N1 , stride):
        for j in range(0, N2 , stride):
            for k in range(0, N3 , stride):
                if(i+size >  N1 or j+size > N2 or k+size > N3):
                        continue
                vol_est[i:i+size,j:j+size,k:k+size] += output_crops[count]*window
                mask[i:i+size,j:j+size,k:k+size] += window
                count += 1
    mask[mask==0] = 1
    vol_est = vol_est / mask
    vol_est = vol_est[:N1_pad,:N2_pad,:N3_pad]
    vol_est_np = vol_est.numpy()
    
    if pre_pad:
        vol_est_np = vol_est_np[pre_pad_size:,pre_pad_size:,pre_pad_size:]
    vol_inp_np = None
    
    del vol_fbp_pad 
    del output_crops
    torch.cuda.empty_cache()

    return vol_est_np, vol_inp_np

