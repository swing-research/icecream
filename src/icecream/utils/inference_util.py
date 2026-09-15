import torch
import torch.nn.functional as F
from icecream.utils.utils import get_measurement
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
              avg_pool = False,
              run_double = True,
              blend = 'hann'):
    """
    Function to run the inference on the volume

    run_double: The is run twice on each crops 
    blend: overlap window used when `window` is None: 'hann' (separable Hann over the whole
           tile, floored at 1e-3 so the weights never vanish) or 'ones' (plain average).
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
    pad_i = compute_padd(vol_fbp.shape[0],size,stride)
    pad_j = compute_padd(vol_fbp.shape[1],size,stride)
    pad_k = compute_padd(vol_fbp.shape[2],size,stride)
    vol_fbp_pad = torch.nn.functional.pad(vol_fbp, (0,pad_k,0,pad_j,0,pad_i))

    # low pass filter the volume to mimic the interpolation used while training
    if avg_pool:
        vol_fbp_pad = F.avg_pool3d(vol_fbp_pad[None,None], kernel_size=3, stride=1, padding=1)[0,0]
    # Output grid: with `upsampled_` the network sees `size` voxels of the 2x volume and
    # returns tiles at the original resolution, so the estimate is accumulated at size // s.
    s = 2 if upsampled_ else 1
    N1, N2, N3 = vol_fbp_pad.shape
    out_size = size // s
    if window is None:
        if blend == 'hann':
            w1 = torch.hann_window(out_size, periodic=False).clamp_min(1e-3)
            window = w1[:, None, None] * w1[None, :, None] * w1[None, None, :]
        else:
            window = torch.ones((out_size, out_size, out_size))
    window = window.to(device=device, dtype=torch.float32)
    vol_est = torch.zeros((N1 // s, N2 // s, N3 // s), device=device)
    weight = torch.zeros_like(vol_est)
    starts = [(i, j, k) for i in range(0, N1 - size + 1, stride)
                        for j in range(0, N2 - size + 1, stride)
                        for k in range(0, N3 - size + 1, stride)]

    with torch.no_grad():
        model.eval()
        for b in tqdm(range(0, len(starts), batch_size), leave=False):
            batch = starts[b:b + batch_size]
            crop = torch.stack([vol_fbp_pad[i:i+size, j:j+size, k:k+size] for i, j, k in batch])
            if upsampled_:
                crop = downsample_crop(crop)
            with torch.autocast(device_type='cuda', dtype=torch.float16):
                output = model(crop[:,None])[:,0]
            # Convert to float32
            output = output.float()
            if run_double:  # second application on the wedge-masked first estimate (paper Eq. 5)
                output = get_measurement(output, wedge)
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(output[:,None])[:,0]
            output = output.float() * window
            for n, (i, j, k) in enumerate(batch):
                i, j, k = i // s, j // s, k // s
                vol_est[i:i+out_size, j:j+out_size, k:k+out_size] += output[n]
                weight[i:i+out_size, j:j+out_size, k:k+out_size] += window

    weight[weight == 0] = 1
    vol_est = (vol_est / weight).cpu()
    # crop back to the input volume: pre-padding at the low end, tail padding at the high end
    p = pre_pad_size // s if pre_pad else 0
    n1, n2, n3 = (n // s for n in vol_input.shape)
    vol_est_np = vol_est[p:p+n1, p:p+n2, p:p+n3].numpy()
    vol_inp_np = None

    del vol_fbp_pad, vol_est, weight
    torch.cuda.empty_cache()

    return vol_est_np, vol_inp_np

