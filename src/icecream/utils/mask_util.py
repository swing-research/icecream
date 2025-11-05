"""
Functions to generate masks for a given volume

This code is modified from the isonet codebase:https://github.com/IsoNet-cryoET/IsoNet/blob/master/bin/make_mask.py#L28
"""
import numpy as np
from scipy.signal import convolve
from skimage.transform import resize
from scipy.ndimage.filters import gaussian_filter,maximum_filter

def make_mask(tomo, mask_boundary = None, side = 5, density_percentage=50., std_percentage=50., surface=None):

    sp=np.array(tomo.shape)
    sp2 = sp//2
    bintomo = resize(tomo,sp2,anti_aliasing=True)
  
    gauss = gaussian_filter(bintomo, side/2)
    if density_percentage <=99.8:
        mask1 = maxmask(gauss,side=side, percentile=density_percentage)
    else:
        mask1 = np.ones(sp2)

    if std_percentage <=99.8:
        mask2 = stdmask(gauss,side=side, threshold=std_percentage)
    else:
        mask2 = np.ones(sp2)

    out_mask_bin = np.multiply(mask1,mask2)
   
    if mask_boundary is not None:
        from IsoNet.util.filter import boundary_mask
        mask3 = boundary_mask(bintomo, mask_boundary)
        out_mask_bin = np.multiply(out_mask_bin, mask3)

    if (surface is not None) and surface < 1:
        for i in range(int(surface*sp2[0])):
            out_mask_bin[i] = 0
        for i in range(int((1-surface)*sp2[0]),sp2[0]):
            out_mask_bin[i] = 0


    out_mask = np.zeros(sp)
    out_mask[0:-1:2,0:-1:2,0:-1:2] = out_mask_bin
    out_mask[0:-1:2,0:-1:2,1::2] = out_mask_bin
    out_mask[0:-1:2,1::2,0:-1:2] = out_mask_bin
    out_mask[0:-1:2,1::2,1::2] = out_mask_bin
    out_mask[1::2,0:-1:2,0:-1:2] = out_mask_bin
    out_mask[1::2,0:-1:2,1::2] = out_mask_bin
    out_mask[1::2,1::2,0:-1:2] = out_mask_bin
    out_mask[1::2,1::2,1::2] = out_mask_bin
    out_mask = (out_mask>0.5).astype(np.uint8)


    return out_mask


def maxmask(tomo, side=5,percentile=60):
    
    # print('maximum_filter')
    filtered = maximum_filter(-tomo, 2*side+1, mode='reflect')
    out =  filtered > np.percentile(filtered,100-percentile)
    out = out.astype(np.uint8)
    return out



def stdmask(tomo,side=10,threshold=60):
    
    # print('std_filter')
    tomosq = tomo**2
    ones = np.ones(tomo.shape)
    kernel = np.ones((2*side+1, 2*side+1, 2*side+1))
    s = convolve(tomo, kernel, mode="same")
    s2 = convolve(tomosq, kernel, mode="same")
    ns = convolve(ones, kernel, mode="same")
    out = (s2 - s**2 / ns) / ns
    out[out < 0] = 0
    out = np.sqrt(out)
    # out = out>np.std(tomo)*threshold
    out  = out>np.percentile(out, 100-threshold)
    return out.astype(np.uint8)