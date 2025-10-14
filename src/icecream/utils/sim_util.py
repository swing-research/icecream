import tomosipo as ts
from ts_algorithms import fbp
import torch

def generate_projections(vol,angles):
    # note angles in randians
    # n1 = vol.shape[0]
    # n2 = vol.shape[1]
    # n3 = vol.shape[2]

    # # arranging the volume to get the same projections as the matlab simulator
    # vol_swap = torch.zeros_like(vol)

    # for i in range(vol.shape[2]):
    #     vol_swap[:,:,i]  = vol[:,:,-i].clone()
    # # vol_swap = vol[:,:,::-1] ?

    # operator =  ParallelBeamGeometry3DOpAngles_rectangular((n1,n2,n3), angles, op_snr=np.inf, fact=1)
    # proj = operator(vol)

    # Using tomosip implementation
    n1,n2,n3 = vol.shape
    pg = ts.parallel(angles = angles, shape =(n1,n2),)
    vg = ts.volume(shape=(n1,n3,n2))  # Reordering so that this is samle as ODL
    A = ts.operator(vg, pg)
    projection = A(vol.permute(0,2,1))
    return projection.permute(1,0,2)


def reconstruct_FBP_volume(projections, angles, n3):
    # Define the forward operator
    # angels in radians
    n1 = projections.shape[1]
    n2 = projections.shape[2]
    pg = ts.parallel(angles = angles, shape =(n1,n2),)
    vg = ts.volume(shape=(n1,n3,n2))  # Reordering so that this is samle as ODL
    A = ts.operator(vg, pg)
    V_FBP = fbp(A, projections.permute(1,0,2)).permute(0,2,1)
    return V_FBP

def find_sigma_noise(SNR_value,x_ref):
    nref = torch.mean(x_ref**2)
    sigma_noise = (10**(-SNR_value/10)) * nref
    return torch.sqrt(sigma_noise)