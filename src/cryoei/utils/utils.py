
import numpy as np
from skimage.transform import rotate as skrotate
import torch
from itertools import permutations, product
from math import sqrt
from scipy.spatial.transform import Rotation as R

def get_wedge(size, max_angle, min_angle, rotation = 0,radius=10):
    """
    The wedge is a 2D array of size (size,size) with a wedge of angle max_angle-min_angle
    """
    size = 2*size
    if isinstance(size, int):
        size = (size,size)
    wedge = np.zeros(size)
    x = np.linspace(-1,1,size[0])
    y = np.linspace(-1,1,size[1])
    xx, yy = np.meshgrid(x, y)

    wedge[xx**2 + yy**2 < radius] = 1


    wedge[yy >np.tan(np.deg2rad(max_angle)) * xx] = 0
    wedge[yy < np.tan(np.deg2rad(min_angle)) * xx] = 0

    wedge_flip = np.fliplr(wedge)

    wedge = wedge + wedge_flip

    wedge = skrotate(wedge, float(rotation), resize=False)
    wedge[wedge < 0.5] = 0
    wedge[wedge >= 0.5] = 1    

    # crop the wedge to the original size

    wedge = wedge[int(size[0]/4):int(3*size[0]/4),int(size[1]/4):int(3*size[1]/4)]

    return wedge


def get_wedge_3d(size,max_angle, 
                 min_angle, 
                 rotation = -30, 
                 low_support = 0,
                 use_spherical_support = False):
    """
    Get 3D wedge with spherical support

    size: int or tuple of 3 ints
    max_angle: float (degrees)
    min_angle: float (degrees)
    rotation: float (degrees) to rotate the wedge
    use_spherical_support: bool to use spherical support or not
    Note: Default rotation is -30 degrees so to match wiht 2d when the angles are from 0 to 120
    """

    if (isinstance(size, int)):
        size = (size,size,size)
    if use_spherical_support:
        wedge_2D = get_wedge(size[0], max_angle, min_angle, rotation = rotation)
    else:
        wedge_2D = get_wedge(size[0], max_angle, min_angle, rotation = rotation, radius = 2)

    x = np.linspace(-1,1,size[0])
    y = np.linspace(-1,1,size[1])
    z = np.linspace(-1,1,size[2])

    xx, yy, zz = np.meshgrid(x, y, z)


    

    if use_spherical_support:
        ball = xx**2 + yy**2 + zz**2 < 1
        wedge_3d = wedge_2D * ball
    else:
        ball = np.ones(size) 
        wedge_3d = wedge_2D*ball


    # adding the low support
    if low_support > 0:
        low_ball = xx**2 + yy**2 + zz**2 < low_support
        wedge_3d[low_ball] = 1

    return wedge_3d, ball

def get_wedge_new(size, max_angle, min_angle, rotation = 0,radius=10):
    """
    The wedge is a 2D array of size (size,size) with a wedge of angle max_angle-min_angle
    """
    #size = 2*size
    if isinstance(size, int):
        size = (size,size)
    wedge = np.zeros(size)
    x = np.linspace(-1,1,size[0], endpoint=True)
    y = np.linspace(-1,1,size[1], endpoint=True)
    xx, yy = np.meshgrid(x, y)

    wedge[xx**2 + yy**2 < radius] = 1


    wedge[yy >np.tan(np.deg2rad(max_angle)) * xx] = 0
    wedge[yy < np.tan(np.deg2rad(min_angle)) * xx] = 0

    #wedge = np.fliplr(wedge)

    wedge =wedge.T

    wedge_flip = np.flipud(wedge)
    wedge_flip= np.fliplr(wedge_flip)

    wedge = wedge + wedge_flip

    wedge = skrotate(wedge, float(rotation), resize=False)
    # wedge[wedge < 0.5] = 0
    # wedge[wedge >= 0.5] = 1
#    

    # crop the wedge to the original size

    #wedge = wedge[int(size[0]/4):int(3*size[0]/4),int(size[1]/4):int(3*size[1]/4)]

    return wedge


def get_wedge_3d_new(size,max_angle, 
                 min_angle, 
                 rotation = 0, 
                 low_support = 0,
                 use_spherical_support = False):
    """
    Get 3D wedge with spherical support

    size: int or tuple of 3 ints
    max_angle: float (degrees)
    min_angle: float (degrees)
    rotation: float (degrees) to rotate the wedge
    use_spherical_support: bool to use spherical support or not
    Note: Default rotation is -30 degrees so to match wiht 2d when the angles are from 0 to 120
    """

    if (isinstance(size, int)):
        size = (size+1,size+1,size+1)
    if use_spherical_support:
        wedge_2D = get_wedge_new(size[0], max_angle, min_angle, rotation = rotation)
    else:
        wedge_2D = get_wedge_new(size[0], max_angle, min_angle, rotation = rotation, radius = 2)

    x = np.linspace(-1,1,size[0], endpoint=True)
    y = np.linspace(-1,1,size[1], endpoint=True)
    z = np.linspace(-1,1,size[2], endpoint=True)

    xx, yy, zz = np.meshgrid(x, y, z)


    

    if use_spherical_support:
        ball = xx**2 + yy**2 + zz**2 < 1
        wedge_3d = wedge_2D * ball
    else:
        ball = np.ones(size) 
        wedge_3d = wedge_2D*ball


    # adding the low support
    if low_support > 0:
        low_ball = xx**2 + yy**2 + zz**2 < low_support
        wedge_3d[low_ball] = 1

    return wedge_3d, ball


def generate_all_cube_symmetries_torch(cube, wedge,use_flips = False, min_distance = 0.5):
    """
    Generates all 48 symmetries of a 3D cube (NxNxN NumPy array), including reflections.

    Parameters:
        cube (numpy.ndarray): A 3D NumPy array of shape (N, N, N).
        wedge (numpy.ndarray): A 3D NumPy array of shape (N, N, N) with
            1s where the wedge is and 0s elsewhere.

    Returns:
        list of numpy.ndarray: List containing all 48 unique transformations of the cube.
    """
    symmetries = []
    wedges = []
    k_sets =[]
    distances =[]
    # Generate rotational symmetries (24)

    # Generate rotational symmetries (24)
    for axes in permutations((0, 1, 2)):  # Permute axes (6 possibilities)
        for kx, ky, kz in product((0, 1, 2, 3), repeat=3):  # 4 rotations per axis
            rotated = torch.rot90(cube, k=kx, dims=(axes[0], axes[1]))
            rotated = torch.rot90(rotated, k=ky, dims=(axes[1], axes[2]))
            rotated = torch.rot90(rotated, k=kz, dims=(axes[0], axes[2]))


            wedge_rot = torch.rot90(wedge, k=kx, dims=(axes[0], axes[1]))
            wedge_rot = torch.rot90(wedge_rot, k=ky, dims=(axes[1], axes[2]))
            wedge_rot = torch.rot90(wedge_rot, k=kz, dims=(axes[0], axes[2]))
            
            if not any(torch.isclose(rotated, existing).all() for existing in symmetries):
                #x = np.isclose(wedge_rot, wedge).all()
                #print(x)
                #print(wedge_rot.shape)
                if torch.isclose(wedge_rot, wedge).all() == False:
                    #print('Hello ')
                    #print('Adding symmetry')

                    distance = torch.linalg.norm(wedge_rot - wedge)/ torch.linalg.norm(wedge)

                    if distance > min_distance:
                    #print(torch.linalg.norm(wedge_rot - wedge))
                        k_set = [kx, ky, kz, -1]
                        symmetries.append(rotated)
                        wedges.append(wedge_rot)
                        k_sets.append(k_set)
                        distances.append(torch.linalg.norm(wedge_rot - wedge))
                    #print(k_set)

            if use_flips:
                # check for symmetry
                for axis in range(3):
                    flipped = torch.flip(rotated, [axis])
                    flipped_wedge = torch.flip(wedge_rot, [axis])
                    if not any(torch.isclose(flipped, existing).all() for existing in symmetries):
                        if torch.isclose(flipped_wedge, wedge).all() == False:

                            distance = torch.linalg.norm(flipped_wedge - wedge)/ torch.linalg.norm(wedge)
                            if distance > min_distance:
                                symmetries.append(flipped)
                                wedges.append(flipped_wedge)
                                k_set = [kx, ky, kz, axis]
                                k_sets.append(k_set)
                                distances.append(torch.linalg.norm(flipped_wedge - wedge))

    return symmetries,wedges, k_sets, distances


def crop_volumes(volume1, volume_2, cropsize, n_crops):
    """
    Randomly crops the volume to generate n_crops crops of size cropsize
    """
    n1,n2,n3 = volume1.shape
    crops_1 = []
    crops_2 = []
    for i in range(n_crops):
        start1 = np.random.randint(0,n1-cropsize)
        start2 = np.random.randint(0,n2-cropsize)
        start3 = np.random.randint(0,n3-cropsize)
        crops_1.append(volume1[start1:start1+cropsize,start2:start2+cropsize,start3:start3+cropsize])
        crops_2.append(volume_2[start1:start1+cropsize,start2:start2+cropsize,start3:start3+cropsize])
    return crops_1, crops_2

def crop_volumes_mask(volume1, volume_2, mask, mask_frac,cropsize, n_crops):
    """
    Randomly crops the volume to generate n_crops crops of size cropsize
    """
    n1,n2,n3 = volume1.shape
    crops_1 = []
    crops_2 = []
    count = 0

    while count < n_crops:
        start1 = np.random.randint(0,n1-cropsize)
        start2 = np.random.randint(0,n2-cropsize)
        start3 = np.random.randint(0,n3-cropsize)
        crop_mask = mask[start1:start1+cropsize,start2:start2+cropsize,start3:start3+cropsize]
        #print(torch.mean(crop_mask))
        if torch.mean(crop_mask) < mask_frac:
            #print('Mask fraction: ',torch.mean(crop_mask))
            continue
        crops_1.append(volume1[start1:start1+cropsize,start2:start2+cropsize,start3:start3+cropsize])
        crops_2.append(volume_2[start1:start1+cropsize,start2:start2+cropsize,start3:start3+cropsize])
        count = count + 1
    return crops_1, crops_2

def generate_random_rotate(vol1,vol2, kset = None, wedge = None):


    if kset is None:
        kset = [[0, 0, 1, -1],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 2],
                [0, 0, 3, -1],
                [0, 0, 3, 1],
                [0, 1, 0, -1],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 0, 2],
                [0, 1, 1, -1],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 1, 1, 2],
                [0, 1, 2, -1],
                [0, 1, 2, 1],
                [0, 1, 3, -1],
                [0, 1, 3, 1],
                [0, 2, 1, -1],
                [0, 2, 3, -1],
                [0, 3, 0, -1],
                [0, 3, 1, -1],
                [0, 3, 2, -1],
                [0, 3, 3, -1],
                [1, 0, 0, -1],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 2],
                [1, 0, 1, -1],
                [1, 0, 1, 0],
                [1, 0, 1, 1],
                [1, 0, 1, 2],
                [1, 0, 2, -1],
                [1, 0, 2, 1],
                [1, 0, 3, -1],
                [1, 0, 3, 1],
                [1, 2, 0, -1],
                [1, 2, 1, -1],
                [1, 2, 2, -1],
                [1, 2, 3, -1]]
    
    k_rand = np.random.choice(len(kset))
    k = kset[k_rand]

    kx, ky, kz, axis = k

    rotated = torch.rot90(vol1, k=kx, dims=(1, 2))
    rotated = torch.rot90(rotated, k=ky, dims=(0, 2))
    rotated = torch.rot90(rotated, k=kz, dims=(0, 1))

    rotated2 = torch.rot90(vol2, k=kx, dims=(1, 2))
    rotated2 = torch.rot90(rotated2, k=ky, dims=(0, 2))
    rotated2 = torch.rot90(rotated2, k=kz, dims=(0, 1))

    if wedge is not None:
        rotated_wedge = torch.rot90(wedge, k=kx, dims=(1, 2))
        rotated_wedge = torch.rot90(rotated_wedge, k=ky, dims=(0, 2))
        rotated_wedge = torch.rot90(rotated_wedge, k=kz, dims=(0, 1))



    if axis != -1:
        rotated = torch.flip(rotated, [axis])
        rotated2 = torch.flip(rotated2, [axis])
        if wedge is not None:
            rotated_wedge = torch.flip(rotated_wedge, [axis])
          

    if wedge is not None:
        return rotated, rotated2, rotated_wedge

    return rotated, rotated2



def generate_random_rotate_4_vols(vol1,vol2, vol3, vol4, kset = None, wedge = None):


    if kset is None:
        kset = [[0, 0, 1, -1],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 2],
                [0, 0, 3, -1],
                [0, 0, 3, 1],
                [0, 1, 0, -1],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 0, 2],
                [0, 1, 1, -1],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 1, 1, 2],
                [0, 1, 2, -1],
                [0, 1, 2, 1],
                [0, 1, 3, -1],
                [0, 1, 3, 1],
                [0, 2, 1, -1],
                [0, 2, 3, -1],
                [0, 3, 0, -1],
                [0, 3, 1, -1],
                [0, 3, 2, -1],
                [0, 3, 3, -1],
                [1, 0, 0, -1],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 2],
                [1, 0, 1, -1],
                [1, 0, 1, 0],
                [1, 0, 1, 1],
                [1, 0, 1, 2],
                [1, 0, 2, -1],
                [1, 0, 2, 1],
                [1, 0, 3, -1],
                [1, 0, 3, 1],
                [1, 2, 0, -1],
                [1, 2, 1, -1],
                [1, 2, 2, -1],
                [1, 2, 3, -1]]
    
    k_rand = np.random.choice(len(kset))
    k = kset[k_rand]

    kx, ky, kz, axis = k

    rotated = torch.rot90(vol1, k=kx, dims=(1, 2))
    rotated = torch.rot90(rotated, k=ky, dims=(0, 2))
    rotated = torch.rot90(rotated, k=kz, dims=(0, 1))

    rotated2 = torch.rot90(vol2, k=kx, dims=(1, 2))
    rotated2 = torch.rot90(rotated2, k=ky, dims=(0, 2))
    rotated2 = torch.rot90(rotated2, k=kz, dims=(0, 1))


    rotated3 = torch.rot90(vol3, k=kx, dims=(1, 2))
    rotated3 = torch.rot90(rotated3, k=ky, dims=(0, 2))
    rotated3 = torch.rot90(rotated3, k=kz, dims=(0, 1))

    rotated4 = torch.rot90(vol4, k=kx, dims=(1, 2))
    rotated4 = torch.rot90(rotated4, k=ky, dims=(0, 2))
    rotated4 = torch.rot90(rotated4, k=kz, dims=(0, 1))




    if wedge is not None:
        rotated_wedge = torch.rot90(wedge, k=kx, dims=(1, 2))
        rotated_wedge = torch.rot90(rotated_wedge, k=ky, dims=(0, 2))
        rotated_wedge = torch.rot90(rotated_wedge, k=kz, dims=(0, 1))



    if axis != -1:
        rotated = torch.flip(rotated, [axis])
        rotated2 = torch.flip(rotated2, [axis])
        rotated3 = torch.flip(rotated3, [axis])
        rotated4 = torch.flip(rotated4, [axis])
        if wedge is not None:
            rotated_wedge = torch.flip(rotated_wedge, [axis])
          

    if wedge is not None:
        return rotated, rotated2, rotated3, rotated4, rotated_wedge

    return rotated, rotated2, rotated3, rotated4



def generate_random_rotate_1_vol(vol,kset = None):


    if kset is None:
        kset = [[0, 0, 1, -1],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 2],
                [0, 0, 3, -1],
                [0, 0, 3, 1],
                [0, 1, 0, -1],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 0, 2],
                [0, 1, 1, -1],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 1, 1, 2],
                [0, 1, 2, -1],
                [0, 1, 2, 1],
                [0, 1, 3, -1],
                [0, 1, 3, 1],
                [0, 2, 1, -1],
                [0, 2, 3, -1],
                [0, 3, 0, -1],
                [0, 3, 1, -1],
                [0, 3, 2, -1],
                [0, 3, 3, -1],
                [1, 0, 0, -1],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 2],
                [1, 0, 1, -1],
                [1, 0, 1, 0],
                [1, 0, 1, 1],
                [1, 0, 1, 2],
                [1, 0, 2, -1],
                [1, 0, 2, 1],
                [1, 0, 3, -1],
                [1, 0, 3, 1],
                [1, 2, 0, -1],
                [1, 2, 1, -1],
                [1, 2, 2, -1],
                [1, 2, 3, -1]]
    
    k_rand = np.random.choice(len(kset))
    k = kset[k_rand]

    kx, ky, kz, axis = k

    rotated = torch.rot90(vol, k=kx, dims=(1, 2))
    rotated = torch.rot90(rotated, k=ky, dims=(0, 2))
    rotated = torch.rot90(rotated, k=kz, dims=(0, 1))



    if axis != -1:
        rotated = torch.flip(rotated, [axis])

    return rotated


def generate_random_rotate_3vols(vol1,vol2, vol3, kset = None):


    if kset is None:
        kset = [[0, 0, 1, -1],
                [0, 0, 1, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 2],
                [0, 0, 3, -1],
                [0, 0, 3, 1],
                [0, 1, 0, -1],
                [0, 1, 0, 0],
                [0, 1, 0, 1],
                [0, 1, 0, 2],
                [0, 1, 1, -1],
                [0, 1, 1, 0],
                [0, 1, 1, 1],
                [0, 1, 1, 2],
                [0, 1, 2, -1],
                [0, 1, 2, 1],
                [0, 1, 3, -1],
                [0, 1, 3, 1],
                [0, 2, 1, -1],
                [0, 2, 3, -1],
                [0, 3, 0, -1],
                [0, 3, 1, -1],
                [0, 3, 2, -1],
                [0, 3, 3, -1],
                [1, 0, 0, -1],
                [1, 0, 0, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 2],
                [1, 0, 1, -1],
                [1, 0, 1, 0],
                [1, 0, 1, 1],
                [1, 0, 1, 2],
                [1, 0, 2, -1],
                [1, 0, 2, 1],
                [1, 0, 3, -1],
                [1, 0, 3, 1],
                [1, 2, 0, -1],
                [1, 2, 1, -1],
                [1, 2, 2, -1],
                [1, 2, 3, -1]]
    
    k_rand = np.random.choice(len(kset))
    k = kset[k_rand]

    kx, ky, kz, axis = k

    rotated = torch.rot90(vol1, k=kx, dims=(1, 2))
    rotated = torch.rot90(rotated, k=ky, dims=(0, 2))
    rotated = torch.rot90(rotated, k=kz, dims=(0, 1))

    rotated2 = torch.rot90(vol2, k=kx, dims=(1, 2))
    rotated2 = torch.rot90(rotated2, k=ky, dims=(0, 2))
    rotated2 = torch.rot90(rotated2, k=kz, dims=(0, 1))


    rotated3 = torch.rot90(vol3, k=kx, dims=(1, 2))
    rotated3 = torch.rot90(rotated3, k=ky, dims=(0, 2))
    rotated3 = torch.rot90(rotated3, k=kz, dims=(0, 1))




    if axis != -1:
        rotated = torch.flip(rotated, [axis])
        rotated2 = torch.flip(rotated2, [axis])
        rotated3 = torch.flip(rotated3, [axis])

    return rotated, rotated2, rotated3


def get_measurement(input_crop, wedge):
    """
    Get the wedge measurement of the input crop
    """
    
    dims = wedge.shape
    b,n1,n2,n3 = input_crop.shape
    input_crop_fft = torch.fft.fftshift(torch.fft.fftn(input_crop, dim=(-3,-2,-1), s =dims), dim=(-3,-2,-1))
    input_crop_fft = input_crop_fft * wedge[None]
    measure_crop = torch.fft.ifftn(torch.fft.ifftshift(input_crop_fft, dim=(-3,-2,-1)), dim=(-3,-2,-1)).real

    measure_crop = measure_crop[:,:n1,:n2,:n3]
    return measure_crop

def get_measurement_multi_wedge(input_crop, wedge):
    """
    Get the wedge measurement of the input crop
    """
    
    dims = wedge[0].shape
    b,n1,n2,n3 = input_crop.shape
    input_crop_fft = torch.fft.fftshift(torch.fft.fftn(input_crop, dim=(-3,-2,-1), s =dims), dim=(-3,-2,-1))
    input_crop_fft = input_crop_fft * wedge
    measure_crop = torch.fft.ifftn(torch.fft.ifftshift(input_crop_fft, dim=(-3,-2,-1)), dim=(-3,-2,-1)).real

    measure_crop = measure_crop[:,:n1,:n2,:n3]
    return measure_crop

def fourier_loss(target, estimate, wedge, criteria, use_fourier = True, window = None,view_as_real = False):
    
    """
    Calculate the Fourier loss between the target and estimate in the Fourier domain.

    Window function is used only for the real loss it is multiplied at the end
    """
    dims = wedge.shape
    b,n1,n2,n3 = estimate.shape

    if use_fourier is True and window is not None:
        # throw an error
        raise ValueError("Window function is not supported for Fourier loss")
    # if window  is not None:
    #     target = target*window[None]
    #     estimate = estimate*window[None]


    target_fft = torch.fft.fftshift(torch.fft.fftn(target, dim=(-3, -2, -1), s =dims), dim=(-3, -2, -1))
    estimate_fft = torch.fft.fftshift(torch.fft.fftn(estimate, dim=(-3, -2, -1),s =dims), dim=(-3, -2, -1))   

    target_fft = target_fft * wedge[None]
    estimate_fft = estimate_fft * wedge[None]

    if use_fourier:
        target_fft = torch.view_as_real(target_fft)
        estimate_fft = torch.view_as_real(estimate_fft)

        return criteria(target_fft, estimate_fft)/(sqrt(n1*n2*n3))
    else:
        target_miss = torch.fft.ifftn(torch.fft.ifftshift(target_fft, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        estimate_miss = torch.fft.ifftn(torch.fft.ifftshift(estimate_fft, dim=(-3, -2, -1)), dim=(-3, -2, -1))

        if view_as_real is False:
            target_miss = target_miss.real
            estimate_miss = estimate_miss.real
        

        target_miss = target_miss[:,:n1,:n2,:n3]
        estimate_miss = estimate_miss[:,:n1,:n2,:n3]

        if window is not None:
            target_miss = target_miss*window[None]
            estimate_miss = estimate_miss*window[None]

    if view_as_real:
        target_miss = torch.view_as_real(target_miss)
        estimate_miss = torch.view_as_real(estimate_miss)

            
            

    # print(target_fft.shape)
    # print(estimate_fft.shape)

    return criteria(target_miss, estimate_miss)


def fourier_loss_batch(target, estimate, wedge, criteria, use_fourier = True, window = None, view_as_real = False):
    
    """
    Calculate the Fourier loss between the target and estimate in the Fourier domain.

    Window function is used only for the real loss it is multiplied at the end
    """
    b,nw1,nw2,nw3 = wedge.shape
    dims = (nw1,nw2,nw3)
    b,n1,n2,n3 = estimate.shape

    if use_fourier is True and window is not None:
        # throw an error
        raise ValueError("Window function is not supported for Fourier loss")
    # if window  is not None:
    #     target = target*window[None]
    #     estimate = estimate*window[None]


    target_fft = torch.fft.fftshift(torch.fft.fftn(target, dim=(-3, -2, -1), s =dims), dim=(-3, -2, -1))
    estimate_fft = torch.fft.fftshift(torch.fft.fftn(estimate, dim=(-3, -2, -1),s =dims), dim=(-3, -2, -1))   

    target_fft = target_fft * wedge
    estimate_fft = estimate_fft * wedge

    if use_fourier:
        target_fft = torch.view_as_real(target_fft)
        estimate_fft = torch.view_as_real(estimate_fft)

        return criteria(target_fft, estimate_fft)/(sqrt(n1*n2*n3))
    else:
        target_miss = torch.fft.ifftn(torch.fft.ifftshift(target_fft, dim=(-3, -2, -1)), dim=(-3, -2, -1))
        estimate_miss = torch.fft.ifftn(torch.fft.ifftshift(estimate_fft, dim=(-3, -2, -1)), dim=(-3, -2, -1))

        if view_as_real is False:
            target_miss = target_miss.real
            estimate_miss = estimate_miss.real
        
        target_miss = target_miss[:,:n1,:n2,:n3]
        estimate_miss = estimate_miss[:,:n1,:n2,:n3]

        if window is not None:
            target_miss = target_miss*window[None]
            estimate_miss = estimate_miss*window[None]
            
    if view_as_real:
        target_miss = torch.view_as_real(target_miss)
        estimate_miss = torch.view_as_real(estimate_miss)

    # print(target_fft.shape)
    # print(estimate_fft.shape)

    return criteria(target_miss, estimate_miss)

def batch_rot(input1, input2, k_sets,wedge = None):
    

    B = input1.shape[0]

    inp_1_rot = torch.zeros_like(input1)
    inp_2_rot = torch.zeros_like(input2)
    if wedge is not None:
        n1_wedge, n2_wedge, n3_wedge = wedge.shape
        wedge_rot = torch.zeros((B, n1_wedge, n2_wedge, n3_wedge), dtype = input1.dtype, device = input1.device )

    for i in range(B):

        rotated_data = generate_random_rotate(input1[i], input2[i], k_sets, wedge = wedge)
        if wedge is not None:
            inp_r1, inp_r2, wedge_r = rotated_data
            wedge_rot[i] = wedge_r
        
        else:
            inp_r1, inp_r2 = rotated_data

        inp_1_rot[i] = inp_r1
        inp_2_rot[i] = inp_r2

    if wedge is not None:
        return inp_1_rot, inp_2_rot, wedge_rot
    

    return inp_1_rot, inp_2_rot

def batch_rot_4vol(input1, input2, input_3,input_4, k_sets,wedge = None):


    B = input1.shape[0]

    inp_1_rot = torch.zeros_like(input1)
    inp_2_rot = torch.zeros_like(input2)
    inp_3_rot = torch.zeros_like(input_3)
    inp_4_rot = torch.zeros_like(input_4)
    if wedge is not None:
        n1, n2, n3 = wedge.shape
        wedge_rot = torch.zeros((B, n1, n2, n3), dtype = input1.dtype, device = input1.device)

    for i in range(B):

        rotated_data = generate_random_rotate_4_vols(input1[i], input2[i], input_3[i], input_4[i], k_sets, wedge = wedge)
        if wedge is not None:
            inp_r1, inp_r2, inp_r3, inp_r4, wedge_r = rotated_data
            wedge_rot[i] = wedge_r
        
        else:
            inp_r1, inp_r2, inp_r3, inp_r4 = rotated_data

        inp_1_rot[i] = inp_r1
        inp_2_rot[i] = inp_r2
        inp_3_rot[i] = inp_r3
        inp_4_rot[i] = inp_r4

    if wedge is not None:
        return inp_1_rot, inp_2_rot, inp_3_rot, inp_4_rot, wedge_rot

    return inp_1_rot, inp_2_rot, inp_3_rot, inp_4_rot


def batch_rot_wedge(input1, input2, wedge, k_sets):
    

    B = input1.shape[0]

    inp_1_rot = torch.zeros_like(input1)
    inp_2_rot = torch.zeros_like(input2)
    wedge_rot = torch.zeros_like(input1)

    for i in range(B):

        inp_r1, inp_r2, wedge_r = generate_random_rotate_3vols(input1[i], input2[i],wedge,k_sets)
        inp_1_rot[i] = inp_r1
        inp_2_rot[i] = inp_r2
        wedge_rot[i] = wedge_r

    return inp_1_rot, inp_2_rot,wedge_rot


def symmetrize_3D(x_inp):

    """ FLip the input tensor leaving the first row/colume of each dimension unchanged. """
    
    x = x_inp.clone()
    x[1:] = torch.flip(x[1:], [0])
    x[:,1:] = torch.flip(x[:,1:], [1])
    x[:,:,1:] = torch.flip(x[:,:,1:], [2])
    return x

def symmetrize_3D_batch(x_inp):
    """ FLip the input tensor leaving the first row/colume of each dimension unchanged. """
    
    x = torch.zeros_like(x_inp)
    B = x_inp.shape[0]

    for i in range(B):
        x[i] = symmetrize_3D(x_inp[i])
    return x



def random_rotate_vols_full(vol_1,vol_2, wedge=None, grid=None):
    device= vol_1.device
    n1,n2,n3 = vol_1.shape

    if grid is None:
        theta = torch.zeros(1,3,4)
        theta[:,:,:3] = torch.eye(3)
        grid = torch.nn.functional.affine_grid(theta, (1,1,n1,n2,n3)).to(device)
        
    rot_mat = torch.tensor(R.random(1).as_matrix(),dtype = torch.float32, device= device)
    grid_rot = grid @ rot_mat

    vol_1_rot =  torch.nn.functional.grid_sample(vol_1[None,None], grid_rot)
    vol_2_rot =  torch.nn.functional.grid_sample(vol_2[None,None], grid_rot)
    if wedge is None:
        return vol_1_rot,vol_2_rot

    wedge_rot =  torch.nn.functional.grid_sample(wedge[None,None], grid_rot)

    return vol_1_rot,vol_2_rot,wedge_rot


def random_rotate_vols_full_4vols(vol_1,vol_2, vol_3, vol_4, wedge=None, grid=None):
    device= vol_1.device
    n1,n2,n3 = vol_1.shape

    if grid is None:
        theta = torch.zeros(1,3,4)
        theta[:,:,:3] = torch.eye(3)
        grid = torch.nn.functional.affine_grid(theta, (1,1,n1,n2,n3)).to(device)
        
    rot_mat = torch.tensor(R.random(1).as_matrix(),dtype = torch.float32, device= device)
    grid_rot = grid @ rot_mat

    vol_1_rot =  torch.nn.functional.grid_sample(vol_1[None,None], grid_rot)
    vol_2_rot =  torch.nn.functional.grid_sample(vol_2[None,None], grid_rot)
    vol_3_rot =  torch.nn.functional.grid_sample(vol_3[None,None], grid_rot)
    vol_4_rot =  torch.nn.functional.grid_sample(vol_4[None,None], grid_rot)
    if wedge is None:
        return vol_1_rot,vol_2_rot, vol_3_rot, vol_4_rot

    wedge_rot =  torch.nn.functional.grid_sample(wedge[None,None], grid_rot)

    return vol_1_rot,vol_2_rot, vol_3_rot, vol_4_rot, wedge_rot


def batch_rot_wedge_full(input1, input2, wedge, grid=None):
    

    B = input1.shape[0]
    n1_wedge, n2_wedge, n3_wedge = wedge.shape

    inp_1_rot = torch.zeros_like(input1)
    inp_2_rot = torch.zeros_like(input2)
    wedge_rot = torch.zeros((B, n1_wedge, n2_wedge, n3_wedge), dtype = input1.dtype, device = input1.device)

    for i in range(B):

        inp_r1, inp_r2, wedge_r = random_rotate_vols_full(input1[i], input2[i],wedge = wedge,grid = grid)
        inp_1_rot[i] = inp_r1
        inp_2_rot[i] = inp_r2
        wedge_rot[i] = wedge_r

    return inp_1_rot, inp_2_rot,wedge_rot


def batch_rot_wedge_full_4vols(input1, input2, input3, input4, wedge, grid=None):
    

    B = input1.shape[0]

    inp_1_rot = torch.zeros_like(input1)
    inp_2_rot = torch.zeros_like(input2)
    inp_3_rot = torch.zeros_like(input3)
    inp_4_rot = torch.zeros_like(input4)
    wedge_rot = torch.zeros_like(input1)

    for i in range(B):

        inp_r1, inp_r2, inp_r3, inp_r4, wedge_r = random_rotate_vols_full_4vols(input1[i], input2[i], input3[i], input4[i], wedge = wedge,grid = grid)
        inp_1_rot[i] = inp_r1
        inp_2_rot[i] = inp_r2
        inp_3_rot[i] = inp_r3
        inp_4_rot[i] = inp_r4
        wedge_rot[i] = wedge_r

    return inp_1_rot, inp_2_rot, inp_3_rot, inp_4_rot, wedge_rot


def batch_rot_full(input1, input2, grid=None):
    B = input1.shape[0]

    inp_1_rot = torch.zeros_like(input1)
    inp_2_rot = torch.zeros_like(input2)

    for i in range(B):
        inp_r1, inp_r2 = random_rotate_vols_full(input1[i], input2[i],wedge = None, grid = grid)
        inp_1_rot[i] = inp_r1
        inp_2_rot[i] = inp_r2

    return inp_1_rot, inp_2_rot
    

def crop_vol(vol, crop_size):
    crop_size_lim = crop_size//2
    if len(vol.shape)==3:
        n1,n2,n3 = vol.shape

        return vol[n1//2-crop_size_lim:n1//2+crop_size_lim,
        n2//2-crop_size_lim:n2//2+crop_size_lim,
        n3//2-crop_size_lim:n3//2+crop_size_lim]
        
    else:
        b,n1,n2,n3 = vol.shape
        return vol[:,n1//2-crop_size_lim:n1//2+crop_size_lim,
        n2//2-crop_size_lim:n2//2+crop_size_lim,
        n3//2-crop_size_lim:n3//2+crop_size_lim]
    



def upsample_fourier_rfft2x(x: torch.Tensor) -> torch.Tensor:
    """
    Upsample a real 3D volume by factor 2 in each axis via Fourier zero-padding,
    using rfftn/irfftn. Input x is (Dx, Dy, Dz) real float tensor (CPU or CUDA).
    Returns a real tensor of shape (2*Dx, 2*Dy, 2*Dz).
    """
    assert x.ndim == 3 and x.is_floating_point(), "x must be real 3D float tensor"

    Dx, Dy, Dz = x.shape
    Ox, Oy, Oz = 2*Dx, 2*Dy, 2*Dz

    # 1) rFFT (real -> complex half-spectrum on last axis)
    X = torch.fft.rfftn(x, dim=(0,1,2))  # shape: (Dx, Dy, Dz//2+1)

    # 2) Shift ONLY the non-last axes so DC is centered there
    Xs = torch.fft.fftshift(X, dim=(0,1))

    # 3) Prepare padded spectrum
    Z = torch.zeros((Ox, Oy, Oz//2 + 1), dtype=X.dtype, device=X.device)

    # 4) Compute central placement slices for non-last axes
    sx0 = (Ox - Dx) // 2
    sy0 = (Oy - Dy) // 2
    sx1 = sx0 + Dx
    sy1 = sy0 + Dy

    # 5) Along the last axis we simply copy the available nonnegative freqs
    #    from 0 .. Dz//2 into 0 .. Dz//2 of the padded spectrum.
    Z[sx0:sx1, sy0:sy1, : (Dz//2 + 1)] = Xs

    # 6) Unshift the non-last axes back
    Z = torch.fft.ifftshift(Z, dim=(0,1))

    # 7) Inverse rFFT to get the upsampled real grid
    y = torch.fft.irfftn(Z, s=(Ox, Oy, Oz), dim=(0,1,2)).real
    return y


def combine_names(vol_1,vol_2):
    """
    Combine the names of two volumes to create a unique identifier for the pair.
    """
    name_1 = vol_1.split('/')[-1].split('.mrc')[0]
    name_2 = vol_2.split('/')[-1].split('.mrc')[0]

    if name_1 < name_2:
        combined_name = name_1 + '_' + name_2
    else:
        combined_name = name_2 + '_' + name_1

    # add ei as the suffix
    combined_name = combined_name + '_ei'

    return combined_name+'.mrc'