"""Train a model using the provided configuration and dataset and evaluate it."""

import os
import json
import traceback
import yaml
import torch
import typer
import mrcfile
import numpy as np
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

from .models import get_model
from .utils.utils import combine_names,get_wedge_3d_new,generate_all_cube_symmetries_torch,symmetrize_3D
from .utils.data_util import load_data
from .trainer import EquivariantTrainerDDP
import torch.distributed as dist
import torch.multiprocessing as mp
import socket
def find_free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def train_model(config_yaml):
    print("Running distributed training with DDP...")
    # Reproducibility
    seed = config_yaml.get('train_params', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Params
    configs = SimpleNamespace(**config_yaml)
    data_config = SimpleNamespace(**configs.data)
    save_path = data_config.save_dir

    # create save directory if it does not exist
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(f"Created directory: {save_path}")
    else:
        print(f"Directory already exists: {save_path}")

    # save configs to a json file
    config_save_path = os.path.join(save_path, 'config.json')
    with open(config_save_path, 'w') as f:
        json.dump(configs.__dict__, f, indent=4)
    data_config = SimpleNamespace(**configs.data)

    # Collect data
    path_1 = data_config.tomo0
    path_2 = data_config.tomo1
    mask_path = data_config.mask
    if len(mask_path) == 0:
        mask_path = None


    # if configs.data has an attribute called 'angles', use it, otherwise set to None
    angles = getattr(data_config, 'angles', None)
    if len(angles) == 0:
        angles = None
    if angles is not None:
        angle_max_set = []
        angle_min_set = []
        if len(angles) == 1:
            print("Using angle file: {}".format(angles[0]))
            angles_arr = np.loadtxt(angles[0], dtype=np.float32)
            angle_min = np.min(angles_arr)
            angle_max = np.max(angles_arr)
            angle_max_set = [angle_max]*len(path_1)
            angle_min_set = [angle_min] * len(path_1)
        else:
            for i in range(len(angles)):
                print(f"Using angle file: {angles[i]}")
                angles_arr = np.loadtxt(angles[i], dtype=np.float32)
                angle_min = np.min(angles_arr)
                angle_max = np.max(angles_arr)
                angle_max_set.append(angle_max)
                angle_min_set.append(angle_min)
    else:
        print(f"Using angle value in [{data_config.tilt_min},{data_config.tilt_max}]")
        angle_min_set = [data_config.tilt_min]*len(path_1)
        angle_max_set = [data_config.tilt_max]*len(path_1)
    # assert (angle_min_set < angle_max_set), "angle_min should be less than angle_max"


    #  Generating ksets 
    print("Generating k-sets for rotations")
    crop_size = configs.train_params['crop_size']

    wedge,ball = get_wedge_3d_new(crop_size,
                                  max_angle = angle_max_set[0],
                                  min_angle = angle_min_set[0],
                                  rotation = 0,
                                  low_support=configs.train_params.get('wedge_low_support', None),
                                  use_spherical_support=configs.train_params.get('use_spherical_support', False))
    wedge_t = torch.tensor(wedge, dtype=torch.float32)
    wedge_t_sym = symmetrize_3D(wedge_t)
    wedge_t = (wedge_t_sym + wedge_t)/2
    wedge_t[wedge_t>0.1]  = 1
    
    random_cube = torch.rand(crop_size,crop_size,crop_size)
    _, _, k_sets, _ = generate_all_cube_symmetries_torch(random_cube,wedge_t,use_flips = 
                                                          configs.train_params['use_flips'])

    # Define the model and the trainer
    model = get_model(**configs.model_params)
    train_config = SimpleNamespace(**configs.train_params)


    configs.predict_params['batch_size'] = configs.train_params['batch_size']
    configs.predict_params['crop_size'] = configs.train_params['crop_size']
    pretrain_params = configs.train_params.get('pretrain_params', None)



    # Load Data
    vol_1_set, vol_2_set, vol_mask_set, mask_frac = load_data(vol_paths_1=path_1, 
                                                   vol_paths_2=path_2,
                                                    vol_mask_path=mask_path, 
                                                     load_device=train_config.load_device,
                                                    device='cpu', **configs.mask_params)

    mp.set_start_method("spawn", force=True)


    gpus =  train_config.device
    world_size = len(gpus)
    free_port = find_free_port()

    mp.spawn(
        worker,
        args=(
            gpus,free_port,
            model,
            pretrain_params,
            train_config,
            angle_max_set,
            angle_min_set,
            save_path,
            configs,
            vol_1_set,
            vol_2_set,
            path_1,
            path_2,
            mask_path,
            mask_frac,
            vol_mask_set,
            k_sets
        ),
        nprocs=world_size,
        join=True
    )


def worker(rank, gpus, free_port,
                 model,
                 pretrain_params,
                 train_config,
                 angle_max_set,
                 angle_min_set, 
                 save_path,
                 configs,
                 vol_1_set, 
                 vol_2_set, 
                 vol_path_1, 
                 vol_path_2,
                 mask_path,
                 mask_frac,
                 vol_mask_set,
                 k_sets):
    """Function to be run on each GPU for distributed training."""
    seed = train_config.seed
    torch.manual_seed(seed+rank)
    np.random.seed(seed+rank)
    torch.backends.cudnn.deterministic = True
    world_size = len(gpus)


    device_id = train_config.device[rank]
    torch.cuda.set_device(device_id)
    device_id = torch.device(f"cuda:{device_id}")

    master_addr = "127.0.0.1"
    master_port = str(free_port)  # choose a free port
    print(f"Rank {rank} initializing process group with master address {master_addr} and port {master_port}.")
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
        device_id=device_id
    )

    try:
        trainer = EquivariantTrainerDDP(configs=train_config,
                                    model=model,
                                    world_size = world_size,
                                    rank=rank,
                                    device=device_id,
                                    angle_max_set=angle_max_set,
                                    angle_min_set=angle_min_set,
                                    angles_set=None,  # Set to specific angles, for further development
                                    save_path=save_path
                                 )
        print(f"Rank {rank} loading data.")
        trainer.load_data(vol_paths_1 =vol_path_1,
                    vol_paths_2=vol_path_2,
                        vol_mask_path=mask_path, 
                        vol_1_set=vol_1_set,
                        vol_2_set=vol_2_set,
                        vol_mask_set=vol_mask_set,
                        mask_frac=mask_frac,
                        k_sets = k_sets,)
        print(f"Rank {rank} data loaded.")

        # Possibly use pre-trained model
        pretrain_params = configs.train_params.get('pretrain_params', None)
        if pretrain_params is not None:
            use_pretrain = pretrain_params.get('use_pretrain', False)
            if use_pretrain:
                print("Using pretrained model parameters.")
                model_path = pretrain_params['model_path']
                if model_path:
                    print(f"Loading pretrained model from {model_path}")
                    trainer.load_model(model_path, pretrained=True)
                else:
                    print("No pretrained model path provided, training from scratch.")

        # Train the model
        print(f"Rank {rank} starting training.")
        trainer.train(iterations=train_config.iterations, configs=configs)
        if rank ==0:
            # Save the final model after full training
            trainer.save_model(iteration=train_config.iterations)

            # Evaluate the model and reconstruct the tomogram
            vol_est = trainer.predict_dir(**configs.predict_params)
            for i in range(len(vol_est)):
                # Save the estimated volume
                name = combine_names(vol_path_1[i], vol_path_2[i])
                vol_save_path = os.path.join(save_path, name)
                out = mrcfile.new(vol_save_path, overwrite=True)
                out.set_data(np.moveaxis(vol_est[i].astype(np.float32), 2, 0))
                out.close()
                print(f"Volume saved at: {os.path.join(save_path, name)}")
    except Exception as e:
        # print stack trace for debugging
        
        traceback.print_exc()
        print(f"An error occurred in rank {rank}: {e}")
    finally:
        dist.destroy_process_group()


