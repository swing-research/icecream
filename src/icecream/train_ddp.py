"""Train a model using the provided configuration and dataset and evaluate it."""

import os
import json
import yaml
import torch
import typer
import mrcfile
import numpy as np
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

from .models import get_model
from .utils.utils import combine_names
from .trainer import BaseTrainerDDP
from icecream.utils.data_util import load_data
import torch.distributed as dist
import torch.multiprocessing as mp

def train_model(config_yaml):
    # Reproducibility
    seed = config_yaml.get('train_params', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Params
    configs = SimpleNamespace(**config_yaml)
    data_config = SimpleNamespace(**configs.data)
    train_config = SimpleNamespace(**configs.train_params)
    pretrain_params = configs.train_params.get('pretrain_params', None)


    print('Devices to be used:', train_config.device)
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

    # # Throw not implemented error if path_1 or path_2 has more than one volume
    # if isinstance(path_1, list) and len(path_1) > 1:
    #     raise NotImplementedError("Multiple volumes not supported yet.")
    # if isinstance(path_2, list) and len(path_2) > 1:
    #     raise NotImplementedError("Multiple volumes not supported yet.")

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

    # Define the model and the trainer
   
    

    # Load Data

    vol_1_set, vol_2_set, vol_mask_set, mask_frac = load_data(vol_paths_1=path_1, 
                                                   vol_paths_2=path_2,
                                                    vol_mask_path=mask_path, 
                                                     load_device=train_config.load_device,
                                                    device='cpu', **configs.mask_params)



    mp.set_start_method("spawn", force=True)


    gpus =  train_config.device
    world_size = len(gpus)

    mp.spawn(
        worker,
        args=(
            gpus,
            get_model(**configs.model_params),
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
            vol_mask_set
        ),
        nprocs=world_size,
        join=True
    )




def main(config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to YAML config file")):
    """Entry point mirroring the old argparse interface."""
    config_dict = {}
    if config:
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
    typer.echo(config_dict)
    train_model(config_dict)

def worker(rank, gpus, 
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
                 vol_mask_set):
    """Function to be run on each GPU for distributed training."""
    seed = train_config.seed
    torch.manual_seed(seed+rank)
    np.random.seed(seed+rank)
    torch.backends.cudnn.deterministic = True
    world_size = len(gpus)

    master_addr = "127.0.0.1"
    master_port = "29501"  # choose a free port
    dist.init_process_group(
        backend="nccl",
        init_method=f"tcp://{master_addr}:{master_port}",
        world_size=world_size,
        rank=rank,
    )

    model = get_model(**configs.model_params)
    print(f"Model initialized on rank {rank}.")
    trainer = BaseTrainerDDP(configs=train_config,
                             model=model,
                             world_size = world_size,
                             rank=rank,
                             angle_max_set=angle_max_set,
                             angle_min_set=angle_min_set,
                             angles_set=None,
                             save_path=save_path
                             )
    
    print(f"Rank {rank} loading data.")
    trainer.load_data(vol_paths_1 =vol_path_1,
                   vol_paths_2=vol_path_2,
                     vol_mask_path=mask_path, 
                     vol_1_set=vol_1_set,
                     vol_2_set=vol_2_set,
                     vol_mask_set=vol_mask_set,
                     mask_frac=mask_frac)
    print(f"Rank {rank} data loaded.")
    
    # Possibly use pre-trained model
    if pretrain_params is not None:
        use_pretrain = pretrain_params.get('use_pretrain', False)
        if use_pretrain:
            print(f"Using pretrained model parameters on rank {rank}.")
            model_path = pretrain_params['model_path']
            if model_path:
                print(f"Loading pretrained model from {model_path} on rank {rank}.")
                trainer.load_model(model_path, pretrained=True)
            else:
                print(f"No pretrained model path provided on rank {rank}, training from scratch.")

    print(f"Rank {rank} starting training for {train_config.iterations} iterations.")
    trainer.train(iterations=train_config.iterations, configs=train_config)

    print(f"Rank {rank} finished training.")


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

    dist.destroy_process_group()     

if __name__ == '__main__':
    typer.run(main)