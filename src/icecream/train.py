"""Train a model using the provided configuration and dataset and evaluate it."""

import os
import glob
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
from .trainer import EquivariantTrainer
from .predict import get_latest_iteration

def train_model(config_yaml):
    # Reproducibility
    seed = config_yaml.get('train_params', {}).get('seed', 42)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

    # Params
    configs = SimpleNamespace(**config_yaml)
    data_config = SimpleNamespace(**configs.data)
    save_path = data_config.save_dir

    max_number_vol = configs.train_params.get('max_number_vol', None)
    if max_number_vol is None:
        configs.train_params['max_number_vol'] = -1
        configs.train_params['iter_update_vol'] = -1
    iter_update_vol = configs.train_params.get('iter_update_vol', None)
    if iter_update_vol is None:
        configs.train_params['iter_update_vol'] = -1

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
    if isinstance(path_1, str):
        path_1 = glob.glob(path_1)
        path_2 = glob.glob(path_2)
    elif isinstance(path_1, list):
        path_1_ = []
        path_2_ = []
        for i in range(len(path_1)):
            path_1_.extend(glob.glob(path_1[i]))
            path_2_.extend(glob.glob(path_2[i]))
        path_1 = path_1_
        path_2 = path_2_
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
    model = get_model(**configs.model_params)
    train_config = SimpleNamespace(**configs.train_params)
    trainer = EquivariantTrainer(configs=train_config,
                                 model=model,
                                 angle_max_set=angle_max_set,
                                 angle_min_set=angle_min_set,
                                 angles_set=None,  # Set to specific angles, for further development
                                 save_path=save_path
                                 )
    print("Loading the tomograms ...")
    trainer.load_data(vol_paths_1=path_1,
                      vol_paths_2=path_2,
                      vol_mask_path=mask_path, max_number_vol=train_config.max_number_vol, **configs.mask_params)
    print("Tomograms loaded.")

    # Possibly use pre-trained model
    pretrain_params = configs.train_params.get('pretrain_params', None)
    if pretrain_params is not None:
        use_pretrain = pretrain_params.get('use_pretrain', False)
        if use_pretrain:
            print("Using pretrained model parameters.")
            model_path = pretrain_params.get('model_path',None)
            if model_path is None:
                model_save_path = os.path.join(data_config.save_dir, 'model')
                iteration = configs.predict_params['iter_load']
                if iteration == -1 and os.path.exists(model_save_path):
                    iteration = get_latest_iteration(model_save_path)
                if iteration == -1:
                    print(f"No saved model found in {model_save_path}")
                model_path = os.path.join(model_save_path, f'model_iteration_{iteration}.pt')
            if not os.path.exists(model_path):
                print("No pretrained model path provided, training from scratch.")
            else:
                print(f"Loading pretrained model from {model_path}")
                trainer.load_model(model_path)

    configs.predict_params['batch_size'] = configs.train_params['batch_size']
    configs.predict_params['crop_size'] = configs.train_params['crop_size']

    # Train the model
    trainer.train(iterations_tot=train_config.iterations, configs=configs)

    # Save the final model after full training
    trainer.save_model(iteration=train_config.iterations)

    # Evaluate the model and reconstruct the tomogram
    vol_est = trainer.predict_dir(**configs.predict_params)
    for i in range(len(vol_est)):
        # Save the estimated volume
        name = combine_names(path_1[i], path_2[i])
        vol_save_path = os.path.join(save_path, name)
        out = mrcfile.new(vol_save_path, overwrite=True)
        out.set_data(np.moveaxis(vol_est[i].astype(np.float32), 2, 0))
        out.close()
        print(f"Volume saved at: {os.path.join(save_path, name)}")

def main(config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to YAML config file")):
    """Entry point mirroring the old argparse interface."""
    config_dict = {}
    if config:
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f) or {}
    typer.echo(config_dict)
    train_model(config_dict)

if __name__ == '__main__':
    typer.run(main)