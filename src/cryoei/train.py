"""Train a model using the provided configuration and dataset and evaluate it."""

from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import json
import os

import mrcfile
import numpy as np
import torch
import typer
import yaml

from .models import get_model
from .trainer import EquivariantTrainer


def train_model(config_yaml):

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

    path_1 = data_config.tomo0_files
    path_2 = data_config.tomo1_files
    mask_path = data_config.mask_file


    # if configs.data has an attribute called 'angles', use it, otherwise set to None
    angle_file = getattr(data_config, 'angle_file', None)
    if angle_file is not None:
        angles = np.loadtxt(angle_file, dtype=np.float32)
        angle_min = np.min(angles)
        angle_max = np.max(angles)
    else:
        angle_min = data_config.tilt_min
        angle_max = data_config.tilt_max

    # test if angle)_min and angle_max are set correctly 
    assert angle_min < angle_max, "angle_min should be less than angle_max"



    model = get_model(**configs.model_params)
    # save model parameters to a json file


    train_config = SimpleNamespace(**configs.train_params)



    trainer = EquivariantTrainer( configs=train_config,
                                model=model, 
                                angle_max=angle_max,
                                angle_min=angle_min,
                                angles=None,  # Set to specific angles if needed
                                save_path=save_path
                                )
    
    trainer.load_data(vol_paths_1=path_1,
                    vol_paths_2=path_2, 
                    vol_mask_path=mask_path, **configs.mask_params)
    

    if hasattr(configs, 'pretrain_params'):
        pretrain_params = SimpleNamespace(**configs.pretrain_params)

        if pretrain_params.use_pretrain:
            print("Using pretrained model parameters.")
            model_path = pretrain_params.model_path
            if model_path:
                print(f"Loading pretrained model from {model_path}")
                trainer.load_model(model_path, pretrained=True)
            else:
                print("No pretrained model path provided, training from scratch.")
    
    trainer.train(repeats=train_config.epochs)

    trainer.save_model(epoch=train_config.epochs)

    # Evaluate the model
    #vol_est = trainer.predict(**configs.predict_params)
    vol_est = trainer.predict_dir(wedge=trainer.wedge_input, **configs.predict_params)


    # Save the estimated volume
    vol_save_path = os.path.join(save_path, 'vol_est.mrc')
    out = mrcfile.new(vol_save_path,overwrite=True)
    out.set_data(np.moveaxis(vol_est.astype(np.float32),2,0))
    out.close()

    


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
