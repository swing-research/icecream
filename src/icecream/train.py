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
from .trainer import EquivariantTrainer

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

    # # Throw not implemented error if path_1 or path_2 has more than one volume
    # if isinstance(path_1, list) and len(path_1) > 1:
    #     raise NotImplementedError("Multiple volumes not supported yet.")
    # if isinstance(path_2, list) and len(path_2) > 1:
    #     raise NotImplementedError("Multiple volumes not supported yet.")

    # if configs.data has an attribute called 'angles', use it, otherwise set to None
    angles = getattr(data_config, 'angles', None)
    if angles is not None:
        angles_arr = np.loadtxt(angles, dtype=np.float32)
        angle_min = np.min(angles_arr)
        angle_max = np.max(angles_arr)
    else:
        angle_min = data_config.tilt_min
        angle_max = data_config.tilt_max
    assert angle_min < angle_max, "angle_min should be less than angle_max"

    # Define the model and the trainer
    model = get_model(**configs.model_params)
    train_config = SimpleNamespace(**configs.train_params)
    trainer = EquivariantTrainer(configs=train_config,
                                 model=model,
                                 angle_max=angle_max,
                                 angle_min=angle_min,
                                 angles_set=None,  # Set to specific angles, for further development
                                 save_path=save_path
                                 )
    print("Loading the tomograms ...")
    trainer.load_data(vol_paths_1=path_1,
                      vol_paths_2=path_2,
                      vol_mask_path=mask_path, **configs.mask_params)
    print("Tomograms loaded.")

    # Possibly use pre-trained model
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

    configs.predict_params['batch_size'] = configs.train_params['batch_size']
    configs.predict_params['crop_size'] = configs.train_params['crop_size']

    # Train the model
    trainer.train(iterations=train_config.iterations, configs=configs)

    # Save the final model after full training
    trainer.save_model(iteration=train_config.iterations)

    # Evaluate the model and reconstruct the tomogram
    vol_est = trainer.predict_dir(**configs.predict_params)
    # Save the estimated volume
    name = combine_names(path_1[0], path_2[0])
    vol_save_path = os.path.join(save_path, name)
    out = mrcfile.new(vol_save_path, overwrite=True)
    out.set_data(np.moveaxis(vol_est.astype(np.float32), 2, 0))
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