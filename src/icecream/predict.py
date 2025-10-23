"""Predict a reconstruction using a trained model configuration."""

import os
import yaml
import typer
import mrcfile
import numpy as np
from pathlib import Path
from typing import Optional
from types import SimpleNamespace

from .models import get_model
from .utils.utils import combine_names
from .trainer import EquivariantTrainer

def get_latest_iteration(model_path):
    """
    Get the latest iteration number from the model path.
    Assumes the model files are named as 'model_iteration_{iteration}.pth'.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # List all files in the model directory
    files = os.listdir(model_path)
    
    # Filter for files that match the pattern
    iteration_files = [f for f in files if f.startswith('model_iteration_') and f.endswith('.pt')]

    if not iteration_files:
        return -1  # No iterations found

    # Extract iteration numbers and find the maximum
    iterations = [int(f.split('_')[-1].split('.')[0]) for f in iteration_files]

    return max(iterations)

def predict(config_yaml, config_path, iteration=-1, crop_size=None, batch_size=0, save_path=None):
    """
    Run the prediction using a trained model and save it.
    """

    # Get params
    configs = SimpleNamespace(**config_yaml)
    data_config = SimpleNamespace(**configs.data)
    path_1 = data_config.tomo0
    path_2 = data_config.tomo1
    mask_path = data_config.mask
    # if explicit save path is not given, save in the folder containing the config file
    save_path = config_path.parent if save_path is None else save_path
    # create save directory if it does not exist
    os.makedirs(save_path, exist_ok=True)

    # Throw not implemented error if path_1 or path_2 has more than one volume
    if isinstance(path_1, list) and len(path_1) > 1:
        raise NotImplementedError("Multiple volumes not supported yet.")
    if isinstance(path_2, list) and len(path_2) > 1:
        raise NotImplementedError("Multiple volumes not supported yet.")

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

    # Define the model
    model = get_model(**configs.model_params)

    # Load model weights
    train_config = SimpleNamespace(**configs.train_params)

    # Define the trainer
    trainer = EquivariantTrainer( configs=train_config,
                                model=model, 
                                angle_max=angle_max,
                                angle_min=angle_min,
                                angles=None,  # Set to specific angles if needed
                                save_path=save_path
                                )
    
    # Get name to load right weights
    model_path = os.path.join(save_path,'model')
    save_name = combine_names(path_1[0],path_2[0]).split('.mrc')[0]
    vol_est_name = save_name
    if iteration != -1:
        vol_est_name += f'_iteration_{iteration}'
    if batch_size !=0:
        print(f"Using batch size: {batch_size}")
        configs.predict_params['batch_size'] = batch_size
        vol_est_name += f'_bs_{batch_size}'
    if crop_size is not None:
        print(f"Using crop size: {crop_size}")
        configs.predict_params['crop_size'] = crop_size
        vol_est_name += f'_crop_{crop_size}'
    vol_est_name += '.mrc'
    if iteration == -1:
        iteration = get_latest_iteration(model_path)
    if iteration == -1:
        raise ValueError(f"No saved model found in {model_path}")

    # Load the model weights
    model_path = os.path.join(model_path, f'model_iteration_{iteration}.pt')
    trainer.load_model(model_path)
    trainer.load_data(vol_paths_1=path_1,
                    vol_paths_2=path_2, 
                    vol_mask_path=mask_path)

    # Evaluate the model
    vol_est = trainer.predict_dir(**configs.predict_params)

    # Save the estimated volume
    vol_save_path = os.path.join(save_path, vol_est_name)
    out = mrcfile.new(vol_save_path,overwrite=True)
    out.set_data(np.moveaxis(vol_est.astype(np.float32),2,0))
    out.close()

def main(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to YAML config file"),
    iteration: int = typer.Option(-1, help="Iteration to load the model from"),
    crop_size: Optional[int] = typer.Option(None, help="Crop size for the tomograms"),
    batch_size: Optional[int] = typer.Option(None, help="Batch size for prediction"),
    save_path: Optional[str] = typer.Option(None, help="Path to save the predicted volume")

):
    """Entry point replacing the old argparse interface."""

    config_dict = {}
    if config:
        with open(config, 'r') as f:
            config_dict = yaml.safe_load(f) or {}

    typer.echo(config_dict)
    typer.echo(f"iteration: {iteration}")
    typer.echo(f"crop_size: {crop_size}")
    typer.echo(f"batch_size: {batch_size}")

    predict(
        config_dict,
        config_path=config,
        save_path=save_path,
        iteration=iteration,
        crop_size=crop_size,
        batch_size=batch_size or 0,
    )

if __name__ == '__main__':
    typer.run(main)
