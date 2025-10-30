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

def predict(config_yaml):
    """
    Run the prediction using a trained model and save it.
    """

    # Get params
    configs = SimpleNamespace(**config_yaml)
    data_config = SimpleNamespace(**configs.data)
    predict_config = SimpleNamespace(**configs.predict_params)
    # if data_config.tomo1 is None:
    #     data_config.tomo1 = data_config.tomo0
    path_1 = data_config.tomo0
    path_2 = data_config.tomo1
    mask_path = data_config.mask
    # create save directory if it does not exist
    save_dir_reconstructions = predict_config.save_dir_reconstructions
    os.makedirs(save_dir_reconstructions, exist_ok=True)

    # if configs.data has an attribute called 'angles', use it, otherwise set to None
    angles = getattr(data_config, 'angles', None)
    if angles is not None:
        angle_max_set = []
        angle_min_set = []
        if len(angles) == 1:
            angles_arr = np.loadtxt(angles[0], dtype=np.float32)
            angle_min = np.min(angles_arr)
            angle_max = np.max(angles_arr)
            angle_max_set = [angle_max]*len(path_1)
            angle_min_set = [angle_min] * len(path_1)
        else:
            for i in range(len(angles)):
                angles_arr = np.loadtxt(angles[i], dtype=np.float32)
                angle_min = np.min(angles_arr)
                angle_max = np.max(angles_arr)
                angle_max_set.append(angle_max)
                angle_min_set.append(angle_min)
    else:
        angle_min_set = [data_config.tilt_min]*len(path_1)
        angle_max_set = [data_config.tilt_max]*len(path_1)
    assert (angle_min_set < angle_max_set).sum(), "angle_min should be less than angle_max"

    # Define the model
    model = get_model(**configs.model_params)

    # Load model weights
    train_config = SimpleNamespace(**configs.train_params)

    # Define the trainer
    trainer = EquivariantTrainer( configs=train_config,
                                model=model, 
                                angle_max_set=angle_max_set,
                                angle_min_set=angle_min_set,
                                angles_set=None,  # Set to specific angles if needed
                                save_path=save_dir_reconstructions
                                )

    model_save_path = os.path.join(data_config.save_dir, 'model')
    iteration = predict_config.iter_load
    if iteration == -1:
        iteration = get_latest_iteration(model_save_path)
    if iteration == -1:
        raise ValueError(f"No saved model found in {model_save_path}")
    model_path = os.path.join(model_save_path, f'model_iteration_{iteration}.pt')
    trainer.load_model(model_path)
    trainer.load_data(vol_paths_1=path_1,
                    vol_paths_2=path_2,
                    vol_mask_path=mask_path)

    # Reconstruct each volume in the list
    print("####################")
    print("  Started inference.")
    print("####################")
    vol_est_list = trainer.predict_dir(**configs.predict_params)
    print("####################")
    print("  Finished inference.")
    print("####################")
    for i in range(len(vol_est_list)):
        # Save the estimated volume
        if path_2 is not None:
            name = combine_names(path_1[i], path_2[i])
        else:
            name = combine_names(path_1[i], '')
        vol_save_path = os.path.join(save_dir_reconstructions, name)
        print("Saving at ",vol_save_path)
        out = mrcfile.new(vol_save_path,overwrite=True)
        out.set_data(np.moveaxis(vol_est_list[i].astype(np.float32),2,0))
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
