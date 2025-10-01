"""
Script to train a model using the provided configuration and dataset and evaluate it.
"""
import torch
import argparse
import yaml
from types import SimpleNamespace
import numpy as np
from .models import get_model
from .trainer import EquivariantTrainer
import mrcfile
import os
import json

def parse_args():
    parser = argparse.ArgumentParser()

    # Option 1: YAML config file
    parser.add_argument('--config', type=str, help='Path to YAML config file')
    # Optionall  use epoch parameter
    parser.add_argument('--epoch', type=int, default=-1, help='Epoch to load the model from')
    


    # Parse the arguments
    parser.add_argument('--crop_size', type=int , default=0, help='Crop size for the tomograms')
    parser.add_argument('--batch_size', type=int , default=0, help='Batch size for prediction')


    args = parser.parse_args()





    #print(args)


    config = {}

    # If YAML config is provided, load it
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f) or {}

    print(config)

    epoch = args.epoch
    crop_size = args.crop_size
    batch_size = args.batch_size


    if crop_size ==0:
        crop_size = None

    return config, epoch,crop_size, batch_size


def get_latest_epoch(model_path):
    """
    Get the latest epoch number from the model path.
    Assumes the model files are named as 'model_epoch_{epoch}.pth'.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    # List all files in the model directory
    files = os.listdir(model_path)
    
    # Filter for files that match the pattern
    epoch_files = [f for f in files if f.startswith('model_epoch_') and f.endswith('.pt')]
    
    if not epoch_files:
        return -1  # No epochs found

    # Extract epoch numbers and find the maximum
    epochs = [int(f.split('_')[-1].split('.')[0]) for f in epoch_files]
    
    return max(epochs)

def predict(config_yaml,epoch=-1,crop_size=None,batch_size=0, save_name=None):

    configs = SimpleNamespace(**config_yaml)

    data_config = SimpleNamespace(**configs.data)

    save_path = data_config.save_dir

    # create save directory if it does not exist
    if not os.path.exists(save_path):
        # throw an error that the directory does not exist
        raise FileNotFoundError(f"Directory not found: {save_path}")


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

        


    # load model model weights

    train_config = SimpleNamespace(**configs.train_params)



    trainer = EquivariantTrainer( configs=train_config,
                                model=model, 
                                angle_max=angle_max,
                                angle_min=angle_min,
                                angles=None,  # Set to specific angles if needed
                                save_path=save_path
                                )
    
    # save model parameters to a json file
    model_path = os.path.join(save_path,'model')

    vol_est_name = 'vol_est'

    if save_name is not None:
        vol_est_name = save_name
    if epoch != -1:
        vol_est_name += f'_epoch_{epoch}'

    if batch_size !=0:
        print(f"Using batch size: {batch_size}")
        configs.predict_params['batch_size'] = batch_size
        vol_est_name += f'_bs_{batch_size}'

    if crop_size is not None:
        print(f"Using crop size: {crop_size}")
        configs.predict_params['crop_size'] = crop_size
        vol_est_name += f'_crop_{crop_size}'


    wedge = None



    vol_est_name += '.mrc'

    if epoch ==-1:
        epoch = get_latest_epoch(model_path)

    model_path = os.path.join(model_path, f'model_epoch_{epoch}.pt')

    # Load the model weights
    trainer.load_model(model_path)
    
    trainer.load_data(vol_paths_1=path_1,
                    vol_paths_2=path_2, 
                    vol_mask_path=mask_path)
    

    # Evaluate the model


    vol_est = trainer.predict_dir(wedge=wedge, **configs.predict_params)

    # Save the estimated volume
    vol_save_path = os.path.join(save_path, vol_est_name)
    out = mrcfile.new(vol_save_path,overwrite=True)
    out.set_data(np.moveaxis(vol_est.astype(np.float32),2,0))
    out.close()


if __name__ == '__main__':
    configs, epoch, crop_size, batch_size = parse_args()
    print("epoch:", epoch)
    print("crop_size:", crop_size)
    print("batch_size:", batch_size)
    predict(configs, epoch, crop_size, batch_size)