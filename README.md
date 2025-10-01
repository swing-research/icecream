# cryoei
Self-supervised method for cryo-et reconstruction using noise2noise and equivariant imaging

## Installation

Clone the repository:

```bash
git clone git@github.com:vinith2/cryoei.git
cd cryoei
```

Create a conda environment or you can use other environment managers like pipenv, poetry, uv etc with python 3.11 or above. We will use conda
with python 3.11 in this example:

```bash
conda create -n cryoei python=3.11 -y
conda activate cryoei
```
Install cuda enabled pytorch from https://pytorch.org/get-started/locally/ based on your system configuration. For example, for linux with cuda 12.8

```bash
pip install torch  --index-url https://download.pytorch.org/whl/cu128
``` 

Install cryoei and its dependencies:

```bash
pip install -e .
```
To test the installation, run:

```bash
cryoei --help
```
It should display the two main commands: `train` and `predict`.



## Usage
To train the model, use the `train` command. Note the the command also reconstruct the volume after training. Howeve, you can use the `predict` command to reconstruct the volume from a trained model using intermediate checkpoints.


To train the model you can use a config file or directly pass the parameters through command line. This uses the default training parameters specified in `src/cryoei/default.yaml`. You can override these by passing  a config file of your own. 

Now, we show how to train the model using default parameters. 

```bash
cryoei train \
  --tomo0 /path/to/tomogram_0.mrc \
  --tomo1 /path/to/tomogram_1.mrc \
  --angles /path/to/angles.tlt \
  --save-dir /path/to/save/dir \
  --batch-size 8
```

This will train the model using the two tomograms `tomogram_0.mrc` and `tomogram_1.mrc` with tilt angles specified in `angles.tlt`. The trained model and reconstructions will be saved in `/path/to/save/dir`. The training batch size is set to 8. You can change other training parameters like number of epochs,scale etc.

To train the model using a config file, create a yaml file with the contents similar to `src/cryoei/defaults.yaml` and pass it to the `--config` option. 
```bash
cryoei train --config /path/to/config.yaml
```

In your config file, make sure to update the  `data` section with the correct tomogram paths and angle information. For example:

```yaml
data:
  save_dir: "runs/my_experiment/"   # directory to save outputs
  tomo0_files: 
    - "/path/to/tomogram_0.mrc"  # first tomogram
  tomo1_files: 
    - "/path/to/tomogram_1.mrc"  # second tomogram
  mask_file: null  # optional mask file

  # Option 1: specify tilt range
  # tilt_max: 57.1
  # tilt_min: -60.0

  # Option 2: specify an external angle file (recommended)
  angle_file: "/path/to/angles.tlt"
```

**Note:** The train command also reconstructs the volume after training. However, you can use the `predict` command to reconstruct the volume from a trained model using intermediate checkpoints.


To reconstruct the volume from a trained model, use the `predict` command. You can specify the epoch of the model to use for reconstruction. If not specified, it will use the latest epoch.

```bash
cryoei predict  --config /path/to/config.json \
  --tomo0 /path/to/tomogram_0.mrc \
  --tomo1 /path/to/tomogram_1.mrc \
  --angles /path/to/angles.tlt \
  --save-name my_reconstruction \
  --epoch 50 \  # epoch to use for reconstruction (optional)
```