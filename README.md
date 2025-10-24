# 🍦 Icecream: High-Fidelity Equivariant Cryo-Electron Tomography

Official repo for ([icecream](https://www.biorxiv.org/content/10.1101/2025.10.17.682746v1)).


Icecream is a self-supervised framework for cryo-ET reconstruction that integrates equivariance principles from modern imaging theory into a deep-learning architecture.
Icecream provides a theoretically grounded and computationally efficient method that jointly performs denoising and missing-wedge correction.  

### 🧊 Note
The codebase is under active development. 

#### Updates 
Date: 23.10.2025
 - Added save_tomo_n_iterations, to compute and save the current reconstruction during training
 - Added the command line split-tilt-series to split a tilt series along the angle dimension.
 - Added comments on the parameters of the default yaml file
 - Code cleanup. 
 - Added support to use a pre-trained model as initialization during training. 
 - Added an option for torch.compile.

The current version supports training on a single split of the tomograms.  
Upcoming updates will include support for **multi-volume training**.

## Installation

Clone the repository:

```bash
git clone git@github.com:swing-research/icecream.git
cd icecream
```

Create a conda environment or you can use other environment managers like pipenv, poetry, uv etc with Python 3.11 or above. We will use conda
with Python 3.11 in this example:

```bash
conda create -n icecream python=3.11 -y
conda activate icecream
```
Install CUDA-enabled PyTorch from https://pytorch.org/get-started/locally/ based on your system configuration. For example, for Linux with CUDA 12.8


```bash
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
``` 
**Note**: PyTorch 2.9 currently has a bug affecting half-precision (FP16) training. Please use PyTorch 2.8 or earlier for now. See [this issue](https://github.com/pytorch/pytorch/issues/166122).


Install Icecream and its dependencies:

```bash
pip install -e .
```
To test the installation, run:

```bash
icecream --help
```
It should display the two main commands: `train` and `predict`.



## Usage
To train the model, use the `train` command. Note that the command also reconstructs the volume after training. However, you can use the `predict` command to reconstruct the volume from a trained model using intermediate checkpoints.


To train the model, you can use a config file or directly pass the parameters through the command line. This uses the default training parameters specified in `src/icecream/default.yaml`. You can override these by passing a config file of your own. 

Now, we show how to train the model using default parameters. 

```bash
icecream train \
  --tomo0 /path/to/tomogram_0.mrc \
  --tomo1 /path/to/tomogram_1.mrc \
  --angles /path/to/angles.tlt \
  --save-dir /path/to/save/dir \
  --batch-size 8
```

This will train the model using the two tomograms `tomogram_0.mrc` and `tomogram_1.mrc` with tilt angles specified in `angles.tlt`. The reconstructions will be saved in the directory `/path/to/save/dir` along with the 'config.json' file containing the training and model parameters. The actual model files will be saved in `/path/to/save/dir/models`. The training batch size is set to 8. You can change other training parameters like the number of epochs, scale, etc. 


### Using a pre-trained model for initialization
You can optionally initialize the weights of the model using a pre-trained model. This can help in faster convergence. To do this, use the `--pretrain-path` option to specify the path to the pre-trained model file (.pt file). Note that the pre-trained model should be compatible with the current model architecture.

```bash
icecream train \
  --tomo0 /path/to/tomogram_0.mrc \
  --tomo1 /path/to/tomogram_1.mrc \
  --angles /path/to/angles.tlt \
  --save-dir /path/to/save/dir \
  --batch-size 8 \
  --pretrain-path /path/to/pretrained_model.pt
```


To train the model using a config file, create a YAML file with the contents similar to `src/icecream/defaults.yaml` and pass it to the `--config` option. 
```bash
icecream train --config /path/to/config.yaml
```

In your config file, make sure to update the `data` section with the correct tomogram paths and angle information. For example:

```yaml
data:
  save_dir: "runs/my_experiment/"   # directory to save outputs
  tomo0: 
    - "/path/to/tomogram_0.mrc"  # first tomogram
  tomo1: 
    - "/path/to/tomogram_1.mrc"  # second tomogram
  mask: null  # optional mask file

  # Option 1: specify tilt range
  # tilt_max: 57.1
  # tilt_min: -60.0

  # Option 2: specify an external angle file (recommended)
  angles: "/path/to/angles.tlt"
```

**Note:** The train command also reconstructs the volume after training. However, you can use the `predict` command to reconstruct the volume from a trained model using intermediate checkpoints.


To reconstruct the volume from a trained model, use the `predict` command. You can specify the epoch of the model to use for reconstruction. If not specified, it will use the latest epoch.

```bash
icecream predict --config /path/to/config.json \
  --tomo0 /path/to/tomogram_0.mrc \
  --tomo1 /path/to/tomogram_1.mrc \
  --angles /path/to/angles.tlt \
  --save_dir /path/to/save/dir \  # directory to save output if different from training (optional)
  --iteration 50000  # iteration to use for reconstruction (optional)
```
Note that the `config.json` file is generated during training and contains the model and training parameters and saved in the training save directory. You can also use a YAML config file instead of JSON. 


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details
