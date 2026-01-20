import json
import subprocess
import sys
from pathlib import Path
import mrcfile
import numpy as np

def generate_random_data(tmp_path: Path):
    """
    Generate random tomograms and angles file for testing.
    """

    tomo0 = tmp_path / "tomo0.mrc"
    tomo1 = tmp_path / "tomo1.mrc"
    mask = tmp_path / "mask.mrc"
    angles = tmp_path / "angles.tlt"

    # generate random small tomograms for faster testing


    with mrcfile.new(tomo0, overwrite=True) as mrc:
        small_data = np.random.rand(256,512, 512).astype(np.float32)
        mrc.set_data(small_data)

    with mrcfile.new(tomo1, overwrite=True) as mrc:
        small_data = np.random.rand(256,512, 512).astype(np.float32)
        mrc.set_data(small_data)

    with mrcfile.new(mask, overwrite=True) as mrc:
        small_data = np.zeros((256,512, 512), dtype=np.float32)
        small_data[64:-64, 32:-32, 32:-32] = 1.0  # simple cubic mask
        mrc.set_data(small_data)

    anlges_array = np.linspace(-60, 57.1, num=120).astype(np.float32)
    np.savetxt(angles, anlges_array, fmt='%.4f')

    return tomo0, tomo1, angles, mask


def test_default_training(tmp_path: Path):
    """
    Default Integration test for `icecream train` with default setting using a single GPU. 

    This runs the training for 100 iterations on a sample pair of tomograms with angles file.
    1. It checks that the training completes successfully.
    2. It verifies the presence of expected output files in the specified save directory.
    3. It ensures that the configuration file is saved correctly.

    TODO: Maybe chaange the data to randomly generated small tomograms to speed up test.
    """
    
    # #save_dir.mkdir()

    # save_dir = './../icecream-runs/temp/'

    tomo0, tomo1, angles,_ = generate_random_data(tmp_path)

    save_dir = tmp_path / "run"




    cmd = [
         "icecream", "train",
        "--tomo0", str(tomo0),
        "--tomo1", str(tomo1),
        "--angles", str(angles),
        "--save-dir", str(save_dir),
        "--batch-size", "2",
        "--device", "0",
        "--iterations", "100",
    ]

    p = subprocess.run(cmd,text=True)
    assert p.returncode == 0, f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"

    assert_outputs_base(Path(save_dir))

def test_default_training_ddp(tmp_path: Path):
    """
    Default Integration test for `icecream train` with default setting using a single GPU. 

    This runs the training for 100 iterations on a sample pair of tomograms with angles file.
    1. It checks that the training completes successfully.
    2. It verifies the presence of expected output files in the specified save directory.
    3. It ensures that the configuration file is saved correctly.

    TODO: Maybe chaange the data to randomly generated small tomograms to speed up test.
    """
    
    # #save_dir.mkdir()

    # save_dir = './../icecream-runs/temp/'

    tomo0, tomo1, angles,_ = generate_random_data(tmp_path)

    save_dir = tmp_path / "run"




    cmd = [
         "icecream", "train",
        "--tomo0", str(tomo0),
        "--tomo1", str(tomo1),
        "--angles", str(angles),
        "--save-dir", str(save_dir),
        "--batch-size", "2",
        "--device", "0",
        "--ddp",
        "--iterations", "100",
    ]

    p = subprocess.run(cmd,text=True)
    assert p.returncode == 0, f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"

    assert_outputs_base(Path(save_dir))



def test_default_training_no_angle_file(tmp_path: Path):
    """
    integration test for `icecream train`  without angles file.
    Runs training without angles file, using tilt-min and tilt-max instead.
    """

    
    # #save_dir.mkdir()

    # save_dir = './../icecream-runs/temp/'

    tomo0, tomo1, angles,mask = generate_random_data(tmp_path)
    save_dir = tmp_path / "run"

    cmd = [
         "icecream", "train",
        "--tomo0", str(tomo0),
        "--tomo1", str(tomo1),
        "--tilt-max", "57.1",
        "--tilt-min", "-60.0",
        "--save-dir", str(save_dir),
        "--batch-size", "2",
        "--device", "0",
        "--iterations", "100",
    ]

    p = subprocess.run(cmd,text=True)
    assert p.returncode == 0, f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"

    assert_outputs_base(Path(save_dir))

    # Check that config file has tilt_min and tilt_max
    config_path_json = save_dir / "config.json"
    config_path_yaml = save_dir / "config.yaml"
    if config_path_json.exists():
        with open(config_path_json, 'r') as f:
            config = json.load(f)
    elif config_path_yaml.exists():
        import yaml
        with open(config_path_yaml, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise AssertionError("No config file found")
    
    # Check  tilt_min and tilt_max  exisits in data section
    assert 'data' in config, "Config file missing 'data' section"
    assert 'tilt_min' in config['data'], "Config file missing 'tilt_min'"
    assert 'tilt_max' in config['data'], "Config file missing 'tilt_max'"
    assert config['data']['tilt_min'] == -60.0, f"tilt_min expected -60.0, got {config['data']['tilt_min']}"
    assert config['data']['tilt_max'] == 57.1, f"tilt_max expected 57.1, got {config['data']['tilt_max']}"


def test_default_training_usr_mask(tmp_path: Path):
    """
    integration test for `icecream train`  with user provided mask.
    Runs training using a user provided mask file.
    1. It checks that the training completes successfully.
    2. It verifies the presence of expected output files in the specified save directory.
    3. It ensures that the configuration file correctly records the mask path.
    4. It checks that the mask path in the config matches the provided mask.
    """
    
    # #save_dir.mkdir()

    # save_dir = './../icecream-runs/temp/'

    tomo0, tomo1, angles,mask = generate_random_data(tmp_path)
    save_dir = tmp_path / "run_with_mask"



    cmd = [
         "icecream", "train",
        "--tomo0", str(tomo0),
        "--tomo1", str(tomo1),
        "--angles", str(angles),
        "--mask", str(mask),
        "--save-dir", str(save_dir),
        "--batch-size", "2",
        "--device", "0",
        "--iterations", "100",
    ]

    p = subprocess.run(cmd,text=True)
    assert p.returncode == 0, f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"

    assert_outputs_base(Path(save_dir))

    # Check that config file has mask path
    config_path_json = save_dir / "config.json"
    config_path_yaml = save_dir / "config.yaml"
    if config_path_json.exists():
        with open(config_path_json, 'r') as f:
            config = json.load(f)
    elif config_path_yaml.exists():
        import yaml
        with open(config_path_yaml, 'r') as f:
            config = yaml.safe_load(f)
    else:
        raise AssertionError("No config file found")
    
    # mask is a list of str paths in config
    mask = [str(mask)]
    # Check  tilt_min and tilt_max  exisits in data section
    assert 'data' in config, "Config file missing 'data' section"
    assert 'mask' in config['data'], "Config file missing 'mask'"
    assert config['data']['mask'] ==mask, f"mask expected {mask}, got {config['data']['mask']}"





def assert_outputs_base(save_dir: Path):
    # Assert config.json or config.yaml exists
    assert (save_dir / "config.json").exists() or (save_dir / "config.yaml").exists()


    # Assest tomo2_L1G1-dose_filt-bin4_EVN.rec_tomo2_L1G1-dose_filt-bin4_ODD.rec_icecream.mrc exists
    tomo_name = "tomo0_tomo1_icecream.mrc"
    assert (save_dir / tomo_name).exists(), f"Missing reconstructed tomo; files: {list(save_dir.glob('*'))}"

    # assert if model folder exists and has files
    model_dir = save_dir / "model"
    assert model_dir.exists() and model_dir.is_dir(), f"Missing model directory; files: {list(save_dir.glob('*'))}"
    model_files = list(model_dir.glob('*'))
    assert len(model_files) > 0, f"Model directory is empty; files: {model_files}"
    

    