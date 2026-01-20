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
    mask01 = tmp_path / "mask.mrc"
    angles01 = tmp_path / "angles.tlt"

    tomo2 = tmp_path / "tomo2.mrc"
    tomo3 = tmp_path / "tomo3.mrc"
    mask23 = tmp_path / "mask2.mrc"
    angles23 = tmp_path / "angles2.tlt"

    # generate random small tomograms for faster testing


    with mrcfile.new(tomo0, overwrite=True) as mrc:
        small_data = np.random.rand(256,512, 512).astype(np.float32)
        mrc.set_data(small_data)

    with mrcfile.new(tomo1, overwrite=True) as mrc:
        small_data = np.random.rand(256,512, 512).astype(np.float32)
        mrc.set_data(small_data)

    with mrcfile.new(mask01, overwrite=True) as mrc:
        small_data = np.zeros((256,512, 512), dtype=np.float32)
        small_data[64:-64, 32:-32, 32:-32] = 1.0  # simple cubic mask
        mrc.set_data(small_data)

    anlges_array = np.linspace(-60, 57.1, num=120).astype(np.float32)
    np.savetxt(angles01, anlges_array, fmt='%.4f')


    with mrcfile.new(tomo2, overwrite=True) as mrc:
        small_data = np.random.rand(256,512, 512).astype(np.float32)
        mrc.set_data(small_data)

    with mrcfile.new(tomo3, overwrite=True) as mrc:
        small_data = np.random.rand(256,512, 512).astype(np.float32)
        mrc.set_data(small_data)

    with mrcfile.new(mask23, overwrite=True) as mrc:
        small_data = np.zeros((256,512, 512), dtype=np.float32)
        small_data[64:-64, 32:-32, 32:-32] = 1.0  # simple cubic mask
        mrc.set_data(small_data)

    anlges_array = np.linspace(-60, 57.1, num=120).astype(np.float32)
    np.savetxt(angles23, anlges_array, fmt='%.4f')

    return tomo0, tomo1, angles01, mask01, tomo2, tomo3, angles23, mask23

def test_default_training(tmp_path: Path):
    
    # #save_dir.mkdir()

    # save_dir = './../icecream-runs/temp/'


    tomo0, tomo1, angles01,_, tomo2, tomo3, angles23,_ = generate_random_data(tmp_path)

    save_dir = tmp_path / "run"

    cmd = [
         "icecream", "train",
        "--tomo0", str(tomo0),
        "--tomo1", str(tomo1),
        "--tomo0", str(tomo2),
        "--tomo1", str(tomo3),
        "--angles", str(angles01),
        "--angles", str(angles23),
        "--save-dir", str(save_dir),
        "--batch-size", "2",
        "--device", "0",
        "--iterations", "100",
    ]

    p = subprocess.run(cmd,text=True)
    assert p.returncode == 0, f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"

    assert_outputs_base(Path(save_dir))



def test_default_training_no_angle_file(tmp_path: Path):

    tomo0, tomo1, angles01,_, tomo2, tomo3, angles23,_ = generate_random_data(tmp_path)

    save_dir = tmp_path / "run"
    cmd = [
         "icecream", "train",
        "--tomo0", str(tomo0),
        "--tomo1", str(tomo1),
        "--tomo0", str(tomo2),
        "--tomo1", str(tomo3),
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
    # #save_dir.mkdir()

    # save_dir = './../icecream-runs/temp/'


    tomo0, tomo1, angles01,mask01, tomo2, tomo3, angles23,mask23 = generate_random_data(tmp_path)
    save_dir = tmp_path / "run_with_mask"



    cmd = [
         "icecream", "train",
        "--tomo0", str(tomo0),
        "--tomo1", str(tomo1),
        "--tomo0", str(tomo2),
        "--tomo1", str(tomo3),
        "--angles", str(angles01),
        "--angles", str(angles23),
        "--mask", str(mask01),
        "--mask", str(mask23),
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
    mask = [str(mask01), str(mask23)]
    # Check  tilt_min and tilt_max  exisits in data section
    assert 'data' in config, "Config file missing 'data' section"
    assert 'mask' in config['data'], "Config file missing 'mask'"


    # Length of mask list should be 2
    assert len(config['data']['mask']) == 2, f"mask length expected 2, got {len(config['data']['mask'])}"
    assert config['data']['mask'][0] ==mask[0], f"mask expected {mask}, got {config['data']['mask']}"
    assert config['data']['mask'][1] ==mask[1], f"mask expected {mask}, got {config['data']['mask']}"



def test_default_ddp_training_single_gpu(tmp_path: Path):
    # #save_dir.mkdir()

    # save_dir = './../icecream-runs/temp/'


    tomo0, tomo1, angles01,mask01, tomo2, tomo3, angles23,mask23 = generate_random_data(tmp_path)
    save_dir = tmp_path / "run"



    cmd = [
         "icecream", "train",
        "--tomo0", str(tomo0),
        "--tomo1", str(tomo1),
        "--tomo0", str(tomo2),
        "--tomo1", str(tomo3),
        "--angles", str(angles01),
        "--angles", str(angles23),
        "--save-dir", str(save_dir),
        "--batch-size", "2",
        "--device", "0",
        "--ddp",
        "--iterations", "100",
    ]

    p = subprocess.run(cmd,text=True)
    assert p.returncode == 0, f"STDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"

    assert_outputs_base(Path(save_dir))





def assert_outputs_base(save_dir: Path):
    # Assert config.json or config.yaml exists
    assert (save_dir / "config.json").exists() or (save_dir / "config.yaml").exists()


    # Assest tomo2_L1G1-dose_filt-bin4_EVN.rec_tomo2_L1G1-dose_filt-bin4_ODD.rec_icecream.mrc exists
    tomo_name = "tomo0_tomo1_icecream.mrc"
    assert (save_dir / tomo_name).exists(), f"Missing reconstructed tomo; files: {list(save_dir.glob('*'))}"


    tomo_name2 = "tomo2_tomo3_icecream.mrc"
    assert (save_dir / tomo_name2).exists(), f"Missing reconstructed tomo; files: {list(save_dir.glob('*'))}"

    # assert if model folder exists and has files
    model_dir = save_dir / "model"
    assert model_dir.exists() and model_dir.is_dir(), f"Missing model directory; files: {list(save_dir.glob('*'))}"
    model_files = list(model_dir.glob('*'))
    assert len(model_files) > 0, f"Model directory is empty; files: {model_files}"
    

    

