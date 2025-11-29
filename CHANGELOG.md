# Changelog
All notable changes to this project will be documented in this file.

The format is based on Keep a Changelog (https://keepachangelog.com)
and uses categories like Added, Changed, Fixed, Removed.

## [Unreleased]  ← optional section

## [v0.3] - 2025-11-05
### Added
- Allow multiple volumes for training and testing.  
  - Iterations now scale independently of number of volumes.  
  - Command-line option can be repeated (`--tomo1 path1 --tomo1 path2 ...`).
- Predict command now supports a single tomogram as input.
- Save CSV and PNG plots of loss during training (requires matplotlib).
- `train_params.load_device` to load tomograms/masks directly on GPU.
- `predict_params.iter_load` to choose which model to load.
- `predict_params.save_dir_reconstructions` to save reconstructions elsewhere.

- Parameters for mask generation:  
  `mask_tomo_side`, `mask_tomo_density_perc`, `mask_tomo_std_perc`.

### Changed
- Renamed `scale` → `eq_weight` for clarity.
- `num_workers` is now fixed to 0 due to performance issues.
- Integrated `pretrain_params` into `train_params`.

### Removed
- Mask argument from the predict command (training-only parameter).

---

## [v0.2] - 2025-10-23
### Added
- `save_tomo_n_iterations` to compute/save intermediate reconstructions.
- `split-tilt-series` command to split a tilt series along the angle dimension.
- Comments added to the default YAML file parameters.
- Support for using a pre-trained model as initialization.
- Optional `torch.compile` flag.

### Changed
- Code clean-up and refactoring.
