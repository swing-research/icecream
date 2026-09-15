"""predict_dir blend: keep the Hann blend when no_window is true, and never leave zero-weight holes
between tiles (the boxcar loss window only covers the central crop//2 cube, so stride must not exceed it).
Regression for the end-of-training reconstruction: train.py passes crop_size, which used to rebuild the
boxcar window and, at stride 48 / crop 72, left 12-voxel grids of exact zeros."""
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icecream.models import get_model
from icecream.trainer.eq_trainer import EquivariantTrainer


class Ones(torch.nn.Module):
    """Stand-in network: a constant 1 output blends to exactly 1 wherever tiles cover, 0 in holes."""
    def forward(self, x):
        return torch.ones_like(x)

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason='needs cuda:1')
CROP = 24


def _trainer(**flags):
    cfg = dict(device=1, load_device=False, learning_rate=1e-4, crop_size=CROP, wedge_double_size=True,
               use_spherical_support=True, wedge_low_support=0, ref_wedge_support=1, wedge_taper=0,
               no_window=True, window_taper=0, use_fourier=False, view_as_real=False, use_mixed_precision=True,
               use_flips=True, min_distance=0.5, eq_weight=2.0, eq_use_direct=False, upsample_volume=False,
               eq_stop_grad_inner=False, eq_single_direction=False)
    cfg.update(flags)
    torch.manual_seed(0)
    model = get_model(name='unet3d_bf', num_levels=2, f_maps=4, layer_order='cr', use_bias=False,
                      padding_mode='zeros', pool_type='max', dropout_prob=0.0, upsample='default')
    t = EquivariantTrainer(configs=SimpleNamespace(**cfg), model=model, angle_max_set=[60], angle_min_set=[-60])
    t.model = Ones().to(t.device)  # the real (random) net can emit exact fp16 zeros, which would look like holes
    # predict_dir only reads the volumes; skip the dataset machinery
    t.vol_data = SimpleNamespace(volume_1_set=[torch.randn(40, 52, 64)], volume_2_set=[])
    return t


def _predict(t):
    # as train.py calls it: crop_size copied from train_params, stride = crop * 2 // 3 (16 here, > crop // 2)
    est, = t.predict_dir(crop_size=CROP, stride=CROP * 2 // 3, batch_size=4, pre_pad=True)
    return est


def test_no_window_keeps_hann_blend_without_holes():
    t = _trainer(no_window=True)
    est = _predict(t)
    assert t.window is None
    assert est.shape == (40, 52, 64)
    np.testing.assert_allclose(est, 1.0, atol=1e-5)


def test_boxcar_blend_clamps_stride_without_holes():
    t = _trainer(no_window=False)
    est = _predict(t)
    assert t.window is not None and t.window.shape == (CROP, CROP, CROP)
    np.testing.assert_allclose(est, 1.0, atol=1e-5)
