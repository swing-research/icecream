"""
eq_stop_grad_inner / eq_single_direction flags of EquivariantTrainer.compute_loss.
CPU, float64, tiny model; no dataset code involved.
"""
import subprocess
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from icecream.models import get_model
from icecream.trainer.eq_trainer import EquivariantTrainer
from icecream.utils.utils import generate_all_cube_symmetries_torch

CROP = 24
PRE_CHANGE_COMMIT = "2d98cf7"  # the fixes commit right below this one on sg-variant: last commit before the flags


def _configs(**flags):
    cfg = dict(device=-1, load_device=False, learning_rate=1e-4, crop_size=CROP, wedge_double_size=True,
               use_spherical_support=True, wedge_low_support=0, ref_wedge_support=1, wedge_taper=0,
               no_window=True, window_taper=0, use_fourier=False, view_as_real=False, use_mixed_precision=False,
               use_flips=True, min_distance=0.5, eq_weight=2.0, eq_use_direct=False,
               eq_stop_grad_inner=False, eq_single_direction=False)
    cfg.update(flags)
    return SimpleNamespace(**cfg)


def _trainer(cls=EquivariantTrainer, **flags):
    torch.manual_seed(0)
    model = get_model(name='unet3d_bf', num_levels=2, f_maps=4, layer_order='cr', use_bias=False,
                      padding_mode='zeros', pool_type='max', dropout_prob=0.0, upsample='default').double()
    t = cls(configs=_configs(**flags), model=model, angle_max_set=[60], angle_min_set=[-60])
    # compute_loss reads self.k_sets, normally built by the dataset from a random cube and the input wedge.
    torch.manual_seed(0)
    cube = torch.rand(CROP, CROP, CROP)
    wedge = (t.wedge_input_set[0] > 0).float()
    _, _, t.k_sets, _ = generate_all_cube_symmetries_torch(cube, wedge, use_flips=True, min_distance=0.5)
    return t


def _inputs():
    torch.manual_seed(1)
    return torch.randn(2, CROP, CROP, CROP, dtype=torch.float64), torch.randn(2, CROP, CROP, CROP, dtype=torch.float64)


def _loss(t, x1, x2, seed):
    # Rotations are drawn with numpy; the direction s of eq_single_direction with torch.rand.
    np.random.seed(seed)
    torch.manual_seed(seed)
    return t.compute_loss(x1, x2, 0)


def _grad(t, x1, x2, seed):
    t.model.zero_grad()
    loss = _loss(t, x1, x2, seed)
    loss.backward()
    return torch.cat([p.grad.flatten() for p in t.model.parameters()]), loss.detach()


def _old_trainer_cls():
    root = Path(__file__).resolve().parents[1]
    try:
        src = subprocess.check_output(['git', 'show', f'{PRE_CHANGE_COMMIT}:src/icecream/trainer/eq_trainer.py'],
                                      cwd=root, text=True, stderr=subprocess.DEVNULL)
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("pre-change eq_trainer.py not available from git")
    src = src.replace('from .base_trainer import BaseTrainer', 'from icecream.trainer.base_trainer import BaseTrainer')
    mod = types.ModuleType('eq_trainer_old')
    exec(compile(src, 'eq_trainer_old.py', 'exec'), mod.__dict__)
    return mod.EquivariantTrainer


def test_all_flags_false_matches_pre_change_code():
    x1, x2 = _inputs()
    new, old = _trainer(), _trainer(_old_trainer_cls())
    l_new, l_old = _loss(new, x1, x2, 2), _loss(old, x1, x2, 2)
    assert torch.allclose(l_new, l_old)
    assert torch.allclose(new.obs_loss_set[-1], old.obs_loss_set[-1])
    assert torch.allclose(new.equi_loss_set[-1], old.equi_loss_set[-1])
    assert torch.allclose(new.diff_loss_set[-1], old.diff_loss_set[-1])


@pytest.mark.parametrize('eq_use_direct', [False, True])
def test_single_direction_averages_to_two_directions(eq_use_direct):
    x1, x2 = _inputs()
    two = _trainer(eq_use_direct=eq_use_direct)
    one = _trainer(eq_use_direct=eq_use_direct, eq_single_direction=True)

    def s_of(seed):  # mirrors the draw in compute_loss
        torch.manual_seed(seed)
        return int(torch.rand(1).item() < 0.5)

    seeds = {s_of(k): k for k in range(20)}
    assert set(seeds) == {0, 1}
    l_two = _loss(two, x1, x2, 3)
    # same numpy seed -> same rotations; the torch seed only picks the direction
    np.random.seed(3); torch.manual_seed(seeds[0]); l_s0 = one.compute_loss(x1, x2, 0)
    np.random.seed(3); torch.manual_seed(seeds[1]); l_s1 = one.compute_loss(x1, x2, 0)
    assert not torch.allclose(l_s0, l_s1)
    assert torch.allclose((l_s0 + l_s1) / 2, l_two)
    for name in ('obs_loss_set', 'equi_loss_set', 'diff_loss_set'):
        a, b = getattr(one, name)
        assert torch.allclose((a + b) / 2, getattr(two, name)[-1]), name


@pytest.mark.parametrize('flag', ['eq_stop_grad_inner'])
def test_stop_gradient_flags_keep_loss_but_change_gradient(flag):
    x1, x2 = _inputs()
    g_base, l_base = _grad(_trainer(), x1, x2, 5)
    g_flag, l_flag = _grad(_trainer(**{flag: True}), x1, x2, 5)
    assert torch.allclose(l_base, l_flag)
    assert not torch.allclose(g_base, g_flag)
