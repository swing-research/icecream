import torch
from types import SimpleNamespace

from icecream.trainer.base_trainer import BaseTrainer
from icecream.utils.utils import get_wedge_3d_new


def _fake_trainer():
    t = SimpleNamespace(configs=SimpleNamespace(wedge_low_support=0, use_spherical_support=True), device='cpu')
    t.initialize_wedge = BaseTrainer.initialize_wedge.__get__(t)
    t.get_real_binary_filter = BaseTrainer.get_real_binary_filter.__get__(t)
    return t


def test_wedge_input_is_not_dilated_and_is_hermitian():
    # Same path as EquivariantTrainer.setup: odd wedge at 2*crop, trim to even, symmetrize.
    crop = 72
    t = _fake_trainer()
    wedge_input = t.get_real_binary_filter(t.initialize_wedge(60, -60, 2 * crop)[:-1, :-1, :-1])

    raw, _ = get_wedge_3d_new(2 * crop, 60, -60, use_spherical_support=True)
    raw = (torch.tensor(raw)[:-1, :-1, :-1] > 0.1).sum()
    # Only voxels lying exactly on the wedge planes may be added by symmetrization.
    assert (wedge_input.sum() - raw) / raw < 1e-4

    x = torch.fft.ifftn(torch.fft.ifftshift(wedge_input.double()))
    assert x.imag.abs().max() < 1e-12 * x.real.abs().max()
