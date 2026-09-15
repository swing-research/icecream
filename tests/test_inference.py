"""GPU tiled inference (utils.inference_util.inference) against a plain per-tile reference."""
import numpy as np
import pytest
import torch
import torch.nn.functional as F

from icecream.models import get_model
from icecream.utils.inference_util import inference, compute_padd
from icecream.utils.utils import get_wedge_3d_new, symmetrize_3D, get_measurement

pytestmark = pytest.mark.skipif(torch.cuda.device_count() < 2, reason='needs cuda:1')
DEVICE = torch.device('cuda:1')
SHAPE, SIZE, STRIDE, PRE_PAD = (60, 76, 92), 24, 12, 6


def _setup():
    torch.manual_seed(0)
    model = get_model(name='unet3d_bf', num_levels=2, f_maps=4, layer_order='cr', use_bias=False,
                      padding_mode='zeros', pool_type='max', dropout_prob=0.0, upsample='default')
    model = model.to(DEVICE).eval()
    vol = torch.randn(SHAPE)
    # double-size binary wedge, built as the trainer does
    w, _ = get_wedge_3d_new(2 * SIZE, 60, -60, use_spherical_support=True)
    w = torch.tensor(w, dtype=torch.float32)
    w[w > 0.1] = 1
    w = w[:-1, :-1, :-1]
    w = (w + symmetrize_3D(w)) / 2
    w[w > 0.1] = 1
    return model, vol, w.to(DEVICE)


def _reference(vol, model, window, wedge):
    """Straightforward loop: same padding, model -> wedge -> model per tile, blend on the CPU."""
    v = F.pad(vol.to(torch.float16), (PRE_PAD, 0) * 3)
    pads = [compute_padd(n, SIZE, STRIDE) for n in v.shape]
    v = F.pad(v, (0, pads[2], 0, pads[1], 0, pads[0]))
    est, wt = torch.zeros(v.shape), torch.zeros(v.shape)
    for i in range(0, v.shape[0] - SIZE + 1, STRIDE):
        for j in range(0, v.shape[1] - SIZE + 1, STRIDE):
            for k in range(0, v.shape[2] - SIZE + 1, STRIDE):
                crop = v[i:i + SIZE, j:j + SIZE, k:k + SIZE][None, None].to(DEVICE)
                with torch.no_grad():
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        out = model(crop)[:, 0]
                    out = get_measurement(out.float(), wedge)
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        out = model(out[:, None])[:, 0]
                out = out.float().cpu()[0]
                est[i:i + SIZE, j:j + SIZE, k:k + SIZE] += out * window
                wt[i:i + SIZE, j:j + SIZE, k:k + SIZE] += window
    wt[wt == 0] = 1
    est = est / wt
    return est[PRE_PAD:PRE_PAD + SHAPE[0], PRE_PAD:PRE_PAD + SHAPE[1], PRE_PAD:PRE_PAD + SHAPE[2]].numpy()


def _run(model, vol, wedge, **kw):
    est, _ = inference(vol, model, SIZE, STRIDE, batch_size=7, wedge=wedge, pre_pad=True,
                       pre_pad_size=PRE_PAD, device=DEVICE, **kw)
    return est


def _rel_max_diff(a, b):
    return float(np.abs(a - b).max() / np.abs(b).max())


def test_boxcar_window_matches_reference():
    model, vol, wedge = _setup()
    window = torch.zeros(SIZE, SIZE, SIZE)
    window[SIZE // 4:-SIZE // 4, SIZE // 4:-SIZE // 4, SIZE // 4:-SIZE // 4] = 1
    ref = _reference(vol, model, window, wedge)
    est = _run(model, vol, wedge, window=window.to(DEVICE))
    assert est.shape == SHAPE
    err = _rel_max_diff(est, ref)
    print(f'boxcar: max |new - ref| / max |ref| = {err:.3e}')
    assert err < 2e-2, err


def test_blend_ones_matches_reference():
    model, vol, wedge = _setup()
    ref = _reference(vol, model, torch.ones(SIZE, SIZE, SIZE), wedge)
    est = _run(model, vol, wedge, window=None, blend='ones')
    err = _rel_max_diff(est, ref)
    print(f'ones: max |new - ref| / max |ref| = {err:.3e}')
    assert err < 2e-2, err


def test_blend_hann_is_finite():
    model, vol, wedge = _setup()
    est = _run(model, vol, wedge)  # window=None, blend='hann' (default)
    assert est.shape == SHAPE and est.dtype == np.float32
    assert np.isfinite(est).all()
    assert np.abs(est).max() > 0
