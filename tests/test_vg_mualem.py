import torch

from softvg_pinn.vg_mualem import vg_mualem


def test_vg_mualem_shapes_and_bounds():
    h = torch.tensor([[-10.0], [-100.0]], dtype=torch.float32)
    tr = torch.full_like(h, 0.05)
    ts = torch.full_like(h, 0.43)
    alpha = torch.full_like(h, 0.02)
    n = torch.full_like(h, 1.5)
    ks = torch.full_like(h, 10.0)
    theta, k, se = vg_mualem(h, tr, ts, alpha, n, ks)
    assert theta.shape == k.shape == se.shape == h.shape
    assert torch.all(theta >= tr)
    assert torch.all(theta <= ts)
    assert torch.all(k > 0)
