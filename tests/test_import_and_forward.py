import torch

from softvg_pinn.model import SoilPINN


def test_forward_shapes():
    model = SoilPINN(
        [(0.0, -50.0, "Silt"), (-50.0, -150.0, "Sand")],
        net_layers=[2, 8, 8, 3],
    )
    x = torch.tensor([[0.0], [-20.0], [-100.0]], dtype=torch.float32)
    t = torch.tensor([[0.0], [1.0], [2.0]], dtype=torch.float32)
    h, k, theta = model(x, t)
    assert h.shape == k.shape == theta.shape == (3, 1)
    assert torch.all(h < 0)
    assert torch.all(k > 0)
    assert torch.all((theta > 0.01) & (theta < 0.56))
    assert model.get_physics_params().shape == (2, 5)
