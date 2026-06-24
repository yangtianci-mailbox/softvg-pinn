import torch

from softvg_pinn.model import SoilPINN


def test_soft_masks_sum_to_one():
    model = SoilPINN(
        [(0.0, -50.0, "Silt"), (-50.0, -150.0, "Sand"), (-150.0, -400.0, "Sandy Loam")],
        net_layers=[2, 8, 8, 3],
        interface_k=10.0,
    )
    x = torch.tensor([[0.0], [-25.0], [-100.0], [-300.0]], dtype=torch.float32)
    masks = model.get_masks(x)
    assert masks.shape == (4, 3)
    assert torch.allclose(masks.sum(dim=1), torch.ones(4), atol=1.0e-5)
    assert masks[0, 0] > 0.9
    assert masks[-1, -1] > 0.9
