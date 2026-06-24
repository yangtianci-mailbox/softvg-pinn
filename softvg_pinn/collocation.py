"""Collocation point generation."""
from __future__ import annotations

from typing import Sequence, Tuple

import torch


def generate_pde_colpoints(
    n_points: int,
    x_min: float,
    x_max: float,
    t_min: float,
    t_max: float,
    interfaces: Sequence[float],
    device: torch.device,
    local_fraction: float = 0.30,
    local_half_width_cm: float = 5.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if n_points <= 0:
        raise ValueError("n_points must be positive.")
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if t_max < t_min:
        t_min, t_max = t_max, t_min

    if not interfaces:
        x_f = (x_max - x_min) * torch.rand(n_points, 1, dtype=torch.float32, device=device) + x_min
        t_f = (t_max - t_min) * torch.rand(n_points, 1, dtype=torch.float32, device=device) + t_min
        return x_f, t_f

    n_local_total = int(n_points * local_fraction)
    n_global = max(1, n_points - n_local_total)

    x_global = (x_max - x_min) * torch.rand(n_global, 1, dtype=torch.float32, device=device) + x_min
    t_global = (t_max - t_min) * torch.rand(n_global, 1, dtype=torch.float32, device=device) + t_min

    n_local_per = max(1, n_local_total // len(interfaces))
    x_local, t_local = [], []
    for interface in interfaces:
        x = 2.0 * local_half_width_cm * torch.rand(n_local_per, 1, dtype=torch.float32, device=device)
        x = x + (interface - local_half_width_cm)
        x = torch.clamp(x, min=x_min, max=x_max)
        t = (t_max - t_min) * torch.rand(n_local_per, 1, dtype=torch.float32, device=device) + t_min
        x_local.append(x)
        t_local.append(t)

    x_f = torch.cat([x_global] + x_local, dim=0)[:n_points]
    t_f = torch.cat([t_global] + t_local, dim=0)[:n_points]
    return x_f, t_f
