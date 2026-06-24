"""Differentiable van Genuchten--Mualem hydraulic functions."""
from __future__ import annotations

from typing import Tuple

import torch


def vg_mualem(
    h: torch.Tensor,
    theta_r: torch.Tensor,
    theta_s: torch.Tensor,
    alpha: torch.Tensor,
    n: torch.Tensor,
    ks: torch.Tensor,
    pore_connectivity: float = 0.5,
    eps: float = 1.0e-6,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return theta(h), K(h), and effective saturation.

    h is pressure head in cm. Negative pressure heads are converted using |h|,
    which matches the unsaturated retention curve used in the manuscript.
    """
    m = 1.0 - 1.0 / n
    se = (1.0 + (alpha * torch.abs(h)) ** n) ** (-m)
    se = torch.clamp(se, eps, 1.0 - eps)

    theta = theta_r + (theta_s - theta_r) * se
    inner = torch.clamp(1.0 - se ** (1.0 / m), min=eps, max=1.0)
    k = ks * (se ** pore_connectivity) * (1.0 - inner ** m) ** 2
    return theta, k, se
