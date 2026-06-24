"""Neural network and differentiable SoftVG-PINN constraints."""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn

from .soil_params import SOIL_BOUNDS
from .vg_mualem import vg_mualem

LayerTuple = Tuple[float, float, str]


class ModifiedMLP(nn.Module):
    """Feature-recombination MLP used as the continuous state approximator.

    The intermediate coefficient z is produced with tanh, not sigmoid; therefore
    it should be described as a learned feature-recombination coefficient rather
    than a convex mixing weight.
    """

    def __init__(self, layers: Sequence[int]):
        super().__init__()
        if len(layers) < 3:
            raise ValueError("layers must include input, at least one hidden layer, and output size.")
        if layers[-1] != 3:
            raise ValueError("The output size must be 3: raw_h, raw_K, raw_theta.")

        self.u_encoder = nn.Sequential(nn.Linear(layers[0], layers[1]), nn.Tanh())
        self.v_encoder = nn.Sequential(nn.Linear(layers[0], layers[1]), nn.Tanh())
        self.hidden_layers = nn.ModuleList(
            nn.Sequential(nn.Linear(layers[i], layers[i + 1]), nn.Tanh())
            for i in range(1, len(layers) - 2)
        )
        self.out_layer = nn.Linear(layers[-2], layers[-1])
        with torch.no_grad():
            self.out_layer.bias.copy_(torch.tensor([-5.0, -2.0, 0.0], dtype=torch.float32))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        u = self.u_encoder(inputs)
        v = self.v_encoder(inputs)
        h = u
        for layer in self.hidden_layers:
            z = layer(h)
            h = (1.0 - z) * u + z * v
        return self.out_layer(h)


class SoilPINN(nn.Module):
    """Soft-layer VG-constrained PINN for one-dimensional layered soils.

    Coordinates follow the manuscript convention: x = 0 cm at the soil surface
    and x < 0 cm downward. With this sign convention, the Richards residual is

        r = theta_t - (K h_x)_x - K_x.

    If HYDRUS depth is exported as positive downward, convert it to x = -depth
    before training. This package performs that conversion when
    depth_positive_downward=True in the YAML config.
    """

    def __init__(
        self,
        layers_config: Sequence[LayerTuple],
        net_layers: Sequence[int] = (2, 128, 128, 128, 128, 128, 3),
        interface_k: float = 5.0,
        interface_buffer_cm: float = 1.5,
    ):
        super().__init__()
        if not layers_config:
            raise ValueError("At least one soil layer is required.")

        self.layers_config = list(layers_config)
        self.num_layers = len(layers_config)
        self.interfaces = [float(layer[1]) for layer in self.layers_config[:-1]]
        self.interface_k = float(interface_k)
        self.interface_buffer_cm = float(interface_buffer_cm)
        self.net = ModifiedMLP(net_layers)

        bounds_min, bounds_max = [], []
        for top, bottom, soil in self.layers_config:
            if top < bottom:
                raise ValueError(
                    f"Invalid layer {soil}: top={top}, bottom={bottom}. "
                    "Use x=0 at surface and negative downward, e.g. top=0, bottom=-50."
                )
            if soil not in SOIL_BOUNDS:
                raise KeyError(f"Unknown soil type '{soil}'. Add it to SOIL_BOUNDS first.")
            b = SOIL_BOUNDS[soil]
            bounds_min.append([b["tr"][0], b["ts"][0], b["a"][0], b["n"][0], b["k"][0]])
            bounds_max.append([b["tr"][1], b["ts"][1], b["a"][1], b["n"][1], b["k"][1]])

        self.register_buffer("bounds_min", torch.tensor(bounds_min, dtype=torch.float32))
        self.register_buffer("bounds_max", torch.tensor(bounds_max, dtype=torch.float32))
        self.raw_params = nn.Parameter(torch.zeros(self.num_layers, 5, dtype=torch.float32))

    def get_physics_params(self) -> torch.Tensor:
        """Return bounded VG parameters for each layer."""
        return self.bounds_min + (self.bounds_max - self.bounds_min) * torch.sigmoid(self.raw_params)

    def get_parameter_table(self) -> List[Dict[str, float | int | str]]:
        params = self.get_physics_params().detach().cpu().numpy()
        rows: List[Dict[str, float | int | str]] = []
        for i, (top, bottom, soil) in enumerate(self.layers_config):
            theta_r, theta_s, alpha, n, ks = params[i]
            rows.append({
                "layer": i + 1,
                "top_cm": float(top),
                "bottom_cm": float(bottom),
                "soil": soil,
                "theta_r": float(theta_r),
                "theta_s": float(theta_s),
                "alpha_cm^-1": float(alpha),
                "n": float(n),
                "Ks_cm_day^-1": float(ks),
            })
        return rows

    def compute_prior_loss(self) -> torch.Tensor:
        """L2 regularization on raw parameters; values near 0 are prior midpoints."""
        return torch.mean(self.raw_params ** 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return h, K, theta.

        h(x,t) = -(15000 sigmoid(z_h) + 0.1)
        K(x,t) = 2000 sigmoid(z_K) + eps
        theta(x,t) = 0.01 + 0.55 sigmoid(z_theta)
        """
        inputs = torch.cat([x / 100.0, t / 100.0], dim=1)
        raw_h, raw_k, raw_theta = torch.split(self.net(inputs), 1, dim=1)
        h = -(15000.0 * torch.sigmoid(raw_h) + 0.1)
        k = 2000.0 * torch.sigmoid(raw_k) + 1.0e-10
        theta = 0.01 + 0.55 * torch.sigmoid(raw_theta)
        return h, k, theta

    def get_masks(self, x: torch.Tensor) -> torch.Tensor:
        """Return continuous soft-layer masks.

        For x=0 at the surface and x<0 downward, interfaces are negative depths.
        Example: layers 0--50, 50--150, 150--400 cm are represented by
        interfaces x=-50 and x=-150. The first mask approaches 1 above the first
        interface; the last mask approaches 1 below the deepest interface.
        """
        if self.num_layers == 1:
            return torch.ones((x.shape[0], 1), dtype=x.dtype, device=x.device)

        s = [torch.sigmoid((x - interface) * self.interface_k) for interface in self.interfaces]
        masks = [s[0]]
        for i in range(1, self.num_layers - 1):
            masks.append(s[i] - s[i - 1])
        masks.append(1.0 - s[-1])
        return torch.cat(masks, dim=1)

    def compute_vg_targets(self, h: torch.Tensor, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        params = self.get_physics_params()
        masks = self.get_masks(x)

        theta_r = torch.sum(masks * params[:, 0], dim=1, keepdim=True)
        theta_s = torch.sum(masks * params[:, 1], dim=1, keepdim=True)
        alpha = torch.sum(masks * params[:, 2], dim=1, keepdim=True)
        n = torch.sum(masks * params[:, 3], dim=1, keepdim=True)
        ks = torch.sum(masks * params[:, 4], dim=1, keepdim=True)

        theta_vg, k_vg, _ = vg_mualem(h, theta_r, theta_s, alpha, n, ks)
        return theta_vg, k_vg

    def predict_theta(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        _, _, theta = self(x, t)
        return theta

    def compute_losses(
        self,
        x_train: torch.Tensor,
        t_train: torch.Tensor,
        theta_obs: torch.Tensor,
        x_pde: torch.Tensor,
        t_pde: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute data, PDE and VG losses.

        The conductivity part of the VG loss uses natural logarithms. Reporting
        metrics may use log10 K(h); the change of logarithm base only rescales
        the squared error by a constant and does not change curve ranking.
        """
        _, _, theta_pred = self(x_train, t_train)
        loss_data = torch.mean((theta_pred - theta_obs) ** 2)

        x_pde = x_pde.detach().clone().requires_grad_(True)
        t_pde = t_pde.detach().clone().requires_grad_(True)
        h_pde, k_pde, theta_pde = self(x_pde, t_pde)

        theta_vg, k_vg = self.compute_vg_targets(h_pde, x_pde)
        loss_vg_theta = torch.mean((theta_pde - theta_vg) ** 2)
        loss_vg_k = torch.mean((torch.log(k_pde + 1.0e-12) - torch.log(k_vg + 1.0e-12)) ** 2)
        loss_vg = loss_vg_theta + 0.1 * loss_vg_k

        theta_t = torch.autograd.grad(theta_pde, t_pde, torch.ones_like(theta_pde), create_graph=True)[0]
        h_x = torch.autograd.grad(h_pde, x_pde, torch.ones_like(h_pde), create_graph=True)[0]
        k_hx = k_pde * h_x
        k_hx_x = torch.autograd.grad(k_hx, x_pde, torch.ones_like(k_hx), create_graph=True)[0]
        k_x = torch.autograd.grad(k_pde, x_pde, torch.ones_like(k_pde), create_graph=True)[0]

        pde_residual = (theta_t - k_hx_x - k_x) / (k_pde.detach() + 1.0)

        # Exclude a narrow buffer around layer interfaces from the second-order
        # PDE residual. Use a weighted mean so the buffer does not dilute the loss.
        valid_region = torch.ones_like(x_pde)
        for interface in self.interfaces:
            valid_region *= (torch.abs(x_pde - interface) > self.interface_buffer_cm).float()
        loss_pde = torch.sum((pde_residual ** 2) * valid_region) / (torch.sum(valid_region) + 1.0e-12)

        return loss_data, loss_pde, loss_vg
