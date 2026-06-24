"""Plotting utilities. All figures are generated from numeric model outputs."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .metrics import vg_curve_values
from .soil_params import TRUE_PARAMS


def plot_vg_curves(model, layers_config, output_dir: str | Path, true_params: bool = True) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    h_vals = np.logspace(-2, 4, 500)
    params = model.get_physics_params().detach().cpu().numpy()
    num_layers = len(layers_config)

    fig, axs = plt.subplots(num_layers, 2, figsize=(12, max(4, 3.8 * num_layers)))
    if num_layers == 1:
        axs = np.expand_dims(axs, axis=0)

    for i, layer in enumerate(layers_config):
        soil = layer[2]
        theta, k = vg_curve_values(params[i], h_vals)
        axs[i, 0].plot(theta, h_vals, label=f"Estimated layer {i + 1}")
        axs[i, 1].plot(theta, k, label=f"Estimated layer {i + 1}")

        if true_params and soil in TRUE_PARAMS:
            theta_ref, k_ref = vg_curve_values(TRUE_PARAMS[soil], h_vals)
            axs[i, 0].plot(theta_ref, h_vals, linestyle="--", label=f"Reference {soil}")
            axs[i, 1].plot(theta_ref, k_ref, linestyle="--", label=f"Reference {soil}")

        axs[i, 0].set_yscale("log")
        axs[i, 1].set_yscale("log")
        axs[i, 0].set_xlabel("Volumetric water content, θ")
        axs[i, 0].set_ylabel("|h| (cm)")
        axs[i, 1].set_xlabel("Volumetric water content, θ")
        axs[i, 1].set_ylabel("K (cm day$^{-1}$)")
        axs[i, 0].set_title(f"Layer {i + 1}: {soil} retention curve")
        axs[i, 1].set_title(f"Layer {i + 1}: {soil} conductivity curve")
        axs[i, 0].grid(True)
        axs[i, 1].grid(True)
        axs[i, 0].legend()
        axs[i, 1].legend()

    fig.tight_layout()
    out = output_dir / f"vg_curves_{num_layers}_layers.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_water_content(
    df: pd.DataFrame,
    depths: Sequence[float],
    col_x: str,
    col_t: str,
    col_obs: str,
    col_pred: str,
    output_dir: str | Path,
) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if not depths:
        depths = sorted(df[col_x].unique())[:8]

    fig, axs = plt.subplots(len(depths), 1, figsize=(9, max(3, 2.4 * len(depths))), sharex=True)
    if len(depths) == 1:
        axs = [axs]

    for ax, depth in zip(axs, depths):
        sub = df[np.isclose(df[col_x], depth, atol=1.0e-6)].sort_values(col_t)
        if sub.empty:
            ax.text(0.5, 0.5, f"No data at x={depth:g} cm", transform=ax.transAxes, ha="center")
            continue
        ax.plot(sub[col_t], sub[col_obs], label="Observed")
        ax.plot(sub[col_t], sub[col_pred], linestyle="--", label="Reconstructed")
        ax.set_ylabel("θ")
        ax.set_title(f"Depth x = {depth:g} cm")
        ax.grid(True)
        ax.legend()

    axs[-1].set_xlabel("Time (day)")
    fig.tight_layout()
    out = output_dir / "water_content_reconstruction.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return out
