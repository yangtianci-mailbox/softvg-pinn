"""Evaluation metrics."""
from __future__ import annotations

from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

from .soil_params import TRUE_PARAMS


def regression_metrics(y_true, y_pred) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mae = float(np.mean(np.abs(err)))
    denom = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = float(1.0 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")
    return {"rmse": rmse, "mae": mae, "r2": r2}


def vg_curve_values(params: Sequence[float], h_abs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    theta_r, theta_s, alpha, n, ks = [float(x) for x in params]
    m = 1.0 - 1.0 / n
    se = (1.0 + (alpha * h_abs) ** n) ** (-m)
    se = np.clip(se, 1.0e-12, 1.0 - 1.0e-12)
    theta = theta_r + (theta_s - theta_r) * se
    k = ks * se ** 0.5 * (1.0 - (1.0 - se ** (1.0 / m)) ** m) ** 2
    return theta, k


def vg_curve_metrics(model, layers_config, h_abs: np.ndarray | None = None) -> List[Dict[str, float | str | int]]:
    """Compute synthetic-benchmark errors against TRUE_PARAMS where available.

    Conductivity is reported as log10 K(h) MSE, even though training uses ln K.
    """
    if h_abs is None:
        h_abs = np.logspace(-2, 4, 1000)

    params = model.get_physics_params().detach().cpu().numpy()
    rows = []
    for i, (_, _, soil) in enumerate(layers_config):
        if soil not in TRUE_PARAMS:
            continue
        theta, k = vg_curve_values(params[i], h_abs)
        theta_ref, k_ref = vg_curve_values(TRUE_PARAMS[soil], h_abs)
        rows.append({
            "layer": i + 1,
            "soil": soil,
            "theta_h_mse": float(np.mean((theta - theta_ref) ** 2)),
            "log10K_h_mse": float(np.mean((np.log10(k + 1.0e-15) - np.log10(k_ref + 1.0e-15)) ** 2)),
        })
    return rows
