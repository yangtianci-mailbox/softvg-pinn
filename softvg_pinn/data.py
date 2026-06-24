"""Data loading and train/evaluation split utilities."""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch

from .config import TrainingConfig, ValidationConfig


def read_observation_table(path: str | Path, sheet_name: str | None = None) -> pd.DataFrame:
    """Read water-content observations from CSV/TXT/Excel.

    The function deliberately raises FileNotFoundError for missing data. It never
    generates random fallback data, because that would make reproducibility checks
    misleading.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}. Put HYDRUS or soil-column observations under data/raw/ "
            "or data/processed/ and update the YAML config. Random fallback data are intentionally disabled."
        )

    suffix = path.suffix.lower()
    if suffix in [".xlsx", ".xls"]:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif suffix in [".csv", ".txt"]:
        df = pd.read_csv(path)
    else:
        raise ValueError(f"Unsupported data format: {suffix}. Use .xlsx, .xls, .csv or .txt.")

    df.columns = df.columns.astype(str).str.strip()
    return df


def prepare_observations(df: pd.DataFrame, config: TrainingConfig) -> pd.DataFrame:
    """Validate, numeric-convert and coordinate-convert observations."""
    required = [config.col_x, config.col_t, config.col_theta]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    out = df.copy()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    out = out.dropna(subset=required).reset_index(drop=True)

    if config.depth_positive_downward:
        # HYDRUS commonly exports depths as positive downward. Convert to the
        # manuscript/PDE convention x=0 at surface and x<0 downward.
        out[config.col_x] = -np.abs(out[config.col_x].to_numpy(dtype=float))

    return out


def make_train_eval_split(df: pd.DataFrame, config: TrainingConfig) -> Tuple[pd.DataFrame, pd.DataFrame, str]:
    """Return train_df, eval_df and metric label.

    method='none' returns the same data for both training and evaluation. Metrics
    should then be described as reconstruction/fitting metrics.
    """
    val: ValidationConfig = config.validation
    if val.method == "none":
        return df.copy(), df.copy(), "reconstruction"

    if val.method == "time_fraction":
        if not (0.0 < val.fraction < 1.0):
            raise ValueError("validation.fraction must be between 0 and 1.")
        cutoff = df[config.col_t].quantile(1.0 - val.fraction)
        train_df = df[df[config.col_t] <= cutoff].copy()
        eval_df = df[df[config.col_t] > cutoff].copy()
        return train_df, eval_df, "held_out_time"

    if val.method == "depths":
        if not val.depths_cm:
            raise ValueError("validation.depths_cm must be non-empty when method='depths'.")
        depths = np.array(val.depths_cm, dtype=float)
        if config.depth_positive_downward:
            depths = -np.abs(depths)
        mask = np.isclose(df[config.col_x].to_numpy()[:, None], depths[None, :], atol=1.0e-6).any(axis=1)
        train_df = df[~mask].copy()
        eval_df = df[mask].copy()
        return train_df, eval_df, "held_out_depths"

    raise ValueError(f"Unsupported validation method: {val.method}")


def dataframe_to_tensors(
    df: pd.DataFrame,
    col_x: str,
    col_t: str,
    col_theta: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x = torch.tensor(df[col_x].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    t = torch.tensor(df[col_t].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    theta = torch.tensor(df[col_theta].to_numpy(), dtype=torch.float32, device=device).view(-1, 1)
    return x, t, theta
