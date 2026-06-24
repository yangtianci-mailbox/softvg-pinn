"""Configuration helpers for SoftVG-PINN experiments."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
import os
import random

import numpy as np
import torch
import yaml


@dataclass
class SoilLayer:
    """One soil layer in the manuscript coordinate system.

    top and bottom are in cm. The package uses x = 0 cm at the soil surface and
    x < 0 cm downward, so a 0--50 cm layer is represented as top=0, bottom=-50.
    """
    top: float
    bottom: float
    soil: str

    def as_tuple(self) -> Tuple[float, float, str]:
        return (float(self.top), float(self.bottom), str(self.soil))


@dataclass
class LossWeights:
    data: float = 100.0
    vg: float = 10.0
    pde: float = 0.01
    prior: float = 1.0e-3


@dataclass
class ValidationConfig:
    """Optional hold-out settings.

    The manuscript uses reconstruction metrics unless this section is explicitly
    changed. method='none' means all observations are used for training and the
    reported theta metrics are fitting/reconstruction metrics, not out-of-sample
    prediction metrics.
    """
    method: Literal["none", "time_fraction", "depths"] = "none"
    fraction: float = 0.2
    depths_cm: List[float] = field(default_factory=list)


@dataclass
class TrainingConfig:
    experiment_name: str = "softvg_pinn"
    data_path: str = "data/raw/hydrus_three_layer.xlsx"
    sheet_name: Optional[str] = None
    output_dir: str = "results"
    seed: int = 2026
    device: str = "auto"

    col_x: str = "Depth [L]"
    col_t: str = "Time [Day]"
    col_theta: str = "Moisture [-]"
    depth_positive_downward: bool = False

    layers: List[SoilLayer] = field(default_factory=lambda: [
        SoilLayer(0.0, -50.0, "Silt"),
        SoilLayer(-50.0, -150.0, "Sand"),
        SoilLayer(-150.0, -400.0, "Sandy Loam"),
    ])

    net_layers: List[int] = field(default_factory=lambda: [2, 128, 128, 128, 128, 128, 3])
    n_collocation: int = 8000
    interface_k_initial: float = 5.0
    interface_k_stage1_max: float = 15.0
    interface_k_stage2_max: float = 20.0
    interface_buffer_cm: float = 1.5

    stage1_epochs: int = 10000
    stage2_epochs: int = 20000
    lbfgs_max_iter: int = 10000
    use_lbfgs: bool = True
    refresh_collocation_every: int = 2000
    print_every: int = 2000

    stage1_lr: float = 1.0e-3
    stage2_lr: float = 5.0e-4
    pde_weight_start: float = 1.0e-4
    pde_weight_end: float = 1.0e-2
    grad_clip: float = 1.0
    loss_weights: LossWeights = field(default_factory=LossWeights)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    # Free-form metadata saved with results; useful for HYDRUS settings.
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def layer_tuples(self) -> List[Tuple[float, float, str]]:
        return [layer.as_tuple() for layer in self.layers]


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Set Python, NumPy and PyTorch seeds."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Deterministic mode may slow training and is not guaranteed for every op.
    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = not bool(deterministic)


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _make_layers(raw_layers: Sequence[Dict[str, Any]]) -> List[SoilLayer]:
    layers = [SoilLayer(float(x["top"]), float(x["bottom"]), str(x["soil"])) for x in raw_layers]
    for layer in layers:
        if layer.top < layer.bottom:
            raise ValueError(
                f"Layer top must be above bottom in x-coordinate convention: {layer}. "
                "Use top=0, bottom=-50 for a 0--50 cm layer."
            )
    return layers


def load_config(path: str | Path) -> TrainingConfig:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    if "layers" in raw:
        raw["layers"] = _make_layers(raw["layers"])
    if "loss_weights" in raw and isinstance(raw["loss_weights"], dict):
        raw["loss_weights"] = LossWeights(**raw["loss_weights"])
    if "validation" in raw and isinstance(raw["validation"], dict):
        raw["validation"] = ValidationConfig(**raw["validation"])
    return TrainingConfig(**raw)


def save_config(config: TrainingConfig, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = asdict(config)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False, allow_unicode=True)
