"""Run ablation experiments from one base config."""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from softvg_pinn.config import load_config
from softvg_pinn.train import train_model

ABLATIONS = {
    "M1_data_only": {"vg": 0.0, "pde": 0.0, "prior": 0.0, "use_lbfgs": False},
    "M2_data_vg": {"vg": 10.0, "pde": 0.0, "prior": 0.0, "use_lbfgs": False},
    "M3_data_pde": {"vg": 0.0, "pde": 0.01, "prior": 0.0, "use_lbfgs": False},
    "M4_data_vg_pde": {"vg": 10.0, "pde": 0.01, "prior": 0.0, "use_lbfgs": False},
    "M5_full": {"vg": 10.0, "pde": 0.01, "prior": 1.0e-3, "use_lbfgs": True},
    "M6_no_soft_mask": {"vg": 10.0, "pde": 0.01, "prior": 1.0e-3, "use_lbfgs": True},
    "M7_no_lbfgs": {"vg": 10.0, "pde": 0.01, "prior": 1.0e-3, "use_lbfgs": False},
}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    for name, opts in ABLATIONS.items():
        cfg = load_config(args.config)
        cfg.experiment_name = f"ablation/{name}"
        if args.device is not None:
            cfg.device = args.device
        cfg.loss_weights.vg = opts["vg"]
        cfg.loss_weights.pde = opts["pde"]
        cfg.loss_weights.prior = opts["prior"]
        cfg.use_lbfgs = opts["use_lbfgs"]

        if name == "M6_no_soft_mask":
            # Approximation of a hard interface: very steep transition and no
            # masked interface buffer. This is not numerically identical to a
            # discontinuous parameter jump but tests the role of soft masks.
            cfg.interface_k_initial = 80.0
            cfg.interface_k_stage1_max = 80.0
            cfg.interface_k_stage2_max = 80.0
            cfg.interface_buffer_cm = 0.0

        train_model(cfg)


if __name__ == "__main__":
    main()
