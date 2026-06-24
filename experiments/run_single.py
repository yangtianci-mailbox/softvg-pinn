"""Run one SoftVG-PINN experiment from a YAML config.

Example:
    python experiments/run_single.py --config configs/hydrus_three_layer.yaml
"""
from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from softvg_pinn.config import load_config
from softvg_pinn.train import train_model


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--seed", type=int, default=None, help="Optional seed override.")
    parser.add_argument("--device", default=None, help="Optional device override: auto, cpu, cuda.")
    args = parser.parse_args()

    config = load_config(args.config)
    if args.seed is not None:
        config.seed = args.seed
    if args.device is not None:
        config.device = args.device
    train_model(config)


if __name__ == "__main__":
    main()
