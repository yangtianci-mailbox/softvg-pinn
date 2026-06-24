"""Run all YAML files in configs/scenarios.

Example:
    python experiments/run_scenarios.py --pattern "configs/scenarios/A*.yaml"
"""
from __future__ import annotations

from pathlib import Path
import glob
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from softvg_pinn.config import load_config
from softvg_pinn.train import train_model


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--pattern", default="configs/scenarios/*.yaml")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    paths = sorted(glob.glob(args.pattern))
    if not paths:
        raise FileNotFoundError(f"No config files matched pattern: {args.pattern}")

    for path in paths:
        cfg = load_config(path)
        if args.device is not None:
            cfg.device = args.device
        train_model(cfg)


if __name__ == "__main__":
    main()
