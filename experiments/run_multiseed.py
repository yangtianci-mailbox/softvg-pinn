"""Run the same config with multiple random seeds.

Example:
    python experiments/run_multiseed.py --config configs/hydrus_three_layer.yaml --seeds 0 1 2 3 4 5 6 7 8 9
"""
from __future__ import annotations

from pathlib import Path
import json
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from softvg_pinn.config import load_config
from softvg_pinn.train import train_model


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    rows = []
    for seed in args.seeds:
        cfg = load_config(args.config)
        cfg.seed = seed
        cfg.experiment_name = f"{cfg.experiment_name}/seed_{seed}"
        if args.device is not None:
            cfg.device = args.device
        result = train_model(cfg)
        metrics = result["metrics"]
        row = {
            "seed": seed,
            "metric_context": metrics["metric_context"],
            "train_rmse": metrics["train_theta"]["rmse"],
            "train_mae": metrics["train_theta"]["mae"],
            "train_r2": metrics["train_theta"]["r2"],
            "eval_rmse": metrics["eval_theta"]["rmse"],
            "eval_mae": metrics["eval_theta"]["mae"],
            "eval_r2": metrics["eval_theta"]["r2"],
        }
        rows.append(row)

    summary = pd.DataFrame(rows)
    base_cfg = load_config(args.config)
    out_dir = Path(base_cfg.output_dir) / base_cfg.experiment_name / "multiseed_summary"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_dir / "multiseed_metrics.csv", index=False)
    summary.describe().to_csv(out_dir / "multiseed_metrics_describe.csv")
    print(summary)


if __name__ == "__main__":
    main()
