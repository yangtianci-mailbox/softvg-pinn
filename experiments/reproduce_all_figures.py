"""Collect generated figures into the top-level figures/ directory.

Run training scripts first. This script does not retrain models.
"""
from __future__ import annotations

from pathlib import Path
import shutil


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    out = root / "figures"
    out.mkdir(exist_ok=True)
    for fig in (root / "results").glob("**/figures/*.png"):
        name = "_".join(fig.relative_to(root / "results").parts)
        shutil.copy2(fig, out / name)
    print(f"Copied figures to {out}")


if __name__ == "__main__":
    main()
