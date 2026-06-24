# SoftVG-PINN reproducibility package

This repository contains a GitHub-ready refactor of the original SoftVG-PINN notebook for estimating effective van Genuchten--Mualem hydraulic functions in layered soils from water-content observations.

The code is organized for manuscript review and reproducibility. It removes local Windows paths, disables random fallback data generation, supports YAML experiment configs, fixes random seeds, writes all outputs to `results/`, and separates source code from experiment scripts.

## Repository contents

```text
softvg_pinn/              Python package: model, losses, data loading, metrics and plotting
configs/                  YAML configs for HYDRUS scenarios and field columns
experiments/              Command-line scripts for single, multi-seed, ablation and scenario runs
data/example/             Tiny example dataset for a smoke test
data/raw/                 Put original HYDRUS/soil-column files here; not tracked by default
data/processed/           Put shareable processed datasets here
docs/                     HYDRUS scenario settings and data-dictionary notes
tests/                    Unit tests and smoke tests
results/                  Generated outputs; ignored by Git
figures/                  Generated/collected figures; ignored by Git except README
```

## Coordinate convention

The PINN uses the manuscript coordinate system:

- `x = 0 cm` at the soil surface.
- `x < 0 cm` downward.
- A 0--50 cm layer is represented as `top: 0`, `bottom: -50`.

HYDRUS often exports depth as positive downward. Set this in a config file:

```yaml
depth_positive_downward: true
```

The loader will convert HYDRUS depth `z` to `x = -z`.

## Data format

Input files may be `.xlsx`, `.xls`, `.csv` or `.txt`. Default column names are:

| Column | Meaning |
| --- | --- |
| `Depth [L]` | depth coordinate in cm |
| `Time [Day]` | elapsed time in days |
| `Moisture [-]` | volumetric water content |

The code intentionally raises `FileNotFoundError` if a file is missing. It does **not** generate random fallback data.

## Installation

```bash
conda env create -f environment.yml
conda activate softvg-pinn
```

or:

```bash
python -m venv .venv
source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick smoke test

This verifies that the package imports, trains for a few iterations and writes outputs:

```bash
python experiments/run_single.py --config configs/debug_smoke_test.yaml --device cpu
```

Expected outputs:

```text
results/debug_smoke_test/
  config_used.yaml
  metadata.json
  model.pt
  figures/
  tables/
```

## Run the HYDRUS baseline

Place the baseline HYDRUS output file at:

```text
data/raw/hydrus/A1.xlsx
```

Then run:

```bash
python experiments/run_single.py --config configs/hydrus_three_layer.yaml
```

## Run multi-seed experiments

```bash
python experiments/run_multiseed.py \
  --config configs/hydrus_three_layer.yaml \
  --seeds 0 1 2 3 4 5 6 7 8 9
```

## Run ablation experiments

```bash
python experiments/run_ablation.py --config configs/hydrus_three_layer.yaml
```

## Run all scenario configs

```bash
python experiments/run_scenarios.py --pattern "configs/scenarios/*.yaml"
```

## Output files

Each run creates:

```text
results/<experiment_name>/
  config_used.yaml
  metadata.json
  model.pt
  tables/
    predictions.csv
    learned_vg_parameters.csv
    training_log.csv
    metrics.json
  figures/
    vg_curves_<N>_layers.png
    water_content_reconstruction.png
```

## Important interpretation notes

1. If `validation.method: none`, all observations are used for training. The theta metrics in `metrics.json` are reconstruction/fitting metrics, not independent prediction metrics.
2. The VG loss uses natural logarithms for conductivity. Reported hydraulic-function metrics use `log10 K(h)`. The different log bases only rescale squared conductivity errors by a constant.
3. L-BFGS is implemented as final refinement of the composite objective; it does not prove uniqueness.
4. For soil-column data without independent pressure-head or conductivity measurements, estimated VG-Mualem parameters should be interpreted as effective parameters.

## Tests

```bash
pip install -r requirements.txt
pytest
```

## Data release recommendation

For journal submission, archive this repository with Zenodo and add the generated DOI to the manuscript. Put small processed datasets in `data/processed/`. For raw files that cannot be redistributed, provide a clear `data/raw/README.md` explaining access restrictions and preprocessing steps.

## Citation

See `CITATION.cff`. Update author names, title and DOI before public release.
