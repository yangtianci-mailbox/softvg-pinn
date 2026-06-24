# GitHub upload checklist

Before pushing this repository:

- Replace author placeholders in `LICENSE`, `CITATION.cff`, `pyproject.toml` and `README.md`.
- Put public processed data in `data/processed/` or explain restricted data in `data/raw/README.md`.
- Do not commit large checkpoints, private raw Excel files or local path information.
- Run `pytest`.
- Run `python experiments/run_single.py --config configs/debug_smoke_test.yaml --device cpu`.
- Archive the release on Zenodo and add the DOI to the manuscript data availability statement.
