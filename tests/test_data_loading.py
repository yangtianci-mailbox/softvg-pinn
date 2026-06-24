from pathlib import Path

from softvg_pinn.config import load_config
from softvg_pinn.data import prepare_observations, read_observation_table


def test_example_data_loads():
    cfg = load_config("configs/debug_smoke_test.yaml")
    df = read_observation_table(cfg.data_path, cfg.sheet_name)
    out = prepare_observations(df, cfg)
    assert len(out) > 0
    assert cfg.col_x in out.columns
