# Data directory

## Expected input columns

Default column names:

- `Depth [L]`: depth in cm.
- `Time [Day]`: time in days.
- `Moisture [-]`: volumetric water content.

If your files use different names, edit the YAML config:

```yaml
col_x: Depth [L]
col_t: Time [Day]
col_theta: Moisture [-]
```

## Coordinate conversion

HYDRUS commonly exports depths as positive downward. The model uses `x = 0` at the surface and `x < 0` downward. Use:

```yaml
depth_positive_downward: true
```

## Raw vs processed data

- Place raw HYDRUS files under `data/raw/hydrus/`.
- Place raw field-column files under `data/raw/field/`.
- Place shareable processed files under `data/processed/`.

Large raw files and restricted data are ignored by Git by default. If raw field data cannot be public, explain access restrictions and provide processed/anonymized datasets when possible.
