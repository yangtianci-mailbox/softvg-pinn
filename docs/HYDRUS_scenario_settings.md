# HYDRUS-1D scenario settings

These settings were transcribed from the project HYDRUS notes and encoded in `configs/scenarios/*.yaml`.

## Common settings

| Item | Setting |
| --- | --- |
| Software | HYDRUS-1D |
| Process | Water Flow |
| Governing equation | Richards equation |
| Soil hydraulic model | van Genuchten--Mualem |
| Hysteresis | Not considered |
| Root water uptake | Not considered |
| Solute transport | Not considered |
| Units | cm, day |
| Profile depth | 0--400 cm |
| Mesh | 1 cm per node |
| Layer interfaces | 50 and 150 cm, unless varied by L scenarios |
| Upper boundary | Atmospheric BC with Surface Layer |
| Lower boundary | Free Drainage |
| Initial condition type | In Pressure Heads |
| Evaporation | 0 |
| Transpiration | 0 |
| Output depths | 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 260, 380 cm |

## Reference VG--Mualem parameters

| Soil | theta_r | theta_s | alpha (cm^-1) | n | Ks (cm day^-1) | l |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Silt | 0.034 | 0.460 | 0.016 | 1.37 | 6.0 | 0.5 |
| Sand | 0.080 | 0.430 | 0.145 | 2.68 | 712.8 | 0.5 |
| Sandy Loam | 0.065 | 0.410 | 0.075 | 1.89 | 106.1 | 0.5 |
| Loam | 0.078 | 0.430 | 0.036 | 1.56 | 24.96 | 0.5 |
| Loamy Sand | 0.057 | 0.410 | 0.124 | 2.28 | 350.2 | 0.5 |
| Clay Loam | 0.095 | 0.410 | 0.019 | 1.31 | 6.24 | 0.5 |
| Silt Loam | 0.060 | 0.450 | 0.020 | 1.41 | 10.8 | 0.5 |
| Sandy Clay Loam | 0.100 | 0.390 | 0.059 | 1.48 | 31.44 | 0.5 |

## A group: infiltration intensity and timing

- A1: baseline infiltration, 1.0 cm day^-1 during days 1--2, 21--22, 41--42, 61--62, and 81--82.
- A2: weak infiltration, same timing as A1 with 0.5 cm day^-1.
- A3: strong infiltration, same timing as A1 with 2.0 cm day^-1.
- A4: very strong infiltration, same timing as A1 with 4.0 cm day^-1.
- A5: single strong rainfall, 10.0 cm day^-1 during days 20--21.
- A6: multiple strong pulses, 5.0 cm day^-1 during days 10--11, 30--31, 50--51, 70--71, and 90--91.

## B group: initial pressure head

- B1: uniform -100 cm.
- B2: uniform -300 cm.
- B3: uniform -1000 cm.
- B4: layered initial condition: 0--50 cm Silt -100 cm; 50--150 cm Sand -500 cm; 150--400 cm Sandy Loam -800 cm.

## C group: simulation duration

The profile, boundary conditions, soil layers and initial pressure head follow A1. Rainfall begins on day 1, occurs once every 20 days, lasts 2 days each time, and has intensity 1.0 cm day^-1.

- C1: 30 days.
- C2: 100 days.
- C3: 200 days.

## L group: number of layers

- L1: 0--400 cm Silt.
- L2: 0--50 cm Silt; 50--400 cm Sand.
- L3: 0--50 cm Silt; 50--150 cm Sand; 150--400 cm Sandy Loam.
- L4: 0--50 cm Silt; 50--100 cm Loam; 100--200 cm Sand; 200--400 cm Sandy Loam.
- L5: 0--40 cm Silt; 40--100 cm Loam; 100--160 cm Sand; 160--260 cm Sandy Loam; 260--400 cm Clay Loam.

## T group: soil-type combinations

All T scenarios use three layers: 0--50, 50--150 and 150--400 cm.

- T1: Silt / Sand / Sandy Loam.
- T2: Loam / Sand / Sandy Loam.
- T3: Clay Loam / Sand / Sandy Loam.
- T4: Silt / Loamy Sand / Sandy Loam.
- T5: Silt / Sand / Loam.
- T6: Sand / Silt / Sandy Loam.
- T7: Silt / Sandy Loam / Sand.
- T8: Loam / Silt Loam / Sandy Clay Loam.

## Noise and sparse observations

- Noise datasets are generated from A1 with 0%, 1%, 3%, 5% and 10% noise.
- Sparse observation datasets retain:
  - P12: 10, 20, 30, 40, 60, 80, 100, 120, 140, 160, 260, 380 cm.
  - P8: 10, 30, 60, 100, 140, 160, 260, 380 cm.
  - P6: 10, 40, 80, 140, 260, 380 cm.
  - P4: 10, 40, 100, 260 cm.
  - P3: 20, 100, 260 cm.
