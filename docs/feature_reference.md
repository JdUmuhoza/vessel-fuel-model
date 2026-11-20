# Feature Reference (All Model Inputs)

This page explains every feature used by the benchmark models and the meaning of `noon` reports.

## 1) What is a `noon` report?

In shipping operations, a **noon report** is an operational report logged by the vessel (traditionally daily at noon UTC, but often at other intervals in modern systems).

In this repository, `noon` rows are per-segment operational measurements joined to AIS segments by `segment_id` (or nearest timestamp during preprocessing). They provide:

- `fuel_mt` (fit/evaluation target): fuel consumed on the segment
- trim/fouling status
- auxiliary and boiler operating loads

**Provenance of `fuel_mt` matters.** If you supply real engine-log noon reports,
it is a measurement. A value derived from a model, or scaled to an annual
regulatory total, is not: agreement with it shows calibration consistency, not
predictive skill. Prediction-skill claims in this project are made only on
instrument-measured benchmarks (Shifts, FuelCast).

If true noon/engine logs are not available, the CLI can generate **proxy noon rows** from the physics model for reproducibility (`_derive_proxy_csvs` path).

## 2) Raw inputs expected from users

### AIS segment fields
- `distance_km`: segment distance in km
- `speed_tw_kn`: through-water speed in knots
- `route_id`, `season`, `timestamp`: split/analysis context

### Meteocean fields (`env`)
- `wind_kn`, `wind_angle_deg`
- `Hs`, `Tp`, `wave_angle_deg`
- `current_kn`, `current_angle_deg`
- `sst_c`, `depth_m`

### Vessel/engine fields (`vessel_params`)
- Geometry: `L`, `B`, `T`, `S`, `CB`, `Cp`, `Cm`, `Cwp`
- Aerodynamics: `A_front`, `A_lateral`, `Cd_air`
- Propulsion/engine: `MCR`, `sfoc_at_mcr`, `eta_0`, `eta_wave_loss`
- Ops state: `trim_m`, `design_draft`, `months_since_cleaning`
- Aux/boiler: `include_aux`, `aux_power_kw`, `aux_sfoc`, `include_boiler`, `boiler_power_kw`, `boiler_sfoc`

## 3) Engineered features used by ML models (45)

| Feature | Unit | Source | Meaning |
|---|---:|---|---|
| `distance_km` | km | AIS | Segment length; scales transit energy |
| `speed_kn` | kn | AIS | Through-water speed |
| `speed2` | kn² | Derived | Nonlinear speed effect |
| `speed3` | kn³ | Derived | Cubic drag tendency |
| `Hs` | m | Meteocean | Significant wave height |
| `Tp` | s | Meteocean | Wave peak period |
| `wind_kn` | kn | Meteocean | Wind speed magnitude |
| `cos_wind` | - | Derived | Wind direction cosine component |
| `sin_wind` | - | Derived | Wind direction sine component |
| `current_kn` | kn | Meteocean | Current speed magnitude |
| `cos_current` | - | Derived | Current direction cosine component |
| `sin_current` | - | Derived | Current direction sine component |
| `depth_m` | m | Meteocean | Water depth; shallow-water effects |
| `sst_c` | °C | Meteocean | Sea-surface temperature |
| `L` | m | Vessel | Length |
| `B` | m | Vessel | Breadth |
| `T` | m | Vessel | Draft |
| `CB` | - | Vessel | Block coefficient |
| `S` | m² | Vessel | Wetted surface area |
| `Cp` | - | Vessel | Prismatic coefficient |
| `Cm` | - | Vessel | Midship coefficient |
| `Cwp` | - | Vessel | Waterplane coefficient |
| `months_since_cleaning` | months | Noon/Vessel | Fouling age proxy |
| `trim_m` | m | Noon/Vessel | Trim condition |
| `design_draft` | m | Vessel | Design draft reference |
| `A_front` | m² | Vessel | Frontal projected area |
| `A_lateral` | m² | Vessel | Lateral projected area |
| `Cd_air` | - | Vessel | Aerodynamic drag coefficient |
| `MCR` | kW | Vessel | Main engine max continuous rating |
| `sfoc_at_mcr` | g/kWh | Vessel | Specific fuel oil consumption at MCR |
| `eta_0` | - | Vessel | Base propulsion efficiency |
| `eta_wave_loss` | - | Vessel | Sea-state efficiency degradation factor |
| `include_aux` | 0/1 | Noon/Vessel | Auxiliary engine on/off |
| `aux_power_kw` | kW | Noon/Vessel | Auxiliary load |
| `include_boiler` | 0/1 | Noon/Vessel | Boiler on/off |
| `boiler_power_kw` | kW | Noon/Vessel | Boiler load |
| `speed_x_Hs` | kn·m | Derived | Speed-wave interaction |
| `speed_x_wind` | kn² | Derived | Speed-wind interaction |
| `speed_x_current` | kn² | Derived | Speed-current interaction |
| `distance_x_Hs` | km·m | Derived | Route exposure to waves |
| `distance_x_wind` | km·kn | Derived | Route exposure to wind |
| `trim_x_draft_offset` | m² | Derived | Trim relative to loaded condition |
| `vessel_class_code` | - | Derived (categorical) | Vessel-class integer code (`fit_category_maps`) |
| `route_code` | - | Derived (categorical) | Route integer code (`fit_category_maps`) |
| `season_code` | - | Derived (categorical) | Season integer code (`fit_category_maps`) |

## 4) Why meteo sensitivity can look near zero

If `sensitivity_ranking.csv` shows low meteo impact, common causes are:

1. **Low test-set variation** in weather variables (check `feature_profile_test_split.csv`).
2. You are viewing **residual-only sensitivity** instead of full model sensitivity.
3. Fuel target is dominated by speed/distance range for that split.
4. Segment-level weather is nearly constant after matching/aggregation.

This project now writes two files:
- `sensitivity_ranking.csv` → total hybrid sensitivity (physics + residual)
- `sensitivity_residual_ranking.csv` → residual-only ML sensitivity

Use both before concluding weather has no effect.

## 5) Model naming (to avoid confusion)

In `results_table.csv`:
- `hybrid` / `hybrid_ridge` is the **main proposed model** (grey-box residual learning; `hybrid` is the stable alias, `is_primary_model = 1`)
- `physics_only` is deterministic calibrated physics
- `pure_ml` / `pure_ml_ridge` is the data-only ridge model
- `speed_power`, `cubic_law`, `physics_reduced` are reference baselines
- Additional backends appear when their library is installed: `ols`,
  `ridge_sklearn`, `bayesian_ridge`, `knn`, `svr`, `quantile`, `gpr`, `mlp`,
  `pnn`, tree ensembles (`rf`, `gbt`, `xgboost`, `catboost`, `lightgbm`), and
  sequence/symbolic models (`lstm`, `prophet`, `symbolic`). A missing library
  skips that backend gracefully — the row is simply absent.

The file also includes:
- `model_display` (human-readable name)
- `is_primary_model` (`1` for `hybrid`, otherwise `0`)
