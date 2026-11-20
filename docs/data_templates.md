# Strict Data Template Pack

This document defines strict input schemas, units, accepted ranges, and validation checks.

## 1) AIS Segments Template

Template file: [templates/data/ais_segments_template.csv](templates/data/ais_segments_template.csv)

Required columns:
- `segment_id` (string, unique)
- `vessel_id` (string)
- `distance_km` (float, > 0)
- `speed_tw_kn` (float, 0 < value <= 35)
- `route_id` (string)
- `season` (`winter|spring|summer|autumn`)

Recommended columns:
- `timestamp` (ISO-8601 UTC)
- `lat` (deg, -90..90)
- `lon` (deg, -180..180)

## 2) Meteocean by Segment Template

Template file: [templates/data/metocean_by_segment_template.csv](templates/data/metocean_by_segment_template.csv)

Required columns:
- `segment_id`
- `wind_kn` (kn, >= 0)
- `wind_angle_deg` (deg, 0..360)
- `Hs` (m, >= 0)
- `Tp` (s, >= 0)
- `current_kn` (kn, can be negative if sign convention uses signed speed)
- `current_angle_deg` (deg, 0..360)
- `sst_c` (°C, -2..40)
- `depth_m` (m, > 0)
- `wave_angle_deg` (deg, 0..360)

## 3) Vessel Particulars Template

Template file: [templates/data/vessel_particulars_template.csv](templates/data/vessel_particulars_template.csv)

Required columns:
- `vessel_id`
- `L` (m, > 10)
- `B` (m, > 2)
- `T` (m, > 0.5)
- `S` (m², > 0)
- `CB` (0..1)
- `MCR` (kW, > 100)
- `sfoc_at_mcr` (g/kWh, 130..260)

Recommended columns:
- `Cp`, `Cm`, `Cwp`
- `A_front`, `A_lateral`, `Cd_air`
- `eta_0`, `eta_wave_loss`
- `design_draft`

## 4) Noon / Engine Template

Template file: [templates/data/noon_engine_template.csv](templates/data/noon_engine_template.csv)

Required columns:
- `segment_id`
- `fuel_mt` (metric ton, >= 0)

Recommended columns:
- `trim_m`
- `months_since_cleaning`
- `include_aux`, `aux_power_kw`, `aux_sfoc`
- `include_boiler`, `boiler_power_kw`, `boiler_sfoc`

Meaning of "noon" in this repository:
- `noon` means operational engine/fuel reports aligned to AIS segments.
- If you have true onboard logs, use them directly.
- If you do not have logs, the CLI can generate deterministic proxy noon rows for reproducible experiments.
- The most important field is `fuel_mt` (observed segment fuel), which is the supervised target.

## Validation Checklist

- [ ] All required columns exist exactly as named.
- [ ] `segment_id` is unique in AIS and present in metocean and noon files.
- [ ] `vessel_id` in AIS exists in vessel particulars.
- [ ] No negative distances or impossible speeds.
- [ ] Angles are normalized to [0, 360].
- [ ] Timestamps are UTC ISO format.
- [ ] Units are exactly as defined above.
- [ ] At least 500+ segments recommended for robust benchmarking.

## Comparator Inputs

Optional comparator sources:
- Frozen baselines from historical outputs (`resultsV5_*`, `resultsV7_*`, `resultsV8_*`).
- Vendor/public tool CSV with `segment_id` + fuel prediction.

Use CLI:
- `--frozen-baselines-root <path>`
- `--frozen-baseline-tags V5,V7,V8`
- `--vendor-baseline-csv <path/to/vendor.csv>`

See also:
- [docs/feature_reference.md](docs/feature_reference.md) for full feature meanings and sensitivity interpretation.
