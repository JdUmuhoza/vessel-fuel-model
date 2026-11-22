# vessel-fuel-model

Python models of vessel fuel consumption that combine a physics core with machine
learning, and propagate the predicted fuel to CO2, CH4, N2O and CO2e.

The package provides:

- a white-box physics model (Holtrop-Mennen calm water, Blendermann wind, STAWAVE-1 added
  wave resistance, Schultz hull fouling, a load-dependent SFOC curve) with a single
  calibration factor;
- black-box learners, a serial grey-box model (physics plus a learned residual) and a
  constrained grey-box model in which a validation-selected coefficient
  `0 <= lambda <= 1` scales the learned residual;
- a multi-gas emission head (IMO Fourth GHG Study emission factors, IPCC AR6 GWP100) and a
  CII-based sensitivity calculation;
- the experiment code that produces the tables and figures of the accompanying paper on
  the Shifts and FuelCast benchmarks.

## Install

Python 3.10 or later.

```bash
pip install -e .            # physics core only (numpy, scipy)
pip install -e ".[ml]"      # learners and experiment code (scikit-learn, boosting libraries, pandas)
pip install -e ".[dev,ml]"  # everything, including the test suite
pytest -q
```

## Quick start

```python
import numpy as np
from vessel_fuel import fuel_model

params = {"L": 120.0, "B": 20.0, "T": 7.0, "design_draft": 6.5, "S": 4200.0, "CB": 0.72,
          "MCR": 10000.0, "sfoc_at_mcr": 180.0, "eta_0": 0.72, "eta_wave_loss": 0.03,
          "A_front": 120.0, "A_lateral": 700.0, "months_since_cleaning": 10.0, "trim_m": 0.3}
env = {"wind_kn": 12.0, "wind_angle_deg": 20.0, "Hs": 1.5, "Tp": 8.0,
       "wave_angle_deg": 0.0, "sst_c": 15.0, "depth_m": 50.0}

fuel_t = fuel_model(100.0, np.array([8.0, 10.0, 12.0]), env, params)   # 100 km at three speeds
```

Every numeric function accepts a scalar or a NumPy array and returns the same type.

## Using your own data

Input files follow the CSV templates in `templates/data/`. Their columns and units are
described in `docs/data_templates.md` and `docs/feature_reference.md`.

## Reproducing the paper

The paper evaluates the models on two public benchmarks with measured targets. Neither is
redistributed here; both are cited and downloaded from their official sources:

| Benchmark | Source | Licence |
|---|---|---|
| Shifts vessel power (Malinin et al., 2022) | Zenodo record 7684813 | CC BY-NC-SA 4.0 |
| FuelCast (Viga et al., 2026) | Hugging Face `krohnedigital/FuelCast` | CC BY-NC-ND 4.0 |

```bash
python scripts/download_benchmarks.py      # fetches both and verifies every file's SHA-256
```

The 13 experiment stages that write the paper's tables are listed in
`reproduce/stages.json`, and `reproduce/table_map.json` links every checked table cell to
the result file that produced it.

```bash
python -m reproduce.reproduce --quick          # compare shipped results with the paper (minutes)
python -m reproduce.reproduce --full --yes     # rerun every stage from the downloaded data, then compare
```

`--full` copies the existing `results/` to `results_shipped/` before it overwrites anything.
It then deletes the cached learner fits (every `_cache/` folder and `_rows.csv` under
`results/`) and the cached physics predictions, so that every learner is refitted rather than
read back. This takes several hours: the SVR residual learner alone took 4,674 s to fit on
the 286,580-row fitting block in the original run.
`reproduce/MISMATCHES.md` lists the table cells that differ from the paper and why.

A frozen reproduction package with the result files, an exact environment and a quick check
that runs in minutes is archived on OSF (DOI to be added on registration).

## Provenance

This research code was developed from May 2026 in a private repository. This repository is
its cleaned public release. The earlier history is not published because it contains
third-party data with restricted licences.

## Citation

See `CITATION.cff`. The accompanying paper: J. d'A. Umuhoza, M. J.-D. Otis, O. Jebbar,
*Constrained Grey-Box Residual Learning for Vessel Fuel-Consumption Prediction under
Distribution Shift: Multi-Gas, Uncertainty, and Decision Assessment* (manuscript).

## Licence

Code: MIT (see `LICENSE`). The benchmark datasets keep their own licences, listed above.
