# Reproduction package: constrained grey-box residual learning for vessel fuel consumption

This package reproduces the tables of the paper *Constrained Grey-Box Residual Learning for
Vessel Fuel-Consumption Prediction under Distribution Shift: Multi-Gas, Uncertainty, and
Decision Assessment* (J. d'A. Umuhoza, M. J.-D. Otis, O. Jebbar). It holds the code that
produced the results, the result files themselves, and a check that compares every mapped
table cell with the value printed in the paper. The maintained library is on GitHub
(address in `CITATION.cff`).

## Contents

| Path | What it is |
|---|---|
| `vessel_fuel/` | The model library (physics core, learners, grey-box models, emission head) |
| `experiments/` | The 13 experiment stages that write the paper's tables and figure data, and their imports |
| `results/revision_final/` | The result files those stages wrote for the paper |
| `reproduce/` | `paper_values.json` (the paper's tables), `table_map.json` (each checked cell and the result value that produced it), `stages.json` (stage order), `reproduce.py`, `requirements-lock.txt`, `MISMATCHES.md` |
| `scripts/download_benchmarks.py` | Downloads the two benchmarks from their official sources and verifies their checksums |

The benchmark data are not included. Shifts (Malinin et al., 2022; Zenodo record 7684813;
CC BY-NC-SA 4.0) and FuelCast (Viga et al., 2026; Hugging Face `krohnedigital/FuelCast`;
CC BY-NC-ND 4.0) are cited and downloaded from their sources for a full rerun only.

## Environment

Tested with Python 3.14.3 on Windows 11 (Intel Core Ultra 9 275HX, 24 logical CPUs, 15 GB
RAM). `reproduce/requirements-lock.txt` pins the exact library versions that produced the
results.

```bash
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install -r reproduce/requirements-lock.txt
pip install -e . --no-deps
```

## Quick check (seconds)

```bash
python -m reproduce.reproduce --quick
```

It compares 476 table cells of 13 tables with the shipped result files, at the precision
printed in the paper, and prints PASS or FAIL per table. Expected outcome: every table
passes except `tab:fuelcast-performance`, whose Poseidon and Triton values in the paper
were supplied by the authors and differ from what the code produces; `reproduce/MISMATCHES.md`
gives each differing cell, the printed value and the computed value. The command therefore
exits with code 1.

## Full rerun (optional)

```bash
python scripts/download_benchmarks.py         # about 110 MB, checksums verified
python -m reproduce.reproduce --full --yes    # reruns every stage, then the quick check
```

`--full` first copies `results/` to `results_shipped/`, then deletes the cached learner
fits (every `_cache/` folder and `_rows.csv` under `results/`) and the cached physics
predictions, so that every learner is refitted rather than read back, then rewrites
`results/` from the downloaded data. This takes several hours: the SVR residual learner
alone took 4,674 s to fit on the 286,580-row fitting block in the original run.
One stage, `s06_fuelcast_wind_and_wbm`, ends with exit code 2 by design:
its own check reports that the hull geometry of the reduced FuelCast prior sits at the edge
of its search grid for two vessels, the same verdict as in the original run.

## Archive

`vessel-fuel-model-osf-v1.0.zip` (218 files, 5.8 MB), SHA-256
`46fda4c8e7ddf67ad98322f6a27f6845a506e0164c40852bff2d5c36b8f09e46`.
