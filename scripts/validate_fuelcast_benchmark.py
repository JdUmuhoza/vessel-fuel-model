"""Data-driven fuel-prediction benchmark on FuelCast (measured flowmeter fuel).

FuelCast vessels are cruise / offshore types whose fuel is dominated by hotel and
dynamic-positioning loads, so the calm-water hull-resistance (physics) model does
not apply. This script therefore validates DATA-DRIVEN fuel prediction against the
real KROHNE-measured total fuel, per vessel, with a temporal holdout.

Target : Consumer_Total_MomentaryFuel (kg/s), measured.
Models : baseline (a * speed^3)  vs  pure ML (HistGradientBoosting).

Run:  python scripts/validate_fuelcast_benchmark.py
"""

from __future__ import annotations

import glob
import pathlib

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

FC = pathlib.Path("data/external/fuelcast")
SEED = 42


def build(df: pd.DataFrame) -> pd.DataFrame:
    def col(c):
        return df[c].to_numpy() if c in df.columns else np.full(len(df), np.nan)
    stw = col("Ship_SpeedThroughWater")
    sog = col("Ship_SpeedOverGround")
    # use STW unless missing/degenerate, else SOG
    speed = stw if (np.isfinite(stw).any() and np.nanstd(stw) > 0.1) else sog
    out = pd.DataFrame({
        "idx": col("index"),
        "speed": speed,
        "draft": 0.5 * (col("Ship_DraftAft") + col("Ship_DraftFore")),
        "wind": col("Ship_AnemometerWindSpeed"),
        "hs": col("Weather_WaveHeight"),
        "tp": col("Weather_WavePeriod"),
        "cur": col("Weather_OceanCurrentVelocity"),
        "depth": col("Environment_SeaFloorDepth"),
        "fuel": col("Consumer_Total_MomentaryFuel"),  # kg/s measured
    })
    return out[(out["speed"] > 0.5) & (out["fuel"] > 0)].dropna(
        axis=1, how="all").dropna(subset=["speed", "fuel", "idx"]).reset_index(drop=True)


def metrics(y, p):
    e = p - y
    return {"MAE": float(np.mean(np.abs(e))), "RMSE": float(np.sqrt(np.mean(e**2))),
            "MAPE": float(np.mean(np.abs(e / y)) * 100),
            "R2": float(1 - np.sum(e**2) / np.sum((y - y.mean())**2))}


def main() -> None:
    print(f"{'vessel':14s} {'model':9s} | {'MAE(kg/s)':>9s} {'RMSE':>7s} {'MAPE%':>6s} {'R2':>7s}")
    for p in sorted(glob.glob(str(FC / "*.parquet"))):
        name = pathlib.Path(p).stem
        d = build(pd.read_parquet(p)).sort_values("idx").reset_index(drop=True)
        feats = [c for c in ["speed", "draft", "wind", "hs", "tp", "cur", "depth"] if c in d.columns]
        n_tr = int(0.70 * len(d))
        tr, te = d.iloc[:n_tr], d.iloc[n_tr:]
        ytr, yte = tr["fuel"].to_numpy(), te["fuel"].to_numpy()
        # baseline: a * speed^3
        a = float(np.sum(ytr * tr.speed.to_numpy()**3) / np.sum(tr.speed.to_numpy()**6))
        mb = metrics(yte, a * te.speed.to_numpy()**3)
        # pure ML
        ml = HistGradientBoostingRegressor(random_state=SEED, max_iter=400).fit(tr[feats].to_numpy(), ytr)
        mm = metrics(yte, ml.predict(te[feats].to_numpy()))
        print(f"{name:14s} {'baseline':9s} | {mb['MAE']:9.4f} {mb['RMSE']:7.4f} {mb['MAPE']:6.1f} {mb['R2']:7.3f}")
        print(f"{name:14s} {'pure_ml':9s} | {mm['MAE']:9.4f} {mm['RMSE']:7.4f} {mm['MAPE']:6.1f} {mm['R2']:7.3f}   (n={len(d)}, feats={len(feats)})")


if __name__ == "__main__":
    main()
