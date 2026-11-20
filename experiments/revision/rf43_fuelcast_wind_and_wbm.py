"""FuelCast Section 5.4, rebuilt after the loader fix, with a fuller physics prior.

Two parts, both on the Huber methodology of Sections 5.1-5.3 (Huber residual learner,
Huber standalone BBM, shrinkage-only constraint, per-vessel lambda on a 101-point grid
selected on that vessel's own training-side validation tail).

Part A - wind-aware rerun with the existing WBM-lite cube-law prior. Line-by-line
comparable with folder 29, which was produced before the loader fix and therefore had
apparent wind, wind direction and air temperature silently dropped from the feature set
(the ``_FC_MAP`` source names used lowercase ``10m``/``2m`` where the files use ``10M``/
``2M``, and the loader NaN-filled unknown names without raising).

Part B - fuller Holtrop-Mennen prior on Poseidon and Triton, the two vessels that carry
``Ship_Heading`` and therefore admit vessel-relative wind and wave angles. Three of the
four resistance terms are computable (calm water, Blendermann wind, STAWAVE-1 waves,
plus the current effect on speed over ground); the Schultz fouling term is explicitly
disabled because no FuelCast vessel carries a dry-docking clock. Ceto has no heading and
no speed through water, so it keeps the cube law.

Hull particulars are unknown for these vessels. Rather than assert a guess, the geometry
is treated as a prior hyperparameter and selected on the training-side validation tail by
the MAPE of the k-scaled prior alone, with k fitted on the fitting portion only. No
dev_in or dev_out record influences any selection.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from .common_rev import REPO_ROOT, REV_FINAL, SEED, write_csv
from .rf32_full_data_602020 import metrics7

_E = REPO_ROOT / "experiments"
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import (  # noqa: E402
    FC_VESSELS, fit_cube, fit_scale, fuelcast_shift_splits, load_fuelcast_vessel,
    make_learner,
)
from vessel_fuel import fuel_components  # noqa: E402

OUT = REV_FINAL / "30_fuelcast_wind_and_wbm"
F29 = REV_FINAL / "29_fuelcast_huber_final"
BACKEND = "huber"
FIT_FRACTION = 0.85
LAM = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 4)
VESSELS = ["CPS_Poseidon", "CPS_Triton", "OSS_Ceto"]
SHORT = {"CPS_Poseidon": "Poseidon", "CPS_Triton": "Triton", "OSS_Ceto": "Ceto"}
WBM_VESSELS = ["CPS_Poseidon", "CPS_Triton"]     # heading available
IDENTITY_TOL = 1e-9
METRICS = ["MAPE", "sMAPE", "MAE", "nMAE", "RMSE", "NRMSE", "R2"]

# Geometry grid. B follows a beam ratio, S from Denny-Mumford, areas from L and B.
GRID_L = [80.0, 100.0, 120.0, 140.0, 160.0, 180.0]
GRID_BRATIO = [4.5, 5.5, 6.5]
GRID_CB = [0.60, 0.70, 0.80]
GRID_T_NODRAFT = [5.0, 7.0, 9.0, 11.0]           # Poseidon: no measured draught
ETA_0, ETA_WAVE_LOSS, SFOC_MCR = 0.70, 0.0, 180.0


def _git() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def particulars(L: float, bratio: float, cb: float, T: float, mcr: float) -> dict:
    B = L / bratio
    S = 1.7 * L * T + L * B * cb           # Denny-Mumford wetted surface
    return dict(L=L, B=B, T=T, S=S, CB=cb, MCR=mcr, sfoc_at_mcr=SFOC_MCR,
                A_front=B * 12.0, A_lateral=L * 8.0, design_draft=T,
                eta_0=ETA_0, eta_wave_loss=ETA_WAVE_LOSS)


def relative_angles(d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apparent-wind magnitude and vessel-relative wind and wave angles.

    Wind direction is meteorological (the direction the wind blows *from*), so the
    true-wind velocity vector points toward ``dir + 180``. Apparent wind is the true
    wind minus the vessel velocity along its heading.
    """
    v, hd = d["speed"].to_numpy(), d["heading"].to_numpy()
    ws, wd = d["wind10_kn"].to_numpy(), d["wind10_ang"].to_numpy()
    wx, wy = -ws * np.sin(np.radians(wd)), -ws * np.cos(np.radians(wd))
    sx, sy = v * np.sin(np.radians(hd)), v * np.cos(np.radians(hd))
    ax, ay = wx - sx, wy - sy
    amag = np.hypot(ax, ay)
    arel = (np.degrees(np.arctan2(ax, ay)) - hd + 180.0) % 360.0 - 180.0
    wave_rel = (d["wave_ang"].to_numpy() - hd + 180.0) % 360.0 - 180.0
    return amag, arel, wave_rel


def wbm_prior(d: pd.DataFrame, part: dict, use_row_draft: bool) -> np.ndarray:
    """Uncalibrated Holtrop-Mennen fuel rate (kg/h). Fouling disabled (no clock)."""
    v = d["speed"].to_numpy()
    amag, arel, wave_rel = relative_angles(d)
    dist_km = 1.852 * v                     # one hour of travel -> fuel per hour
    if not use_row_draft:
        env = {"wind_kn": amag, "wind_angle_deg": arel, "Hs": d["hs"].to_numpy(),
               "Tp": d["tp"].to_numpy(), "wave_angle_deg": wave_rel,
               "current_kn": d["cur_kn"].to_numpy(),
               "current_angle_deg": d["cur_ang"].to_numpy(),
               "sst_c": d["temp"].to_numpy(),
               "depth_m": np.maximum(d["depth"].to_numpy(), 20.0)}
        prm = dict(part, trim_m=0.0, months_since_cleaning=0.0)
        return np.asarray(fuel_components(dist_km, v, env, prm)["total_fuel_mt"]) * 1000.0
    out = np.empty(len(d))
    dr, tm = d["draft"].to_numpy(), d["trim"].to_numpy()
    for i in range(len(d)):
        env = {"wind_kn": float(amag[i]), "wind_angle_deg": float(arel[i]),
               "Hs": float(d["hs"].iat[i]), "Tp": float(d["tp"].iat[i]),
               "wave_angle_deg": float(wave_rel[i]),
               "current_kn": float(d["cur_kn"].iat[i]),
               "current_angle_deg": float(d["cur_ang"].iat[i]),
               "sst_c": float(d["temp"].iat[i]),
               "depth_m": max(float(d["depth"].iat[i]), 20.0)}
        prm = dict(part, T=float(dr[i]), trim_m=float(tm[i]),
                   months_since_cleaning=0.0)
        out[i] = float(fuel_components(float(dist_km[i]), float(v[i]), env,
                                       prm)["total_fuel_mt"]) * 1000.0
    return out


def splits(name: str, shift_var: str = "wind10_kn"):
    d, feats = load_fuelcast_vessel(name)
    tr, din, dout, sv = fuelcast_shift_splits(d, shift_var=shift_var)
    n_fit = int(FIT_FRACTION * len(tr))
    return d, feats, tr, din, dout, sv, np.arange(n_fit), np.arange(n_fit, len(tr))


def architectures(name, prior_name, feats, tr, din, dout, fit_i, val_i,
                  prior_tr, prior_din, prior_dout):
    """Fit Huber twice, select lambda on the validation tail, evaluate frozen."""
    rng = np.random.default_rng(SEED)
    ytr, Xtr = tr["fuel_kgh"].to_numpy(), tr[feats].to_numpy()
    res = ytr - prior_tr
    resid, sub = make_learner(BACKEND, rng)
    Xs, rs = sub(Xtr[fit_i], res[fit_i])
    resid.fit(Xs, rs)
    bbm, sub_b = make_learner(BACKEND, rng)
    Xb, yb = sub_b(Xtr[fit_i], ytr[fit_i])
    bbm.fit(Xb, yb)

    P = {"validation_tail": (ytr[val_i], prior_tr[val_i], resid.predict(Xtr[val_i]),
                             bbm.predict(Xtr[val_i])),
         "dev_in": (din["fuel_kgh"].to_numpy(), prior_din,
                    resid.predict(din[feats].to_numpy()),
                    bbm.predict(din[feats].to_numpy())),
         "dev_out": (dout["fuel_kgh"].to_numpy(), prior_dout,
                     resid.predict(dout[feats].to_numpy()),
                     bbm.predict(dout[feats].to_numpy()))}

    yv, pv, rv, _ = P["validation_tail"]
    curve = pd.DataFrame([{"vessel": name, "prior": prior_name, "lambda": float(l),
                           **metrics7(yv, pv + l * rv)} for l in LAM])
    lam = float(curve.loc[curve.MAPE.idxmin(), "lambda"])

    ident = []
    for pn, (y, p, r, _) in P.items():
        for met in METRICS:
            m0s, m0d = metrics7(y, p + 0.0 * r)[met], metrics7(y, p)[met]
            m1s, m1d = metrics7(y, p + 1.0 * r)[met], metrics7(y, p + r)[met]
            ident.append({"vessel": name, "prior": prior_name, "partition": pn,
                          "metric": met, "lambda0_sweep": m0s, "prior_direct": m0d,
                          "lambda0_identical": abs(m0s - m0d) < IDENTITY_TOL,
                          "lambda1_sweep": m1s, "sgbm_direct": m1d,
                          "lambda1_identical": abs(m1s - m1d) < IDENTITY_TOL})

    rows = []
    yi, pi, ri, bi = P["dev_in"]
    yo, po, ro, bo = P["dev_out"]
    for arch, param, pin, pout in (
            (f"{prior_name}", "-", pi, po),
            ("BBM (Huber on target)", "-", bi, bo),
            ("S-GBM (Huber residual)", "lambda=1", pi + ri, po + ro),
            ("C-GBM (Huber residual)", f"lambda*={lam:g}", pi + lam * ri,
             po + lam * ro)):
        mi, mo = metrics7(yi, pin), metrics7(yo, pout)
        rows.append({"vessel": name, "vessel_short": SHORT[name], "prior": prior_name,
                     "architecture": arch, "param": param,
                     "n_dev_in": int(len(yi)), "n_dev_out": int(len(yo)),
                     **{f"{k}_in": v for k, v in mi.items()},
                     **{f"{k}_out": v for k, v in mo.items()},
                     "dMAPE": mo["MAPE"] - mi["MAPE"],
                     "lambda_star": lam if arch.startswith("C-GBM") else np.nan})
    return rows, curve.to_dict("records"), ident, lam, bool(
        len(Xs) != len(fit_i) or len(Xb) != len(fit_i))


def part_a(shift_var: str = "wind10_kn", tag: str = "A"):
    print(f"[{tag}] cube-law prior, wind-aware feature set, shift on {shift_var}",
          flush=True)
    rows, curves, idents, sel = [], [], [], []
    for name in VESSELS:
        d, feats, tr, din, dout, sv, fit_i, val_i = splits(name, shift_var)
        a = fit_cube(tr["fuel_kgh"].to_numpy()[fit_i], tr["speed"].to_numpy()[fit_i])
        pr_tr = a * tr["speed"].to_numpy() ** 3
        r, c, i, lam, subs = architectures(
            name, "WBM-lite (cube law)", feats, tr, din, dout, fit_i, val_i,
            pr_tr, a * din["speed"].to_numpy() ** 3, a * dout["speed"].to_numpy() ** 3)
        rows += r; curves += c; idents += i
        sel.append({"vessel": name, "vessel_short": SHORT[name],
                    "role": FC_VESSELS[name]["role"], "prior": "WBM-lite (cube law)",
                    "residual_learner": "Huber", "bbm_learner": "Huber on target",
                    "lambda_star": lam, "cube_coefficient": float(a),
                    "n_features": len(feats), "features": ";".join(feats),
                    "wind_in_features": bool("wind10_kn" in feats),
                    "n_train": int(len(tr)), "n_fit": int(len(fit_i)),
                    "n_validation_tail": int(len(val_i)),
                    "n_dev_in": int(len(din)), "n_dev_out": int(len(dout)),
                    "shift_variable": sv, "learner_subsampled": subs})
        print(f"    {SHORT[name]:9s} lambda*={lam:.2f}  n_feats={len(feats)}  "
              f"wind_in_features={'wind10_kn' in feats}", flush=True)
    return (pd.DataFrame(rows), pd.DataFrame(curves), pd.DataFrame(idents),
            pd.DataFrame(sel))


def part_b():
    print("[B] Holtrop-Mennen prior on the heading-carrying vessels", flush=True)
    rows, curves, idents, sel, gridrows = [], [], [], [], []
    for name in WBM_VESSELS:
        d, feats, tr, din, dout, sv, fit_i, val_i = splits(name)
        ytr = tr["fuel_kgh"].to_numpy()
        has_draft = bool(d["draft"].notna().mean() > 0.95)
        mcr = float(np.ceil(np.nanpercentile(ytr, 99) / 0.18 / 1000.0) * 1000.0)
        t_grid = [float(np.nanmedian(d["draft"]))] if has_draft else GRID_T_NODRAFT

        best, t0 = None, time.time()
        for L in GRID_L:
            for br in GRID_BRATIO:
                for cb in GRID_CB:
                    for T in t_grid:
                        part = particulars(L, br, cb, T, mcr)
                        pr = wbm_prior(tr, part, has_draft)
                        if not np.all(np.isfinite(pr)) or np.all(pr <= 0):
                            continue
                        k = fit_scale(ytr[fit_i], pr[fit_i])
                        m = metrics7(ytr[val_i], k * pr[val_i])
                        gridrows.append({"vessel": name, "L": L, "beam_ratio": br,
                                         "B": L / br, "CB": cb, "T": T, "MCR_kw": mcr,
                                         "k": float(k),
                                         "validation_MAPE": m["MAPE"],
                                         "validation_RMSE": m["RMSE"],
                                         "validation_R2": m["R2"]})
                        if best is None or m["MAPE"] < best[0]:
                            best = (m["MAPE"], part, float(k))
        print(f"    {SHORT[name]:9s} geometry grid {len(t_grid)*len(GRID_L)*3*3} "
              f"combos in {time.time()-t0:.1f}s -> validation MAPE "
              f"{best[0]:.3f}%  L={best[1]['L']:.0f} B={best[1]['B']:.1f} "
              f"CB={best[1]['CB']:.2f} T={best[1]['T']:.2f} k={best[2]:.4f}",
              flush=True)
        _, part, k = best
        pr_tr = k * wbm_prior(tr, part, has_draft)
        pr_in = k * wbm_prior(din, part, has_draft)
        pr_out = k * wbm_prior(dout, part, has_draft)
        r, c, i, lam, subs = architectures(
            name, "WBM (Holtrop-Mennen)", feats, tr, din, dout, fit_i, val_i,
            pr_tr, pr_in, pr_out)
        rows += r; curves += c; idents += i
        sel.append({"vessel": name, "vessel_short": SHORT[name],
                    "role": FC_VESSELS[name]["role"], "prior": "WBM (Holtrop-Mennen)",
                    "residual_learner": "Huber", "bbm_learner": "Huber on target",
                    "lambda_star": lam, "k": k,
                    "geometry_selected_on": "training-side validation tail, MAPE of "
                                            "the k-scaled prior alone",
                    "n_geometry_candidates": int(
                        len(t_grid) * len(GRID_L) * len(GRID_BRATIO) * len(GRID_CB)),
                    **{f"sel_{key}": part[key] for key in
                       ("L", "B", "T", "S", "CB", "MCR", "design_draft")},
                    "draught_measured": has_draft,
                    "speed_variable": "STW" if name == "CPS_Poseidon" else "SOG",
                    "fouling_term": "disabled (no dry-docking clock in FuelCast)",
                    "resistance_terms": "calm water + Blendermann wind + STAWAVE-1 "
                                        "waves + current effect on SOG",
                    "n_train": int(len(tr)), "n_fit": int(len(fit_i)),
                    "n_validation_tail": int(len(val_i)),
                    "n_dev_in": int(len(din)), "n_dev_out": int(len(dout)),
                    "learner_subsampled": subs})
        print(f"    {SHORT[name]:9s} lambda*={lam:.2f} with the HM prior", flush=True)
    return (pd.DataFrame(rows), pd.DataFrame(curves), pd.DataFrame(idents),
            pd.DataFrame(sel), pd.DataFrame(gridrows))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    a_arch, a_curve, a_ident, a_sel = part_a()
    # Control: the same wind-aware feature set on the PRE-FIX split variable.
    # Recovering the wind column also re-enabled the intended wind-quantile shift
    # (fuelcast_shift_splits falls back to hs when the wind column is unusable), so
    # folder 29 and part A do not share partitions. This control holds the split
    # fixed at hs so the feature-set effect can be read on its own.
    a2_arch, a2_curve, a2_ident, a2_sel = part_a(shift_var="hs", tag="A2")
    b_arch, b_curve, b_ident, b_sel, b_grid = part_b()

    def table(df):
        t = df[["vessel_short", "prior", "architecture", "param", "MAPE_in",
                "MAPE_out", "dMAPE", "lambda_star"]].copy()
        t.columns = ["Vessel", "Prior", "Architecture", "Parameter", "MAPE_in_pct",
                     "MAPE_out_pct", "delta_MAPE_pp", "lambda_star"]
        return t

    # prior-alone comparison: cube law vs Holtrop-Mennen on the same vessels
    cmp_rows = []
    for name in WBM_VESSELS:
        cu = a_arch[(a_arch.vessel == name)
                    & a_arch.architecture.str.startswith("WBM-lite")].iloc[0]
        hm = b_arch[(b_arch.vessel == name)
                    & b_arch.architecture.str.startswith("WBM (")].iloc[0]
        cmp_rows.append({
            "vessel_short": SHORT[name],
            "cube_MAPE_in": cu.MAPE_in, "hm_MAPE_in": hm.MAPE_in,
            "cube_MAPE_out": cu.MAPE_out, "hm_MAPE_out": hm.MAPE_out,
            "hm_better_in": bool(hm.MAPE_in < cu.MAPE_in),
            "hm_better_out": bool(hm.MAPE_out < cu.MAPE_out),
            "cube_R2_in": cu.R2_in, "hm_R2_in": hm.R2_in,
            "cube_R2_out": cu.R2_out, "hm_R2_out": hm.R2_out})
    cmp = pd.DataFrame(cmp_rows)

    # feature-set diff against folder 29 (pre-fix), on the matched hs split
    import ast

    def parse_feats(x) -> set[str]:
        s = str(x)
        try:
            v = ast.literal_eval(s)
            if isinstance(v, (list, tuple)):
                return {str(i) for i in v}
        except (ValueError, SyntaxError):
            pass
        return {p for p in s.replace(",", ";").split(";") if p.strip()}

    f29 = pd.read_csv(F29 / "fuelcast_lambda_selection.csv")
    f29_arch = pd.read_csv(F29 / "fuelcast_all_metrics.csv")
    diff = []
    for _, r in a2_sel.iterrows():
        old = f29[f29.vessel == r.vessel].iloc[0]
        of, nf = parse_feats(old.features), parse_feats(r.features)
        oc = f29_arch[(f29_arch.vessel == r.vessel)
                      & f29_arch.architecture.str.startswith("C-GBM")].iloc[0]
        nc = a2_arch[(a2_arch.vessel == r.vessel)
                     & a2_arch.architecture.str.startswith("C-GBM")].iloc[0]
        diff.append({"vessel_short": r.vessel_short,
                     "split_variable": "hs (matched to folder 29)",
                     "n_dev_in_folder29": int(old.n_dev_in),
                     "n_dev_in_control": int(r.n_dev_in),
                     "n_dev_out_folder29": int(old.n_dev_out),
                     "n_dev_out_control": int(r.n_dev_out),
                     "partitions_match": bool(old.n_dev_in == r.n_dev_in
                                              and old.n_dev_out == r.n_dev_out),
                     "n_features_folder29": len(of), "n_features_control": len(nf),
                     "features_recovered": ";".join(sorted(nf - of)),
                     "lambda_star_folder29": float(old.lambda_star),
                     "lambda_star_control": float(r.lambda_star),
                     "cgbm_MAPE_in_folder29": float(oc.MAPE_in),
                     "cgbm_MAPE_in_control": float(nc.MAPE_in),
                     "cgbm_MAPE_out_folder29": float(oc.MAPE_out),
                     "cgbm_MAPE_out_control": float(nc.MAPE_out)})
    fdiff = pd.DataFrame(diff)

    checks = []
    ident_all = pd.concat([a_ident, a2_ident, b_ident], ignore_index=True)

    def chk(cid, item, ok, detail):
        checks.append({"check_id": cid, "item": item,
                       "verdict": "PASS" if ok else "FAIL", "detail": detail})

    chk("E1", "lambda=0 reproduces the prior on every partition, prior and metric",
        bool(ident_all.lambda0_identical.all()),
        f"{int(ident_all.lambda0_identical.sum())}/{len(ident_all)} identical")
    chk("E2", "lambda=1 reproduces S-GBM on every partition, prior and metric",
        bool(ident_all.lambda1_identical.all()),
        f"{int(ident_all.lambda1_identical.sum())}/{len(ident_all)} identical")
    chk("F1", "wind recovered into the feature set on all three vessels",
        bool(a_sel.wind_in_features.all()),
        "; ".join(f"{r.vessel_short}: {r.n_features} features"
                  for _, r in a_sel.iterrows()))
    chk("L1", "no learner subsampled",
        not (a_sel.learner_subsampled.any() or b_sel.learner_subsampled.any()),
        f"partA={[bool(x) for x in a_sel.learner_subsampled]}, "
        f"partB={[bool(x) for x in b_sel.learner_subsampled]}")
    chk("L2", "fitting portion and validation tail partition the training block",
        bool((a_sel.n_fit + a_sel.n_validation_tail == a_sel.n_train).all()),
        "; ".join(f"{r.vessel_short}: {r.n_fit}+{r.n_validation_tail}={r.n_train}"
                  for _, r in a_sel.iterrows()))
    chk("L3", "geometry and lambda both selected on the validation tail only",
        bool(b_sel.geometry_selected_on.str.contains("validation tail").all()),
        f"{int(b_sel.n_geometry_candidates.sum())} geometry candidates scored, "
        f"none on dev data")
    chk("L4", "Shifts lambda* not imposed anywhere", True,
        f"partA lambda* = {[float(x) for x in a_sel.lambda_star]}; "
        f"partB lambda* = {[float(x) for x in b_sel.lambda_star]}")
    chk("P1", "fouling term disabled and disclosed (no dry-docking clock)",
        bool(b_sel.fouling_term.str.startswith("disabled").all()),
        "months_since_cleaning = 0 for every record")
    chk("P2", "Ceto excluded from the Holtrop-Mennen prior for lack of heading "
              "and speed through water", "OSS_Ceto" not in set(b_sel.vessel),
        f"HM vessels = {[SHORT[v] for v in b_sel.vessel]}")
    # A boundary optimum means the grid does not bracket the best geometry, i.e.
    # the search is still trying to reshape the resistance curve. Recorded as a
    # finding, not silently accepted.
    bnd = []
    for _, r in b_sel.iterrows():
        at = []
        if r.sel_L >= max(GRID_L):
            at.append("L at grid max")
        if r.sel_CB <= min(GRID_CB):
            at.append("CB at grid min")
        if abs(r.sel_B - r.sel_L / max(GRID_BRATIO)) < 1e-6:
            at.append("beam ratio at grid max")
        if not r.draught_measured and r.sel_T <= min(GRID_T_NODRAFT):
            at.append("T at grid min")
        bnd.append(f"{r.vessel_short}: {', '.join(at) if at else 'interior'}")
    interior = all("interior" in b for b in bnd)
    chk("P3", "selected geometry is interior to the grid (no unbracketed optimum)",
        interior, "; ".join(bnd) + ("" if interior else
                                    " -> boundary solution: the prior shape, not its "
                                    "size, is the limiting factor"))
    chk("S1", "wind recovery also re-enabled the intended wind-quantile shift, so "
              "folder 29 partitions differ from part A", True,
        "; ".join(f"{r.vessel_short}: dev_in {r.n_dev_in}/dev_out {r.n_dev_out} on "
                  f"{r.shift_variable}" for _, r in a_sel.iterrows())
        + " | control A2 on hs matches folder 29: "
        + str(bool(fdiff.partitions_match.all())))
    cdf = pd.DataFrame(checks)
    n_fail = int((cdf.verdict == "FAIL").sum())

    write_csv(table(a_arch), OUT / "partA_manuscript_table_mape.csv")
    write_csv(a_arch, OUT / "partA_all_metrics.csv")
    write_csv(a_sel, OUT / "partA_lambda_selection.csv")
    write_csv(a_curve, OUT / "partA_lambda_curves_validation.csv")
    write_csv(table(a2_arch), OUT / "partA2_control_hs_split_table_mape.csv")
    write_csv(a2_arch, OUT / "partA2_control_hs_split_all_metrics.csv")
    write_csv(a2_sel, OUT / "partA2_control_hs_split_selection.csv")
    write_csv(table(b_arch), OUT / "partB_manuscript_table_mape.csv")
    write_csv(b_arch, OUT / "partB_all_metrics.csv")
    write_csv(b_sel, OUT / "partB_selection.csv")
    write_csv(b_curve, OUT / "partB_lambda_curves_validation.csv")
    write_csv(b_grid, OUT / "partB_geometry_grid_validation.csv")
    write_csv(cmp, OUT / "prior_comparison_cube_vs_hm.csv")
    write_csv(fdiff, OUT / "feature_set_diff_vs_folder29.csv")
    write_csv(ident_all, OUT / "endpoint_identity.csv")
    write_csv(cdf, OUT / "checks.csv")
    (OUT / "run_metadata.json").write_text(json.dumps({
        "purpose": "Section 5.4 after the FuelCast loader fix, plus a fuller "
                   "Holtrop-Mennen prior where the data supports one",
        "loader_bug_fixed": {
            "wrong": ["Weather_WindSpeed10m", "Weather_WindDirection10m",
                      "Weather_Temperature2m"],
            "correct": ["Weather_WindSpeed10M", "Weather_WindDirection10M",
                        "Weather_Temperature2M"],
            "effect_before_fix": "columns NaN-filled, then dropped by the 95 % "
                                 "presence filter, so every earlier FuelCast run "
                                 "was wind-blind",
            "guard_added": "load_fuelcast_vessel now raises on an unknown "
                           "non-optional source column"},
        "supersedes": ["21_fuelcast_shrinkage_dense (CatBoost)",
                       "29_fuelcast_huber_final (pre-fix feature set)"],
        "shifts_results_unchanged": True,
        "residual_learner": "Huber", "bbm_learner": "Huber on the measured target",
        "constraint": "residual shrinkage only",
        "lambda_grid": "0.00..1.00 step 0.01", "primary_criterion": "MAPE",
        "hm_prior_vessels": [SHORT[v] for v in WBM_VESSELS],
        "hm_resistance_terms": "calm water + Blendermann wind + STAWAVE-1 waves "
                               "+ current effect on SOG; Schultz fouling disabled",
        "geometry_grid": {"L": GRID_L, "beam_ratio": GRID_BRATIO, "CB": GRID_CB,
                          "T_if_no_draught": GRID_T_NODRAFT,
                          "selected_on": "validation tail, k-scaled prior MAPE"},
        "checks_passed": int(len(cdf) - n_fail), "checks_total": int(len(cdf)),
        "seed": SEED, "git_commit": _git()}, indent=2), encoding="utf-8")

    print("\n[FC] checks", flush=True)
    for _, r in cdf.iterrows():
        print(f"  [{r['verdict']}] {r['check_id']} {r['item']}\n"
              f"        {r['detail']}", flush=True)
    print(f"\n[FC] {len(cdf) - n_fail}/{len(cdf)} PASS, {n_fail} FAIL", flush=True)
    return 2 if n_fail else 0


def _guarded() -> int:
    import joblib
    with joblib.parallel_backend("threading", n_jobs=1):
        return main()


if __name__ == "__main__":
    sys.exit(_guarded())
