"""FINAL manuscript results, stage 1: core regeneration under the final protocol.

Frozen final configuration, used everywhere in this stage:

    WBM    k * F_phys,  k fitted on the 60 % training block only
    BBM    ExtraTrees, selected independently on the direct target y
    S-GBM  WBM + Huber residual
    C-GBM  WBM + 0.70 * Huber residual

Data: full ``train.csv`` (530,706 records) split chronologically 60/20/20, with
the last 10 % of the training block reserved for early stopping. No 40,000-record
thinning. Neither ``dev_in`` nor ``dev_out`` influences any fit or selection.

Produces the architecture comparison, the lambda sweep and its metric
sensitivity, the physical diagnostics (with the algebraic ones flagged),
measured-versus-predicted, operating regimes, split-conformal uncertainty,
multi-gas/CO2e/CII propagation, the benchmark-shift characterisation, and fresh
block-bootstrap statistics.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from .common_rev import REPO_ROOT, REV_FINAL, SEED, write_csv, write_text
from .rf32_full_data_602020 import (
    ES_FRACTION, FEATS, LOWER_BETTER, METRICS, PRIMARY, build_splits,
    fit_learner, metrics7, physics,
)

_E = REPO_ROOT / "experiments"
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import CONFIG, fit_scale, speed_sweep_frame  # noqa: E402
from benchmark_measured.run_cii_probabilistic import (  # noqa: E402
    ASSUMED_YEAR, DWT_CENTRAL_T, SHIP_TYPES, cii_rating, cii_required,
)
from benchmark_measured.run_uq_emissions import power_to_fuel_mt_h  # noqa: E402
from vessel_fuel import fuel_to_co2e  # noqa: E402
from .common_rev import physics_power_eta  # noqa: E402

OUT = REV_FINAL / "FINAL_MANUSCRIPT_RESULTS"
LAMBDA_STAR = 0.70
BBM_KEY, RESID_KEY = "et", "huber"
LAM = np.round(np.arange(0.0, 1.0 + 1e-9, 0.01), 4)
B_BOOT, BLOCK, ALPHA_CONF = 2000, 24, 0.10
ARCHS = ["WBM", "BBM", "S-GBM", "C-GBM"]
FUEL_TYPE, GWP = "MDO", "AR6"
GWP_CH4, GWP_N2O = 29.8, 273.0
MCR = CONFIG["particulars"]["MCR"]
SECTORS = [("head", 0, 30), ("bow", 30, 60), ("beam", 60, 120),
           ("quarter", 120, 150), ("following", 150, 181)]


def _git() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def smd(a, b) -> float:
    s = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2.0)
    return float((np.mean(a) - np.mean(b)) / max(s, 1e-12))


def block_idx(n, rng):
    nb = int(np.ceil(n / BLOCK))
    st = rng.integers(0, max(n - BLOCK + 1, 1), size=nb)
    return np.concatenate([np.arange(s, min(s + BLOCK, n)) for s in st])[:n]


def holm(p):
    order = np.argsort(p)
    m, out, prev = len(p), [0.0] * len(p), 0.0
    for rank, i in enumerate(order):
        adj = min(1.0, max(prev, (m - rank) * p[i]))
        out[i], prev = adj, adj
    return out


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = build_splits()
    ph = physics(parts)
    tr = parts["train"]
    y_tr, X_tr = tr["power"].to_numpy(), tr[FEATS].to_numpy()
    k = fit_scale(y_tr, ph["train"])
    res_tr = y_tr - k * ph["train"]
    fi, ei = parts["_fit_idx"], parts["_es_idx"]
    print(f"[F] k = {k:.6f}", flush=True)

    bbm, info_b = fit_learner(BBM_KEY, X_tr, y_tr, fi, ei)
    resid, info_r = fit_learner(RESID_KEY, X_tr, res_tr, fi, ei)
    models = {"bbm": bbm, "resid": resid}

    # ---------- 1. data protocol ----------
    proto = []
    for nm in ("train", "validation", "internal_test", "dev_in", "dev_out"):
        d = parts[nm]
        proto.append({"partition": nm, "n": int(len(d)),
                      "time_id_min": int(d["time_id"].min()),
                      "time_id_max": int(d["time_id"].max()),
                      "role": {"train": "parameter fitting (+ early stopping tail)",
                               "validation": "model and lambda selection",
                               "internal_test": "frozen check, never used for selection",
                               "dev_in": "benchmark holdout",
                               "dev_out": "benchmark holdout"}[nm]})
    proto += [{"partition": "train:fitting", "n": int(len(fi)),
               "time_id_min": int(tr["time_id"].iloc[0]),
               "time_id_max": int(tr["time_id"].iloc[len(fi) - 1]),
               "role": f"first {1 - ES_FRACTION:.0%} of the training block"},
              {"partition": "train:early_stopping", "n": int(len(ei)),
               "time_id_min": int(tr["time_id"].iloc[len(fi)]),
               "time_id_max": int(tr["time_id"].iloc[-1]),
               "role": f"last {ES_FRACTION:.0%} of the training block"}]
    proto_df = pd.DataFrame(proto)
    write_csv(proto_df, OUT / "T01_data_protocol.csv")

    # ---------- 2. benchmark-shift characterisation (verified, not copied) ----------
    di, do = parts["dev_in"].copy(), parts["dev_out"].copy()
    for d in (di, do):
        a = np.abs(d["wind_ang"].to_numpy())
        s = np.empty(len(d), dtype=object)
        for nm2, lo, hi in SECTORS:
            s[(a >= lo) & (a < hi)] = nm2
        d["sector"] = s
    shift_rows = []
    for v, lbl in (("months", "time since drydock (months)"),
                   ("hs", "significant wave height (m)"),
                   ("stw", "speed through water (kn)"),
                   ("wind_kn", "apparent wind magnitude (kn)"),
                   ("draft", "mean draught (m)"),
                   ("power", "measured shaft power (kW)")):
        a, b = do[v].to_numpy(), di[v].to_numpy()
        shift_rows.append({"variable": lbl, "dev_in_n": len(di), "dev_out_n": len(do),
                           "dev_in_mean": float(np.mean(b)),
                           "dev_out_mean": float(np.mean(a)),
                           "dev_in_median": float(np.median(b)),
                           "dev_out_median": float(np.median(a)),
                           "dev_in_sd": float(np.std(b)), "dev_out_sd": float(np.std(a)),
                           "SMD_devout_minus_devin": smd(a, b)})
    shift_df = pd.DataFrame(shift_rows).sort_values(
        "SMD_devout_minus_devin", key=abs, ascending=False)
    write_csv(shift_df, OUT / "T02_shift_characterization.csv")
    sec_rows = [{"sector": s, "dev_in_fraction": float((di.sector == s).mean()),
                 "dev_out_fraction": float((do.sector == s).mean()),
                 "difference_pp": float(((do.sector == s).mean()
                                         - (di.sector == s).mean()) * 100)}
                for s, _, _ in SECTORS]
    sec_df = pd.DataFrame(sec_rows)
    write_csv(sec_df, OUT / "T02b_wind_sector_composition.csv")
    print("[F] shift characterisation verified from data:", flush=True)
    for _, r in shift_df.iterrows():
        print(f"      {r['variable']:34s} SMD {r['SMD_devout_minus_devin']:+.3f}",
              flush=True)

    # ---------- 3. architecture comparison on four partitions ----------
    def preds_for(nm):
        d = parts[nm]
        X = d[FEATS].to_numpy()
        base = k * ph[nm]
        r = resid.predict(X)
        return d["power"].to_numpy(), {
            "WBM": base, "BBM": bbm.predict(X), "S-GBM": base + r,
            "C-GBM": base + LAMBDA_STAR * r}

    arch_rows, pred_cache = [], {}
    for nm in ("validation", "internal_test", "dev_in", "dev_out"):
        y, pr = preds_for(nm)
        pred_cache[nm] = (y, pr)
        for a in ARCHS:
            arch_rows.append({"partition": nm, "architecture": a, "n": int(len(y)),
                              **metrics7(y, pr[a])})
    arch = pd.DataFrame(arch_rows)
    write_csv(arch, OUT / "T04_architecture_all_partitions.csv")
    piv = arch.pivot_table(index="architecture", columns="partition", values=METRICS)
    ch = []
    for a in ARCHS:
        rec = {"architecture": a}
        for mt in METRICS:
            i = float(arch[(arch.partition == "dev_in")
                           & (arch.architecture == a)][mt].iloc[0])
            o = float(arch[(arch.partition == "dev_out")
                           & (arch.architecture == a)][mt].iloc[0])
            rec[f"{mt}_dev_in"], rec[f"{mt}_dev_out"] = i, o
            rec[f"{mt}_change"] = o - i
            rec[f"{mt}_ratio"] = o / i if abs(i) > 1e-12 else np.nan
        ch.append(rec)
    write_csv(pd.DataFrame(ch), OUT / "T05_dev_in_to_dev_out_changes.csv")

    # ---------- 4. lambda sweep + endpoint identity ----------
    yv = parts["validation"]["power"].to_numpy()
    base_v = k * ph["validation"]
    r_v = resid.predict(parts["validation"][FEATS].to_numpy())
    sweep = pd.DataFrame([{"lambda": float(l), **metrics7(yv, base_v + l * r_v)}
                          for l in LAM])
    write_csv(sweep, OUT / "T06_lambda_sweep_validation.csv")
    m0, m1 = metrics7(yv, base_v), metrics7(yv, base_v + r_v)
    ep = []
    for mt in METRICS:
        v0 = float(sweep[np.isclose(sweep["lambda"], 0)][mt].iloc[0])
        v1 = float(sweep[np.isclose(sweep["lambda"], 1)][mt].iloc[0])
        ep.append({"metric": mt, "lambda0_sweep": v0, "WBM_direct": m0[mt],
                   "lambda0_identical": bool(abs(v0 - m0[mt]) < 1e-9),
                   "lambda1_sweep": v1, "SGBM_direct": m1[mt],
                   "lambda1_identical": bool(abs(v1 - m1[mt]) < 1e-9)})
    ep_df = pd.DataFrame(ep)
    write_csv(ep_df, OUT / "T07_lambda_endpoint_identity.csv")
    assert ep_df.lambda0_identical.all() and ep_df.lambda1_identical.all()
    sens = []
    for mt in METRICS:
        i = sweep[mt].idxmin() if LOWER_BETTER[mt] else sweep[mt].idxmax()
        sens.append({"metric": mt, "optimal_lambda": float(sweep.loc[i, "lambda"]),
                     "value_at_optimum": float(sweep.loc[i, mt]),
                     "value_at_0_WBM": float(sweep[np.isclose(sweep["lambda"], 0)][mt].iloc[0]),
                     "value_at_lambda_star": float(sweep[np.isclose(sweep["lambda"], LAMBDA_STAR)][mt].iloc[0]),
                     "value_at_1_SGBM": float(sweep[np.isclose(sweep["lambda"], 1)][mt].iloc[0]),
                     "is_primary_criterion": mt == PRIMARY})
    sens_df = pd.DataFrame(sens)
    write_csv(sens_df, OUT / "T08_lambda_metric_sensitivity.csv")
    print(f"[F] lambda endpoints exact; optima by metric: "
          f"{sens_df.optimal_lambda.min():g}-{sens_df.optimal_lambda.max():g}",
          flush=True)

    # ---------- 5. physical diagnostics, algebraic ones flagged ----------
    sw = speed_sweep_frame(tr, n=60)
    ph_sw = physics_power_eta(sw, None, CONFIG["particulars"]["eta_wave_loss"])
    Xsw = sw[FEATS].to_numpy()
    base_sw = k * ph_sw
    r_sw = resid.predict(Xsw)
    from scipy.stats import t as tdist
    phys_rows = []
    for a in ARCHS:
        p = {"WBM": base_sw, "BBM": bbm.predict(Xsw), "S-GBM": base_sw + r_sw,
             "C-GBM": base_sw + LAMBDA_STAR * r_sw}[a]
        v = sw["stw"].to_numpy()
        viol = float(np.mean(np.diff(p) < 0.0))
        ok = p > 0
        x, yy = np.log(v[ok]), np.log(p[ok])
        b_, c_ = np.polyfit(x, yy, 1)
        nn = len(x)
        s2 = float(np.sum((yy - (b_ * x + c_)) ** 2) / (nn - 2))
        se = float(np.sqrt(s2 / np.sum((x - x.mean()) ** 2)))
        tc = float(tdist.ppf(0.975, nn - 2))
        d = {}
        for nm in ("dev_in", "dev_out"):
            _, pr = pred_cache[nm]
            wbmp = pr["WBM"]
            d[nm] = float(np.mean(np.abs(pr[a] - wbmp) / np.maximum(wbmp, 1e-9)))
        phys_rows.append({"architecture": a, "mono_violation_frac": viol,
                          "speed_exponent_b": float(b_),
                          "exponent_lo95": float(b_ - tc * se),
                          "exponent_hi95": float(b_ + tc * se),
                          "D_phys_dev_in": d["dev_in"],
                          "D_phys_dev_out": d["dev_out"],
                          "independent_of_lambda_by_construction":
                              a not in ("C-GBM",)})
    phys = pd.DataFrame(phys_rows)
    sg = phys[phys.architecture == "S-GBM"].iloc[0]
    cg = phys[phys.architecture == "C-GBM"].iloc[0]
    phys["D_phys_is_algebraic_in_lambda"] = [False, False, False, True]
    ratio_in = cg.D_phys_dev_in / sg.D_phys_dev_in
    ratio_out = cg.D_phys_dev_out / sg.D_phys_dev_out
    write_csv(phys, OUT / "T09_physical_diagnostics.csv")
    write_csv(pd.DataFrame([{
        "quantity": "D_phys",
        "C_GBM_over_S_GBM_dev_in": float(ratio_in),
        "C_GBM_over_S_GBM_dev_out": float(ratio_out),
        "lambda_star": LAMBDA_STAR,
        "matches_lambda_exactly": bool(abs(ratio_in - LAMBDA_STAR) < 1e-9
                                       and abs(ratio_out - LAMBDA_STAR) < 1e-9),
        "verdict": ("ALGEBRAIC: D_phys(C-GBM) = lambda * D_phys(S-GBM) by "
                    "construction; it is a restatement of lambda and is NOT "
                    "independent empirical evidence"),
        "recommendation": "report as descriptive, or remove as evidence"}]),
        OUT / "T09b_D_phys_algebraic_check.csv")
    print(f"[F] D_phys ratio C-GBM/S-GBM = {ratio_in:.6f} (dev_in), "
          f"{ratio_out:.6f} (dev_out); lambda = {LAMBDA_STAR}", flush=True)

    # ---------- 6. measured vs predicted ----------
    mvp = []
    for nm in ("dev_in", "dev_out"):
        y, pr = pred_cache[nm]
        for a in ARCHS:
            sl, ic = np.polyfit(y, pr[a], 1)
            mvp.append({"partition": nm, "architecture": a, "n": int(len(y)),
                        **metrics7(y, pr[a]), "regression_slope": float(sl),
                        "regression_intercept_kw": float(ic)})
    write_csv(pd.DataFrame(mvp), OUT / "T10_measured_vs_predicted.csv")

    # ---------- 7. operating regimes ----------
    edges = {"speed_kn": np.quantile(di["stw"], [0, .25, .5, .75, 1]),
             "wind_kn": np.quantile(di["wind_kn"], [0, .25, .5, .75, 1]),
             "sea_state_hs_m": np.array([0.0, 1.25, 2.5, 4.0, np.inf]),
             "engine_load": np.array([0.0, 0.25, 0.50, 0.75, 1.01])}
    var_of = {"speed_kn": "stw", "wind_kn": "wind_kn",
              "sea_state_hs_m": "hs", "engine_load": None}
    reg = []
    for regime, e in edges.items():
        for nm in ("dev_in", "dev_out"):
            d = parts[nm]
            y, pr = pred_cache[nm]
            v = (y / MCR if regime == "engine_load" else d[var_of[regime]].to_numpy())
            b_ = np.clip(np.digitize(v, e[1:-1]), 0, len(e) - 2)
            for bi in range(len(e) - 1):
                m_ = b_ == bi
                if m_.sum() < 30:
                    continue
                lab = (f"[{e[bi]:.3g}, {e[bi+1]:.3g})" if np.isfinite(e[bi + 1])
                       else f"[{e[bi]:.3g}, inf)")
                for a in ARCHS:
                    reg.append({"regime": regime, "bin_range": lab, "partition": nm,
                                "architecture": a, "n": int(m_.sum()),
                                **metrics7(y[m_], pr[a][m_])})
    write_csv(pd.DataFrame(reg), OUT / "T11_operating_regimes.csv")

    # ---------- 8. split-conformal uncertainty ----------
    q_by = {}
    for a in ARCHS:
        pv = {"WBM": base_v, "BBM": bbm.predict(parts["validation"][FEATS].to_numpy()),
              "S-GBM": base_v + r_v, "C-GBM": base_v + LAMBDA_STAR * r_v}[a]
        q_by[a] = float(np.quantile(np.abs(yv - pv), 1.0 - ALPHA_CONF))
    unc = []
    for nm in ("internal_test", "dev_in", "dev_out"):
        y, pr = pred_cache[nm]
        for a in ARCHS:
            lo, hi = np.maximum(pr[a] - q_by[a], 0.0), pr[a] + q_by[a]
            unc.append({"partition": nm, "architecture": a, "n": int(len(y)),
                        "nominal_coverage": 1 - ALPHA_CONF,
                        "empirical_coverage": float(np.mean((y >= lo) & (y <= hi))),
                        "coverage_error": float(np.mean((y >= lo) & (y <= hi))
                                                - (1 - ALPHA_CONF)),
                        "mean_interval_width_kw": float(np.mean(hi - lo)),
                        "conformal_radius_kw": q_by[a]})
    write_csv(pd.DataFrame(unc), OUT / "T12_conformal_uncertainty.csv")

    # ---------- 9. multi-gas / CO2e / CII ----------
    gas, comp, cii = [], [], []
    for nm in ("dev_in", "dev_out"):
        d = parts[nm]
        y, pr = pred_cache[nm]
        series = [("Measured-power reference", y)] + [(a, pr[a]) for a in ARCHS]
        for lbl, p in series:
            fuel = power_to_fuel_mt_h(p)
            load = np.clip(p / MCR, 0.0, 1.0)
            g = fuel_to_co2e(fuel, load_fraction=load, fuel_type=FUEL_TYPE, gwp=GWP)
            co2 = np.asarray(g["co2_mt"], float)
            co2e = np.asarray(g["co2e_mt"], float)
            gas.append({"partition": nm, "series": lbl, "n_hours": int(len(p)),
                        "fuel_t": float(fuel.sum()), "co2_t": float(co2.sum()),
                        "ch4_kg": float(np.asarray(g["ch4_kg"], float).sum()),
                        "n2o_kg": float(np.asarray(g["n2o_kg"], float).sum()),
                        "co2e_t": float(co2e.sum()),
                        "classification": ("measured power -> CALCULATED gases"
                                           if lbl.startswith("Measured")
                                           else "predicted power -> CALCULATED gases")})
            comp.append({"partition": nm, "series": lbl,
                         "co2_share_pct": float(co2.sum() / co2e.sum() * 100),
                         "co2e_per_fuel_t_per_t": float(co2e.sum() / fuel.sum())})
            dist_nm = float(np.sum(d["stw"].to_numpy()))
            for st in SHIP_TYPES:
                req = cii_required(DWT_CENTRAL_T, ASSUMED_YEAR, st)
                att = float(co2.sum() * 1e6) / max(DWT_CENTRAL_T * dist_nm, 1e-9)
                cii.append({"partition": nm, "series": lbl, "ship_type": st,
                            "attained_cii": att, "required_cii": req,
                            "ratio": att / req, "rating": cii_rating(att, req, st),
                            "gas_basis": "CO2 only, tank-to-wake"})
    gas_df = pd.DataFrame(gas)
    write_csv(gas_df, OUT / "T13_multigas.csv")
    write_csv(pd.DataFrame(comp), OUT / "T13b_gas_composition.csv")
    write_csv(pd.DataFrame(cii), OUT / "T14_cii.csv")

    # ---------- 10. statistics ----------
    st_rows = []
    for nm in ("dev_in", "dev_out"):
        y, pr = pred_cache[nm]
        n = len(y)
        raw, recs = [], []
        for ref in ("WBM", "BBM", "S-GBM"):
            point = metrics7(y, pr["C-GBM"])["MAPE"] - metrics7(y, pr[ref])["MAPE"]
            rng = np.random.default_rng(SEED)
            dd = np.empty(B_BOOT)
            for b in range(B_BOOT):
                i = block_idx(n, rng)
                dd[b] = (metrics7(y[i], pr["C-GBM"][i])["MAPE"]
                         - metrics7(y[i], pr[ref][i])["MAPE"])
            lo, hi = float(np.percentile(dd, 2.5)), float(np.percentile(dd, 97.5))
            pv = float(np.clip(2 * min(float(np.mean(dd <= 0)),
                                       float(np.mean(dd >= 0))), 1 / B_BOOT, 1))
            raw.append(pv)
            recs.append({"partition": nm, "comparison": f"C-GBM minus {ref}",
                         "reference": ref, "metric": "MAPE",
                         "effect_size_pp": point, "ci95_lo": lo, "ci95_hi": hi,
                         "p_raw": pv, "method": "paired moving-block bootstrap",
                         "block_length_hours": BLOCK, "n_resamples": B_BOOT,
                         "n_records": n})
        for r_, p_ in zip(recs, holm(raw)):
            r_["p_holm"] = p_
            r_["statistically_resolved"] = bool(
                p_ < 0.05 and (r_["ci95_lo"] > 0 or r_["ci95_hi"] < 0))
            st_rows.append(r_)
    st = pd.DataFrame(st_rows)
    write_csv(st, OUT / "T15_statistical_tests.csv")
    print(f"[F] statistics: {int(st.statistically_resolved.sum())}/{len(st)} resolved",
          flush=True)

    # ---------- provenance ----------
    prov = pd.DataFrame([{
        "script": "rf36_final_core.py", "dataset": "Shifts marine, real_data",
        "n_train_csv": int(len(parts['train']) + len(parts['validation'])
                           + len(parts['internal_test'])),
        "protocol": "chronological 60/20/20, no thinning",
        "physics_k": float(k), "bbm": "ExtraTrees (target = y)",
        "residual_learner": "Huber (target = y - k*F_phys)",
        "lambda_star": LAMBDA_STAR,
        "lambda_selected_on": "validation MAPE, 101-point grid",
        "seed": SEED, "bootstrap_block_h": BLOCK, "bootstrap_B": B_BOOT,
        "conformal_alpha": ALPHA_CONF, "git_commit": _git()}])
    write_csv(prov, OUT / "T00_provenance.csv")
    (OUT / "core_metadata.json").write_text(json.dumps({
        "k": float(k), "lambda_star": LAMBDA_STAR,
        "bbm": "ExtraTrees", "residual_learner": "Huber",
        "bbm_info": {kk: (vv if isinstance(vv, (int, float, bool, str)) else str(vv))
                     for kk, vv in info_b.items()},
        "resid_info": {kk: (vv if isinstance(vv, (int, float, bool, str)) else str(vv))
                       for kk, vv in info_r.items()},
        "D_phys_ratio_dev_in": float(ratio_in),
        "D_phys_ratio_dev_out": float(ratio_out),
        "git_commit": _git()}, indent=2), encoding="utf-8")
    print("[F] stage 1 complete", flush=True)
    return 0


def _guarded() -> int:
    import joblib
    with joblib.parallel_backend("threading", n_jobs=1):
        return main()


if __name__ == "__main__":
    sys.exit(_guarded())
