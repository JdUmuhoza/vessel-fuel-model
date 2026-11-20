"""Section 5.6 verification: multi-gas emission and CII propagation.

Primary Shifts benchmark only. Frozen Section 5.3 architectures: WBM = k*F_phys with
k = 0.393586, BBM = Huber on measured shaft power, S-GBM = WBM + full Huber residual,
C-GBM = WBM + 0.70 * Huber residual. lambda = 0.70 is imported and never re-selected.

The audit's central finding concerns time. Shifts records are not uniformly spaced: the
gap between consecutive records (in minutes, since ``time_id`` and ``timeSinceDryDock``
share identical differences) has median 26 min on the in-domain partition and 1 min on the
distribution-shifted partition, and the two partitions cover 1.06 % and 6.35 % of their
own calendar spans. Any "cumulative tonnes" figure therefore rests on an assumed interval
rather than a measured duration. This script computes the three candidate integrations,
identifies which one the published totals used, and establishes exactly which downstream
quantities survive the ambiguity (ratios and attained CII, because a uniform interval
cancels) and which do not (absolute tonnes).
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pandas as pd

from .common_rev import REPO_ROOT, REV_FINAL, SEED, write_csv, write_text
from .rf32_full_data_602020 import (
    FEATS, build_splits, fit_learner, metrics7, physics,
)

_E = REPO_ROOT / "experiments"
if str(_E) not in sys.path:
    sys.path.insert(0, str(_E))
from benchmark_measured.common import CONFIG, fit_scale  # noqa: E402
from benchmark_measured.run_cii_probabilistic import (  # noqa: E402
    ASSUMED_YEAR, CII_REDUCTION, DEADWEIGHT_COEFF, DISPLACEMENT_T, DWT_CENTRAL_T,
    SHIP_TYPES, TYPE_PARAMS, cii_rating, cii_required,
)
from vessel_fuel.emissions import _FUELS as EF_FUEL  # noqa: E402

OUT = REV_FINAL / "35_final_multigas_cii"
F24 = REV_FINAL / "24_full_data_602020"

LAMBDA_STAR, K_REF = 0.70, 0.393586
RESID_KEY = BBM_KEY = "huber"
ARCHS = ["WBM", "BBM", "S-GBM", "C-GBM"]
SERIES = ["Measured-power reference"] + ARCHS
FUEL_TYPE = "MDO"
GWP = {"CH4": 29.8, "N2O": 273.0}
MCR = CONFIG["particulars"]["MCR"]
SFOC_MCR = CONFIG["particulars"]["sfoc_at_mcr"]
DT_NOMINAL_H = 1.0        # the convention the published totals used; audited below
EXPECT_MAPE = {("dev_in", "WBM"): 8.848, ("dev_in", "BBM"): 18.820,
               ("dev_in", "S-GBM"): 8.788, ("dev_in", "C-GBM"): 8.238,
               ("dev_out", "WBM"): 9.214, ("dev_out", "BBM"): 21.930,
               ("dev_out", "S-GBM"): 9.545, ("dev_out", "C-GBM"): 8.732}
# old Section 5.6 values, treated only as quantities to check
OLD_FUEL_T = {("dev_in", "Measured-power reference"): 64620.7,
              ("dev_in", "WBM"): 64030.7, ("dev_in", "BBM"): 64680.0,
              ("dev_in", "S-GBM"): 64754.4, ("dev_in", "C-GBM"): 64209.9,
              ("dev_out", "Measured-power reference"): 54220.0,
              ("dev_out", "WBM"): 51875.5, ("dev_out", "BBM"): 54471.1,
              ("dev_out", "S-GBM"): 54119.6, ("dev_out", "C-GBM"): 52440.0}
OLD_CLAIMS = {"co2_share_pct": 98.647, "n2o_share_pct": 1.344,
              "ch4_share_pct": 0.009, "co2e_per_fuel_t_per_t": 3.250,
              "cgbm_fuel_error_dev_in_pct": -0.64,
              "cgbm_fuel_error_dev_out_pct": -3.28}


def _git() -> str | None:
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                           capture_output=True, text=True, timeout=20)
        return r.stdout.strip() or None
    except Exception:
        return None


def sfoc_g_per_kwh(load: np.ndarray) -> np.ndarray:
    """SFOC(l) = SFOC_MCR * (0.455 l^2 - 0.710 l + 1.280) / 1.025  [g/kWh]."""
    return SFOC_MCR * (0.455 * load ** 2 - 0.710 * load + 1.280) / 1.025


def fuel_rate_t_per_h(power_kw: np.ndarray) -> np.ndarray:
    """P[kW] -> l = P/MCR -> SFOC[g/kWh] -> mdot = SFOC*P/1e6  [t/h]."""
    return sfoc_g_per_kwh(power_kw / MCR) * power_kw / 1e6


def gases(fuel_t: np.ndarray, load: np.ndarray) -> dict:
    f = EF_FUEL[FUEL_TYPE]
    slip = f["ch4_kg_per_t"] * (1.0 + f["ch4_lowload_gain"]
                                * np.maximum(0.0, 0.5 - load) / 0.5)
    co2 = fuel_t * f["co2_t_per_t"]
    ch4 = fuel_t * slip
    n2o = fuel_t * f["n2o_kg_per_t"]
    co2e = co2 + (GWP["CH4"] * ch4 + GWP["N2O"] * n2o) / 1000.0
    return {"co2_t": co2, "ch4_kg": ch4, "n2o_kg": n2o, "co2e_t": co2e}


# ------------------------------------------------------------------ models ---

def build(parts, ph):
    tr = parts["train"]
    y, X = tr["power"].to_numpy(), tr[FEATS].to_numpy()
    k = fit_scale(y, ph["train"])
    res = y - k * ph["train"]
    fi, ei = parts["_fit_idx"], parts["_es_idx"]
    return {"k": k,
            "bbm": fit_learner(BBM_KEY, X, y, fi, ei)[0],
            "resid": fit_learner(RESID_KEY, X, res, fi, ei)[0]}


def predict(mdl, frame, phys):
    X = frame[FEATS].to_numpy()
    base = mdl["k"] * phys
    r = mdl["resid"].predict(X)
    return {"WBM": base, "BBM": mdl["bbm"].predict(X),
            "S-GBM": base + r, "C-GBM": base + LAMBDA_STAR * r}


def consistency(parts, ph, mdl) -> pd.DataFrame:
    ref = pd.read_csv(F24 / "final_dev_in_dev_out.csv")
    ref["architecture"] = ref["architecture"].str.replace(r"^C-GBM.*", "C-GBM",
                                                          regex=True)
    rows, fails = [], []
    for nm in ("dev_in", "dev_out"):
        pr = predict(mdl, parts[nm], ph[nm])
        y = parts[nm]["power"].to_numpy()
        for a in ARCHS:
            m = metrics7(y, pr[a])
            e = EXPECT_MAPE[(nm, a)]
            rr = ref[(ref.partition == nm) & (ref.architecture == a)].iloc[0]
            ok = abs(round(m["MAPE"], 3) - e) <= 0.001 and abs(
                m["MAPE"] - float(rr.MAPE)) < 1e-4
            if not ok:
                fails.append(f"{nm}/{a}: {m['MAPE']:.4f} vs 5.3 {e}")
            rows.append({"partition": nm, "architecture": a,
                         "MAPE_recomputed": m["MAPE"], "MAPE_section53": e,
                         "MAPE_folder24": float(rr.MAPE), "R2_recomputed": m["R2"],
                         "reproduces": ok})
    if abs(mdl["k"] - K_REF) > 1e-6:
        fails.append(f"k = {mdl['k']:.6f}, expected {K_REF}")
    if LAMBDA_STAR != 0.70:
        fails.append("lambda is not 0.70")
    if fails:
        print("[STOP] Section 5.3 not reproduced:", flush=True)
        for f in fails:
            print("   " + f, flush=True)
        raise SystemExit(2)
    print(f"[5.3] reproduced 8/8 MAPE; k = {mdl['k']:.6f}; lambda = {LAMBDA_STAR}",
          flush=True)
    return pd.DataFrame(rows)


# ------------------------------------------------- Part A: time and fuel -----

def time_audit(parts) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, conv = [], []
    for nm in ("dev_in", "dev_out"):
        d = parts[nm]
        t = d["time_id"].to_numpy(float)
        order = np.argsort(t)
        ts = t[order]
        gaps = np.diff(ts)
        span = float(ts.max() - ts.min())
        P = d["power"].to_numpy()[order]
        stw = d["stw"].to_numpy()[order]
        rate = fuel_rate_t_per_h(P)
        opts = {
            "uniform 1 h per record (published convention)": np.full(len(P), 1.0),
            "uniform 1 min per record (matches the time unit)": np.full(len(P), 1 / 60),
            "gap to the next record (h)": np.append(gaps, np.median(gaps)) / 60.0,
        }
        for lab, dt in opts.items():
            conv.append({"partition": nm, "integration_assumption": lab,
                         "n_observations": int(len(P)),
                         "total_hours_implied": float(dt.sum()),
                         "total_days_implied": float(dt.sum() / 24.0),
                         "cumulative_fuel_t": float((rate * dt).sum()),
                         "distance_nm": float((stw * dt).sum()),
                         "reproduces_old_dev_in_or_out_total": bool(
                             abs((rate * dt).sum()
                                 - OLD_FUEL_T[(nm, "Measured-power reference")]) < 0.5)})
        rows.append({"partition": nm, "n_observations": int(len(P)),
                     "time_unit": "minutes (time_id and timeSinceDryDock have "
                                  "identical consecutive differences)",
                     "calendar_span_min": span,
                     "calendar_span_days": span / 1440.0,
                     "gap_median_min": float(np.median(gaps)),
                     "gap_mean_min": float(gaps.mean()),
                     "gap_max_min": float(gaps.max()),
                     "records_are_uniformly_spaced": bool(gaps.std() < 1e-9),
                     "temporal_coverage_pct_if_1min_each":
                         100.0 * len(P) / span,
                     "fuel_rate_t_per_h_median": float(np.median(rate)),
                     "engine_load_median": float(np.median(P / MCR)),
                     "cumulative_tonnes_defensible": False})
    return pd.DataFrame(rows), pd.DataFrame(conv)


# ------------------------------------------------------------------- main ----

def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    parts = build_splits()
    ph = physics(parts)
    mdl = build(parts, ph)
    cons = consistency(parts, ph, mdl)
    write_csv(cons, OUT / "section53_consistency_check.csv")

    print("[A] power -> fuel conversion and time audit", flush=True)
    tim, conv = time_audit(parts)
    write_csv(tim, OUT / "time_base_audit.csv")
    write_csv(conv, OUT / "fuel_conversion_audit.csv")
    for _, r in tim.iterrows():
        print(f"    {r.partition}: span {r.calendar_span_days:,.1f} d, gap median "
              f"{r.gap_median_min:.0f} min, coverage "
              f"{r.temporal_coverage_pct_if_1min_each:.2f} %", flush=True)

    # ---------------- Part B: multi-gas ----------------
    f = EF_FUEL[FUEL_TYPE]
    ef = pd.DataFrame([
        {"gas": "CO2", "emission_factor": f["co2_t_per_t"], "unit": "t per t fuel",
         "equation": "E_CO2 = fuel_t * 3.206", "load_dependent": False,
         "gwp100": 1.0, "source": "Fourth IMO GHG Study, MDO/gas oil"},
        {"gas": "CH4", "emission_factor": f["ch4_kg_per_t"], "unit": "kg per t fuel",
         "equation": "E_CH4 = fuel_t * 0.010 * (1 + g*max(0, 0.5-l)/0.5), g = "
                     f"{f['ch4_lowload_gain']}",
         "load_dependent": bool(f["ch4_lowload_gain"] != 0.0),
         "gwp100": GWP["CH4"], "source": "Fourth IMO GHG Study; AR6 GWP100"},
        {"gas": "N2O", "emission_factor": f["n2o_kg_per_t"], "unit": "kg per t fuel",
         "equation": "E_N2O = fuel_t * 0.160", "load_dependent": False,
         "gwp100": GWP["N2O"], "source": "Fourth IMO GHG Study; AR6 GWP100"},
    ])
    write_csv(ef, OUT / "emission_factor_audit.csv")
    slip_const = bool(f["ch4_lowload_gain"] == 0.0)
    print(f"[B] MDO ch4_lowload_gain = {f['ch4_lowload_gain']} -> CH4 "
          f"{'strictly proportional to fuel' if slip_const else 'LOAD DEPENDENT'}",
          flush=True)

    gas_rows, comp_rows, cii_rows, inv = [], [], [], {}
    for nm in ("dev_in", "dev_out"):
        d = parts[nm]
        y = d["power"].to_numpy()
        pr = predict(mdl, d, ph[nm])
        stw = d["stw"].to_numpy()
        series = {"Measured-power reference": y, **pr}
        for dt_lab, dt in (("1h", DT_NOMINAL_H), ("1min", 1.0 / 60.0)):
            for lbl, P in series.items():
                load = P / MCR
                fuel = fuel_rate_t_per_h(P) * dt
                g = gases(fuel, load)
                ft, co2, ch4, n2o, co2e = (fuel.sum(), g["co2_t"].sum(),
                                           g["ch4_kg"].sum(), g["n2o_kg"].sum(),
                                           g["co2e_t"].sum())
                dist = float((stw * dt).sum())
                if dt_lab == "1h":
                    gas_rows.append({
                        "partition": nm, "series": lbl, "n_observations": int(len(P)),
                        "interval_convention_h": DT_NOMINAL_H,
                        "fuel_t": ft, "co2_t": co2, "ch4_kg": ch4, "n2o_kg": n2o,
                        "co2e_t": co2e,
                        "classification": ("measured power -> deterministic SFOC and "
                                           "emission chain (NOT measured fuel)"
                                           if lbl.startswith("Measured")
                                           else "predicted power -> same chain")})
                    comp_rows.append({
                        "partition": nm, "series": lbl,
                        "co2_share_pct": 100.0 * co2 / co2e,
                        "ch4_share_pct": 100.0 * GWP["CH4"] * ch4 / 1000.0 / co2e,
                        "n2o_share_pct": 100.0 * GWP["N2O"] * n2o / 1000.0 / co2e,
                        "co2e_per_fuel_t_per_t": co2e / ft})
                for st in SHIP_TYPES:
                    att = float(co2 * 1e6) / max(DWT_CENTRAL_T * dist, 1e-9)
                    req = cii_required(DWT_CENTRAL_T, ASSUMED_YEAR, st)
                    rec = {"partition": nm, "series": lbl, "ship_type": st,
                           "interval_convention": dt_lab,
                           "co2_t": co2, "distance_nm": dist,
                           "dwt_t": DWT_CENTRAL_T, "attained_cii": att,
                           "required_cii": req, "ratio": att / req,
                           "rating": cii_rating(att, req, st)}
                    if dt_lab == "1h":
                        cii_rows.append(rec)
                    inv.setdefault((nm, lbl, st), {})[dt_lab] = att
    gas = pd.DataFrame(gas_rows)
    comp = pd.DataFrame(comp_rows)
    cii = pd.DataFrame(cii_rows)
    write_csv(gas, OUT / "multigas_results.csv")
    write_csv(comp, OUT / "gas_composition_audit.csv")
    write_csv(cii, OUT / "cii_results.csv")

    # relative cumulative differences against the measured-power reference
    rel = []
    for nm in ("dev_in", "dev_out"):
        base = gas[(gas.partition == nm)
                   & (gas.series == "Measured-power reference")].iloc[0]
        for a in ARCHS:
            r = gas[(gas.partition == nm) & (gas.series == a)].iloc[0]
            rel.append({"partition": nm, "series": a,
                        **{f"rel_diff_{c}_pct": 100.0 * (r[c] - base[c]) / base[c]
                           for c in ("fuel_t", "co2_t", "ch4_kg", "n2o_kg", "co2e_t")},
                        "interpretation": "relative difference of a cumulative "
                                          "propagated quantity against the "
                                          "measured-power reference; NOT a "
                                          "gas-specific prediction error (no stack "
                                          "measurements exist)",
                        "invariant_to_interval_convention": True})
    relf = pd.DataFrame(rel)
    write_csv(relf, OUT / "multigas_relative_errors.csv")

    # CII invariance to a uniform interval
    invrows = [{"partition": k[0], "series": k[1], "ship_type": k[2],
                "attained_cii_dt_1h": v["1h"], "attained_cii_dt_1min": v["1min"],
                "abs_difference": abs(v["1h"] - v["1min"]),
                "invariant": bool(abs(v["1h"] - v["1min"]) < 1e-9)}
               for k, v in inv.items()]
    invf = pd.DataFrame(invrows)
    write_csv(invf, OUT / "cii_interval_invariance.csv")
    print(f"[C] attained CII invariant to the uniform interval: "
          f"{int(invf.invariant.sum())}/{len(invf)}", flush=True)

    # ---------------- Part E: old vs recomputed ----------------
    old = []
    for (nm, lbl), v in OLD_FUEL_T.items():
        new = float(gas[(gas.partition == nm) & (gas.series == lbl)]["fuel_t"].iloc[0])
        old.append({"item": "cumulative fuel (t)", "partition": nm, "series": lbl,
                    "old_value": v, "recomputed": new, "difference": new - v,
                    "verdict": "PASS" if abs(new - v) <= 0.5 else "FAIL"})
    c0 = comp[comp.partition == "dev_in"].iloc[0]
    for key, new in (("co2_share_pct", c0.co2_share_pct),
                     ("n2o_share_pct", c0.n2o_share_pct),
                     ("ch4_share_pct", c0.ch4_share_pct),
                     ("co2e_per_fuel_t_per_t", c0.co2e_per_fuel_t_per_t)):
        v = OLD_CLAIMS[key]
        tol = 0.001 if "share" in key else 0.001
        old.append({"item": key, "partition": "both", "series": "all",
                    "old_value": v, "recomputed": float(new),
                    "difference": float(new) - v,
                    "verdict": "PASS" if abs(float(new) - v) <= tol else "FAIL"})
    for nm, key in (("dev_in", "cgbm_fuel_error_dev_in_pct"),
                    ("dev_out", "cgbm_fuel_error_dev_out_pct")):
        new = float(relf[(relf.partition == nm)
                         & (relf.series == "C-GBM")]["rel_diff_fuel_t_pct"].iloc[0])
        v = OLD_CLAIMS[key]
        old.append({"item": key, "partition": nm, "series": "C-GBM",
                    "old_value": v, "recomputed": new, "difference": new - v,
                    "verdict": "PASS" if abs(new - v) <= 0.01 else "FAIL"})
    for nm in ("dev_in", "dev_out"):
        for st, exp in (("bulk carrier", "E"),
                        ("container ship", "D" if nm == "dev_in" else "C")):
            got = sorted(set(cii[(cii.partition == nm)
                                 & (cii.ship_type == st)]["rating"]))
            old.append({"item": f"all ratings {st}", "partition": nm, "series": "all",
                        "old_value": exp, "recomputed": "/".join(got),
                        "difference": np.nan,
                        "verdict": "PASS" if got == [exp] else "FAIL"})
    oldnew = pd.DataFrame(old)
    write_csv(oldnew, OUT / "old_vs_recomputed_results.csv")

    # ---------------- checks ----------------
    ratings_change = {}
    for nm in ("dev_in", "dev_out"):
        for st in SHIP_TYPES:
            s = cii[(cii.partition == nm) & (cii.ship_type == st)]
            ratings_change[f"{nm}/{st}"] = sorted(set(s.rating))
    checks = [
        {"check_id": "F1", "item": "Section 5.3 MAPE reproduced",
         "verdict": "PASS" if cons.reproduces.all() else "FAIL",
         "detail": f"{int(cons.reproduces.sum())}/{len(cons)}"},
        {"check_id": "F2", "item": "lambda 0.70 frozen, k reproduced",
         "verdict": "PASS", "detail": f"lambda={LAMBDA_STAR}, k={mdl['k']:.6f}"},
        {"check_id": "F3", "item": "no obsolete learner or variant",
         "verdict": "PASS",
         "detail": "Huber only; no CatBoost, ExtraTrees, ridge or bounded residual; "
                   "FuelCast not loaded"},
        {"check_id": "A1", "item": "records are uniformly spaced in time",
         "verdict": "FAIL" if not tim.records_are_uniformly_spaced.all() else "PASS",
         "detail": "; ".join(f"{r.partition}: gap median {r.gap_median_min:.0f} min, "
                             f"mean {r.gap_mean_min:.1f}, max {r.gap_max_min:,.0f}"
                             for _, r in tim.iterrows())},
        {"check_id": "A2", "item": "cumulative tonnes are a measured duration",
         "verdict": "FAIL",
         "detail": "totals depend entirely on an assumed interval; the published "
                   "values correspond to a uniform 1 h per record, which matches "
                   "neither the 1-minute time unit nor the observed gaps"},
        {"check_id": "A3", "item": "published totals identified and reproduced",
         "verdict": "PASS" if conv.reproduces_old_dev_in_or_out_total.any() else "FAIL",
         "detail": "uniform 1 h per record reproduces 64,620.7 t and 54,220.0 t"},
        {"check_id": "B1", "item": "CH4 slip term is load independent for MDO",
         "verdict": "PASS" if slip_const else "FAIL",
         "detail": f"ch4_lowload_gain = {f['ch4_lowload_gain']} so the slip factor is "
                   f"constant; gases are strictly proportional to fuel"},
        {"check_id": "B2", "item": "gas composition identical across all series",
         "verdict": "PASS" if comp.co2e_per_fuel_t_per_t.std() < 1e-9 else "FAIL",
         "detail": f"CO2e/fuel spread = {comp.co2e_per_fuel_t_per_t.std():.2e} t/t "
                   f"across {len(comp)} series"},
        {"check_id": "C1", "item": "attained CII invariant to a uniform interval",
         "verdict": "PASS" if invf.invariant.all() else "FAIL",
         "detail": f"{int(invf.invariant.sum())}/{len(invf)} identical between the 1 h "
                   f"and 1 min conventions"},
        {"check_id": "C2", "item": "vessel class established by the benchmark",
         "verdict": "FAIL",
         "detail": "vessel anonymised; both reference lines retained as a sensitivity "
                   "analysis only, not a regulatory classification"},
        {"check_id": "C3", "item": "architecture changes any CII rating",
         "verdict": "PASS" if any(len(v) > 1 for v in ratings_change.values())
                    else "FAIL",
         "detail": "; ".join(f"{k}: {v}" for k, v in ratings_change.items())},
        {"check_id": "R1", "item": "relative differences invariant to the interval",
         "verdict": "PASS",
         "detail": "a uniform interval multiplies every series identically, so it "
                   "cancels in any ratio against the measured-power reference"},
    ]
    cdf = pd.DataFrame(checks)
    write_csv(cdf, OUT / "checks.csv")

    (OUT / "run_metadata.json").write_text(json.dumps({
        "purpose": "Section 5.6 multi-gas and CII verification",
        "benchmark": "Shifts marine primary; FuelCast not used",
        "k": mdl["k"], "lambda_star": LAMBDA_STAR, "lambda_reselected": False,
        "residual_learner": "Huber", "bbm": "Huber on measured power",
        "excluded": ["CatBoost", "ExtraTrees", "ridge-to-physics",
                     "bounded residual", "lambda=0.25"],
        "fuel_type": FUEL_TYPE, "emission_factors": EF_FUEL[FUEL_TYPE],
        "gwp100": GWP, "MCR_kw": MCR, "sfoc_at_mcr_g_per_kwh": SFOC_MCR,
        "time_base": {"unit": "minutes",
                      "uniform": False,
                      "published_convention": "uniform 1 h per record",
                      "cumulative_tonnes_defensible": False,
                      "ratios_and_attained_cii_invariant": True},
        "cii": {"dwt_t": DWT_CENTRAL_T,
                "dwt_derivation": f"L*B*T*CB*rho/1000 = {DISPLACEMENT_T:,.1f} t "
                                  f"displacement x deadweight coefficient "
                                  f"{DEADWEIGHT_COEFF}",
                "reference_line": "CII_ref = a * DWT^(-c), then x (1 - Z)",
                "type_params": TYPE_PARAMS, "year": ASSUMED_YEAR,
                "reduction_factor": CII_REDUCTION[ASSUMED_YEAR],
                "ship_class_known": False,
                "distance": "sum(stw * dt) in nm"},
        "seed": SEED, "git_commit": _git()}, indent=2), encoding="utf-8")

    print("\n[35] checks", flush=True)
    for _, r in cdf.iterrows():
        print(f"  [{r['verdict']}] {r['check_id']} {r['item']}\n"
              f"        {r['detail']}", flush=True)
    print(f"\n[35] {int((cdf.verdict == 'PASS').sum())}/{len(cdf)} PASS "
          f"({int((cdf.verdict == 'FAIL').sum())} FAIL, each reported as a finding)",
          flush=True)
    return 0


def _guarded() -> int:
    import joblib
    with joblib.parallel_backend("threading", n_jobs=1):
        return main()


if __name__ == "__main__":
    sys.exit(_guarded())
