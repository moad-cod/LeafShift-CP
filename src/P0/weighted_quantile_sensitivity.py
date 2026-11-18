#!/usr/bin/env python3
"""P0.4b: hierarchical-weighted quantile sensitivity under LOGO.

Run from project root:
    uv run --active python src/P0/run_weighted_quantile_sensitivity.py

Uses original E3 code for model/data loading, compares original weighted 90th
percentile with an augmented +infinity test-mass convention. No claim of
finite-sample coverage under leave-one-genotype-out shift is implied.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "E3"))
import run_normalized_conformal as base  # noqa: E402
import run_cqr_baselines as cqr  # noqa: E402

OUT = ROOT / "experiments" / "P0" / "P0_4b"
OUT.mkdir(parents=True, exist_ok=True)
FOLDS = getattr(base, "FOLDS", ROOT / "configs" / "E1_1_genotype_folds.json")
NOMINAL = float(getattr(base, "NOMINAL", 0.9))
HORIZONS = (24, 48)
CATEGORIES = ("small_lt500", "medium_500to1999", "large_ge2000")


def safe_weights(genotypes, plants):
    """Reuse project weighting, with validation of globally unique plant IDs."""
    g = np.asarray(genotypes).astype(str)
    p = np.asarray(plants).astype(str)
    if len(g) == 0:
        raise ValueError("Empty calibration subset")
    relation = pd.DataFrame({"g": g, "p": p}).groupby("p")["g"].nunique()
    if (relation > 1).any():
        raise ValueError("Plant IDs appear in multiple genotypes. Existing E3 "
                         "weighting uses plant ID alone; resolve this collision first.")
    w = base.balanced_weights(g, p)
    if not np.all(np.isfinite(w)) or np.any(w <= 0):
        raise ValueError("Invalid hierarchical calibration weights")
    return w / w.sum()


def masses(genotypes, plants, weights):
    """Genotype and plant-level masses induced by window-level weights."""
    df = pd.DataFrame({
        "g": np.asarray(genotypes).astype(str),
        "p": np.asarray(plants).astype(str),
        "w": np.asarray(weights, dtype=float),
    })
    plant = df.groupby(["g", "p"], as_index=False)["w"].sum()
    geno = df.groupby("g", as_index=False)["w"].sum()
    plant_masses = plant["w"].to_numpy()
    geno_masses = geno["w"].to_numpy()
    return {
        "n_windows": int(len(weights)),
        "n_plants": int(len(plant_masses)),
        "n_genotypes": int(len(geno_masses)),
        "ess_windows": float(1 / np.sum(np.asarray(weights)**2)),
        "ess_plants": float(1 / np.sum(plant_masses**2)),
        "ess_genotypes": float(1 / np.sum(geno_masses**2)),
        "max_plant_mass": float(plant_masses.max()),
        "max_genotype_mass": float(geno_masses.max()),
    }


def weighted_with_infinity(scores, weights, test_mass):
    """Weighted (1-alpha) quantile of scores + hypothetical +inf test atom.

    Calibration masses sum to 1 and test mass is test_mass, giving cumulative
    calibration mass <= 1/(1+test_mass). Returns +inf when it cannot reach
    NOMINAL. This is a *sensitivity convention*, not a LOGO validity theorem.
    """
    s = np.asarray(scores, dtype=float)
    w = np.asarray(weights, dtype=float)
    if len(s) == 0 or len(s) != len(w):
        raise ValueError("Mismatched or empty score/weight array")
    if not np.all(np.isfinite(s)):
        raise ValueError("Non-finite calibration scores")
    if not np.isclose(w.sum(), 1, atol=1e-8):
        raise ValueError("Calibration weights must sum to 1")
    if not (test_mass > 0 and np.isfinite(test_mass)):
        raise ValueError("Test mass must be finite and positive")
    # Equivalent to a quantile of (w, +inf atom) after renormalization.
    target = NOMINAL * (1 + test_mass)
    if target > 1 + 1e-12:
        return math.inf
    order = np.argsort(s, kind="mergesort")
    cum = np.cumsum(w[order])
    idx = int(np.searchsorted(cum, target, side="left"))
    return float(s[order[min(idx, len(order) - 1)]]) if idx < len(s) else math.inf


def metrics(y, lower, upper, plants):
    """Equal-plant macro metrics; +inf endpoints explicitly marked vacuous."""
    if np.isinf(upper).any() or np.isinf(lower).any():
        return dict(coverage=1.0, width=math.inf, interval_score=math.inf,
                    test_plants=int(len(np.unique(plants))),
                    test_windows=int(len(y)), vacuous=1)
    y, lower, upper, plants = map(np.asarray, (y, lower, upper, plants))
    cover = (y >= lower) & (y <= upper)
    width = upper - lower
    score = width + 2 / (1 - NOMINAL) * (np.maximum(lower-y, 0) + np.maximum(y-upper, 0))
    arr = np.array([(cover[plants == p].mean(), width[plants == p].mean(),
                     score[plants == p].mean()) for p in np.unique(plants)])
    return dict(coverage=float(arr[:, 0].mean()), width=float(arr[:, 1].mean()),
                interval_score=float(arr[:, 2].mean()), test_plants=len(arr),
                test_windows=len(y), vacuous=0)


def score_interval(point, scale, threshold, method, q_lo=None, q_hi=None):
    """Match E3 lower clipping and nonnegative CQR expansion convention."""
    n = len(point)
    if math.isinf(threshold):
        return np.zeros(n), np.full(n, math.inf)
    if method == "U3_normalized_balanced":
        half = threshold * scale
        return np.maximum(0, point - half), point + half
    if method in ("U1_balanced", "U7_Mondrian_balanced"):
        return np.maximum(0, point - threshold), point + threshold
    if method == "U5_CQR_balanced":
        q = max(0., threshold)
        lo = np.maximum(0, q_lo-q)
        hi = np.maximum(lo, q_hi+q)
        return lo, hi
    raise KeyError(method)


def compare_method(common, method, scores, cal_g, cal_p, test_y,
                   test_p, point, test_scale, q_lo=None, q_hi=None,
                   test_mask=None, group="all"):
    w = safe_weights(cal_g, cal_p)
    support = masses(cal_g, cal_p, w)
    original = float(base.balanced_quantile(scores, w))
    conventions = {
        "original_weighted": original,
        "plant_max_mass_plus_infinity": weighted_with_infinity(
            scores, w, support["max_plant_mass"]),
        "plant_ess_mass_plus_infinity": weighted_with_infinity(
            scores, w, 1 / support["ess_plants"]),
        "genotype_mass_plus_infinity": weighted_with_infinity(
            scores, w, support["max_genotype_mass"]),
    }
    rows = []
    for variant, q in conventions.items():
        lo, hi = score_interval(point, test_scale, q, method, q_lo, q_hi)
        row = dict(common, method=method, category=group, variant=variant,
                   calibration_threshold=float(q), **support,
                   hypothetical_test_mass=(0.0 if variant == "original_weighted" else
                     support["max_plant_mass"] if variant == "plant_max_mass_plus_infinity" else
                     1/support["ess_plants"] if variant == "plant_ess_mass_plus_infinity" else
                     support["max_genotype_mass"]),
                   **metrics(test_y, lo, hi, test_p))
        rows.append(row)
    return rows


def evaluate(d, outer, inner, horizon):
    fitting = set(inner["train_genotypes"])
    calibration = set(inner["validation_genotypes"])
    heldout = outer["test_genotype"]
    if len(fitting) != 15 or len(calibration) != 5 or fitting & calibration or heldout in fitting | calibration:
        raise ValueError("Genotype split invalid")
    g = d["g"]
    fit, cal, test = np.isin(g, list(fitting)), np.isin(g, list(calibration)), g == heldout
    if set(g[fit]) != fitting or set(g[cal]) != calibration or set(g[test]) != {heldout}:
        raise ValueError("Genotype missing from source")
    wf = safe_weights(g[fit], d["p"][fit])
    point_model = HistGradientBoostingRegressor(**base.PARAMS)
    point_model.fit(d["X"][fit], d["y"][fit], sample_weight=wf * len(wf))
    pred_cal = np.maximum(point_model.predict(d["X"][cal]), 0.0)
    pred_test = np.maximum(point_model.predict(d["X"][test]), 0.0)
    residual = np.abs(d["y"][cal] - pred_cal)
    scale_cal, scale_test = d["scale"][cal], d["scale"][test]
    if np.any(scale_cal <= 0) or np.any(scale_test <= 0):
        raise ValueError("Scale must be strictly positive")
    base_fields = dict(horizon_hours=horizon, outer_fold=int(outer["outer_fold"]),
                       inner_fold=int(inner["inner_fold"]), test_genotype=heldout)
    rows = []
    shared = dict(cal_g=g[cal], cal_p=d["p"][cal], test_y=d["y"][test],
                  test_p=d["p"][test], point=pred_test, test_scale=scale_test)
    rows += compare_method(base_fields, "U1_balanced", residual, **shared)
    rows += compare_method(base_fields, "U3_normalized_balanced", residual / scale_cal, **shared)

    # Fit CQR once and reuse scores for original and all sensitivity variants.
    qlo_model = cqr.fit_quantile_model(d["X"][fit], d["y"][fit], wf, .05)
    qhi_model = cqr.fit_quantile_model(d["X"][fit], d["y"][fit], wf, .95)
    ca = qlo_model.predict(d["X"][cal]); cb = qhi_model.predict(d["X"][cal])
    cal_lo, cal_hi = np.minimum(ca, cb), np.maximum(ca, cb)
    cqr_score = np.maximum(cal_lo-d["y"][cal], d["y"][cal]-cal_hi)
    ta = qlo_model.predict(d["X"][test]); tb = qhi_model.predict(d["X"][test])
    tlo, thi = np.minimum(ta, tb), np.maximum(ta, tb)
    rows += compare_method(base_fields, "U5_CQR_balanced", cqr_score, **shared,
                           q_lo=tlo, q_hi=thi)

    for cat in CATEGORIES:
        c_mask = cal & (d["size"] == cat)
        t_mask = test & (d["size"] == cat)
        if not c_mask.any() or not t_mask.any():
            continue
        if len(np.unique(g[c_mask])) < 2 or len(np.unique(d["p"][c_mask])) < 2:
            raise ValueError(f"Insufficient Mondrian calibration support: {cat}")
        rows += compare_method(base_fields, "U7_Mondrian_balanced",
                               residual[d["size"][cal] == cat],
                               cal_g=g[c_mask], cal_p=d["p"][c_mask],
                               test_y=d["y"][t_mask], test_p=d["p"][t_mask],
                               point=pred_test[d["size"][test] == cat],
                               test_scale=scale_test[d["size"][test] == cat],
                               group=cat)
    return rows


def summarize(df):
    by_geno = df.groupby(["horizon_hours", "method", "category", "variant", "test_genotype"],
                         as_index=False)[["coverage", "width", "interval_score", "vacuous"]].mean()
    summary = by_geno.groupby(["horizon_hours", "method", "category", "variant"], as_index=False).agg(
        macro_coverage=("coverage", "mean"), macro_width=("width", "mean"),
        macro_interval_score=("interval_score", "mean"),
        n_genotypes=("test_genotype", "nunique"),
        vacuous_genotype_assignment_fraction=("vacuous", "mean"))
    # Averages of a mix of finite and infinite intervals are correctly infinite.
    return by_geno, summary


def report(summary, df):
    lines = ["# P0.4b — Hierarchical Weighted-Quantile Sensitivity", "",
             "All variants reuse the same model and calibration scores in a given split.",
             "Neither plant-mass nor genotype-mass correction establishes LOGO coverage.",
             "An infinite threshold is reported as a vacuous interval, not silently capped.", "",
             "## Interpretation of hypothetical test masses", "",
             "- `original_weighted`: empirical weighted nominal quantile.",
             "- `plant_max_mass_plus_infinity`: add +infinity with the largest represented calibration plant mass.",
             "- `plant_ess_mass_plus_infinity`: add +infinity with mass 1 / plant ESS.",
             "- `genotype_mass_plus_infinity`: add +infinity with largest calibration genotype mass.",
             "", "## Why genotype-level correction can be vacuous", "",
             "For five equally weighted calibration genotypes, hypothetical test mass = 1/5.",
             "Maximum finite cumulative mass = 1/(1+1/5) = 0.8333 < 0.9.",
             "Thus every genotype-mass corrected 90% threshold is +infinity.", "",
             "## Summary (full methods)", "",
             "| Horizon | Method | Variant | Macro coverage | Macro width | Interval score | Vacuous fraction |",
             "|---|---|---|---:|---:|---:|---:|"]
    for _, r in summary.iterrows():
        lines.append(f"| {int(r.horizon_hours)}h | {r.method} ({r.category}) | {r.variant} | "
                     f"{r.macro_coverage:.4f} | {r.macro_width:.2f} | "
                     f"{r.macro_interval_score:.2f} | {r.vacuous_genotype_assignment_fraction:.3f} |")
    lines += ["", "## Caveats", "",
              "1. Calibration windows within a leaf/plant are dependent; weight-based sample size is not independent sample count.",
              "2. The five calibration genotypes need not be exchangeable with the unseen target genotype.",
              "3. A hypothesized test-unit mass is not a density ratio and does not reproduce a general weighted-CP theorem.",
              "4. Held-out genotypes recur across four fitting/calibration assignments; do not treat them as 84 independent test groups.",
              "5. The study remains exploratory because the same outer genotypes informed preceding research decisions.",
              "6. Mondrian category results are evaluated on the category's test subset, not stitched into full-genotype U7 metrics.",
              "", "## Decision guidance", "",
              "Compare changes in coverage, width, and interval score within each 21-genotype paired design; do not favor +infinity methods based on coverage alone.",
              "Retain the original U1/U3/U5/U7 results unchanged while evaluating robustness."]
    (OUT / "P0_4b_report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")


def main():
    print("P0.4b — WEIGHTED-QUANTILE SENSITIVITY")
    print("Output:", OUT, flush=True)
    with Path(FOLDS).open(encoding="utf-8") as f:
        folds = json.load(f)["folds"]
    if len(folds) != 21:
        raise ValueError("Expected 21 outer folds")
    data = base.load_data()
    rows = []
    for horizon in HORIZONS:
        for j, outer in enumerate(folds, 1):
            for inner in outer["inner_folds"]:
                rows.extend(evaluate(data[horizon], outer, inner, horizon))
            print(f"{horizon}h | completed {j:02d}/21 | {outer['test_genotype']}", flush=True)
            pd.DataFrame(rows).to_csv(OUT / "P0_4b_checkpoint.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "P0_4b_assignment_metrics.csv", index=False)
    by_geno, summary = summarize(df)
    by_geno.to_csv(OUT / "P0_4b_genotype_metrics.csv", index=False)
    summary.to_csv(OUT / "P0_4b_summary.csv", index=False)
    report(summary, df)
    print("\nSUMMARY — U5 balanced CQR (all leaf sizes):")
    print(summary[(summary.method == "U5_CQR_balanced") & (summary.category == "all")].to_string(index=False))
    print("\nCompleted. Results saved at", OUT)


if __name__ == "__main__":
    main()
