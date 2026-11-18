#!/usr/bin/env python3
"""P1.2: frozen, exploratory within-genotype growth vs U5 coverage.

Run from the repository root: uv run --active python src/P1/p1_2_within_genotype_growth.py
No retraining; reads verified P1.1 per-window exports.
"""
from __future__ import annotations

from pathlib import Path
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "experiments/P1/P1_1/tables"
OUT = ROOT / "experiments/P1/P1_2"
TABLE = OUT / "tables"
FIG = OUT / "figures"
REPORT = OUT / "reports"
SEED = 20261109
REPS = 10_000
HORIZONS = (24, 48)
GROWTH = "history_growth_abs_px_per_hour"
COLS = ["horizon_hours", "outer_fold", "inner_fold", "test_genotype",
        "method", "plant_id", "window_id", "leaf_id", "covered",
        GROWTH, "history_span_hours", "actual_target_offset_hours"]


def load_unique_windows(h):
    path = SOURCE / f"P1_1_U5_window_predictions_{h}h.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path, usecols=COLS)
    if frame.empty or set(frame.method.unique()) != {"U5_CQR_balanced"}:
        raise ValueError("Expected U5 data only")
    if set(frame.horizon_hours.unique()) != {h}:
        raise ValueError("Horizon mismatch")
    keys = ["test_genotype", "plant_id", "window_id"]
    if frame.duplicated(keys + ["inner_fold"]).any():
        raise ValueError("Duplicate window×assignment rows")
    counts = frame.groupby(keys, sort=False)["inner_fold"].agg(["size", "nunique"])
    if not ((counts["size"] == 4) & (counts["nunique"] == 4)).all():
        raise ValueError("Not exactly four assignments per window")
    if set(frame.inner_fold.unique()) != {1, 2, 3, 4}:
        raise ValueError("Unexpected assignment labels")
    static = ["leaf_id", GROWTH, "history_span_hours", "actual_target_offset_hours"]
    for c in static:
        if frame.groupby(keys, sort=False)[c].nunique(dropna=False).max() != 1:
            raise ValueError(f"Window metadata varies between assignments: {c}")
    unique = frame.groupby(keys, as_index=False, sort=False).agg(
        coverage=("covered", "mean"),
        growth=(GROWTH, "first"),
        span=("history_span_hours", "first"),
        offset=("actual_target_offset_hours", "first"),
        leaf_id=("leaf_id", "first"),
    )
    if not np.isfinite(unique[["coverage", "growth", "span", "offset"]].to_numpy()).all():
        raise ValueError("Nonfinite analysis data")
    if (unique.coverage.lt(0) | unique.coverage.gt(1)).any():
        raise ValueError("Invalid window coverage")
    expected = {24: 121431, 48: 107214}[h]
    if len(unique) != expected:
        raise ValueError(f"Expected {expected} unique windows; got {len(unique)}")
    if unique.test_genotype.nunique() != 21:
        raise ValueError("Expected 21 genotypes")
    print(f"{h}h | verified {len(frame):,} rows -> {len(unique):,} paired unique windows")
    return unique


def plant_weighted_context(d, genotype, label):
    """Plant-equal coverage and growth from filtered distinct test windows."""
    if d.empty:
        return {"test_genotype": genotype, "subset": label,
                "coverage": np.nan, "growth": np.nan,
                "plants": 0, "windows": 0}
    p = d.groupby("plant_id", sort=False).agg(
        coverage=("coverage", "mean"), growth=("growth", "mean"))
    return {"test_genotype": genotype, "subset": label,
            "coverage": float(p.coverage.mean()), "growth": float(p.growth.mean()),
            "plants": int(len(p)), "windows": int(len(d))}


def within_contrast(d, genotype, subset, cut_low, cut_high, span_lo, span_hi):
    dd = d[(d.growth <= cut_low) | (d.growth >= cut_high)].copy()
    dd["stratum"] = np.where(dd.growth <= cut_low, "low", "high")
    p = dd.groupby(["plant_id", "stratum"], sort=False).coverage.mean().unstack()
    if "low" not in p:
        p["low"] = np.nan
    if "high" not in p:
        p["high"] = np.nan
    both = p.dropna(subset=["low", "high"]).copy()
    both["difference"] = both["high"] - both["low"]
    result = {
        "test_genotype": genotype, "subset": subset,
        "growth_low_cut": float(cut_low), "growth_high_cut": float(cut_high),
        "span_lower": float(span_lo), "span_upper": float(span_hi),
        "retained_windows": int(len(d)), "retained_plants": int(d.plant_id.nunique()),
        "low_windows": int((dd.stratum == "low").sum()),
        "high_windows": int((dd.stratum == "high").sum()),
        "low_plants": int(p["low"].notna().sum()),
        "high_plants": int(p["high"].notna().sum()),
        "paired_plants": int(len(both)),
        "low_coverage_paired": float(both.low.mean()) if len(both) else np.nan,
        "high_coverage_paired": float(both.high.mean()) if len(both) else np.nan,
        "high_minus_low": float(both.difference.mean()) if len(both) >= 2 else np.nan,
        "status": "eligible" if len(both) >= 2 else "insufficient_paired_plants",
    }
    return result, both.reset_index().assign(test_genotype=genotype, subset=subset)


def bootstrap_median(v, rng):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 2:
        return np.nan, np.nan
    draws = np.median(v[rng.integers(0, len(v), size=(REPS, len(v)))], axis=1)
    return tuple(float(q) for q in np.quantile(draws, [0.025, 0.975]))


def spearman_ci(d, rng):
    d = d.dropna(subset=["coverage", "growth"])
    n = len(d)
    if n < 4:
        return np.nan, np.nan, np.nan, n
    from scipy.stats import spearmanr
    a = d.growth.to_numpy(float)
    b = d.coverage.to_numpy(float)
    rho = float(spearmanr(a, b).statistic)
    rs = []
    for _ in range(REPS):
        idx = rng.integers(0, n, n)
        if np.unique(a[idx]).size < 2 or np.unique(b[idx]).size < 2:
            continue
        r = spearmanr(a[idx], b[idx]).statistic
        if np.isfinite(r):
            rs.append(float(r))
    lo, hi = np.quantile(rs, [0.025, 0.975]) if len(rs) >= 100 else (np.nan, np.nan)
    return rho, float(lo), float(hi), n


def analyse():
    for path in (TABLE, FIG, REPORT):
        path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    rows, plant_rows, contexts, summaries, correlations = [], [], [], [], []
    for h in HORIZONS:
        df = load_unique_windows(h)
        for genotype in sorted(df.test_genotype.unique()):
            test = df[df.test_genotype == genotype]
            dev = df[df.test_genotype != genotype]
            cut_low, cut_high = np.quantile(dev.growth, [1 / 3, 2 / 3])
            span_lo, span_hi = np.quantile(dev.span, [0.25, 0.75])
            matched = test[
                test.span.between(span_lo, span_hi)
                & (test.offset - h).abs().le(1.5)
            ]
            for subset, part in (("full", test), ("span_matched", matched)):
                contrast, paired = within_contrast(
                    part, genotype, subset, cut_low, cut_high, span_lo, span_hi)
                rows.append({"horizon_hours": h, **contrast})
                if not paired.empty:
                    paired.insert(0, "horizon_hours", h)
                    plant_rows.append(paired)
                contexts.append({"horizon_hours": h, **plant_weighted_context(part, genotype, subset)})
        print(f"{h}h | all 21 genotype contrasts calculated", flush=True)
    contrasts = pd.DataFrame(rows)
    plant_contrasts = pd.concat(plant_rows, ignore_index=True) if plant_rows else pd.DataFrame()
    context = pd.DataFrame(contexts)
    for h in HORIZONS:
        for subset in ("full", "span_matched"):
            dd = contrasts[(contrasts.horizon_hours == h) & (contrasts.subset == subset)]
            eligible = dd[dd.status == "eligible"].high_minus_low.dropna().to_numpy(float)
            lo, hi = bootstrap_median(eligible, rng)
            from scipy.stats import wilcoxon
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p = (float(wilcoxon(eligible).pvalue)
                     if len(eligible) >= 3 and np.any(eligible != 0) else np.nan)
            cc = context[(context.horizon_hours == h) & (context.subset == subset)]
            rho, rlo, rhi, n_corr = spearman_ci(cc, rng)
            summaries.append({
                "horizon_hours": h, "subset": subset,
                "genotypes_eligible": len(eligible),
                "genotypes_total": len(dd),
                "paired_plants_total": int(dd.paired_plants.sum()),
                "retained_unique_windows": int(dd.retained_windows.sum()),
                "mean_high_minus_low": float(np.mean(eligible)) if len(eligible) else np.nan,
                "median_high_minus_low": float(np.median(eligible)) if len(eligible) else np.nan,
                "median_ci_lower": lo, "median_ci_upper": hi,
                "wilcoxon_p_exploratory": p,
                "spearman_rho_across_genotypes": rho,
                "spearman_ci_lower": rlo, "spearman_ci_upper": rhi,
                "genotypes_for_correlation": n_corr,
            })
            correlations.append({"horizon_hours": h, "subset": subset,
                                 "rho": rho, "ci_lower": rlo, "ci_upper": rhi,
                                 "n_genotypes": n_corr})
    result = pd.DataFrame(summaries)
    contrasts.to_csv(TABLE / "p1_within_genotype_growth.csv", index=False)
    plant_contrasts.to_csv(TABLE / "p1_within_genotype_plant_pairs.csv", index=False)
    context.to_csv(TABLE / "p1_growth_coverage_genotype_context.csv", index=False)
    result.to_csv(TABLE / "p1_growth_analysis_summary.csv", index=False)
    contrasts[contrasts.subset == "span_matched"].to_csv(
        TABLE / "p1_within_genotype_growth_span_matched.csv", index=False)
    for h in HORIZONS:
        for subset in ("full", "span_matched"):
            d = contrasts[(contrasts.horizon_hours == h) & (contrasts.subset == subset)]
            d = d.sort_values("test_genotype")
            fig, ax = plt.subplots(figsize=(11, 4.5))
            x = np.arange(len(d))
            vals = d.high_minus_low.to_numpy(float)
            ax.plot(x, vals, "o-")
            ax.axhline(0, linestyle="--", linewidth=1)
            ax.set_xticks(x, d.test_genotype, rotation=65, ha="right")
            ax.set_ylabel("High minus low historical-growth coverage")
            ax.set_title(f"U5 {h}h: within-genotype contrast ({subset.replace('_', ' ')})")
            fig.tight_layout()
            fig.savefig(FIG / f"p1_growth_contrast_{h}h_{subset}.png", dpi=180)
            plt.close(fig)
    lines = ["# P1.2 — Exploratory historical-growth and U5 coverage analysis", "",
             "## Analysis status", "", "Locked secondary analysis; prior inspection of held-out genotypes prevents confirmatory interpretation.", "",
             "## Method", "", "Development-only growth terciles per held-out genotype; high–low plant-paired coverage; 10,000 genotype bootstrap replicates; development-only history-span IQR and target-offset ±1.5h sensitivity.", "",
             "## Results", "", "```", result.to_string(index=False), "```", "",
             "## Interpretation constraints", "",
             "- Genotype sample size is 21; four inner assignments use the same test windows.",
             "- Bootstrap intervals and Wilcoxon values are post-selection exploratory summaries.",
             "- Growth strata and history span reflect acquisition/schedule as well as biology; no causal attribution.",
             "- Paired plant analysis includes only plants observed in both growth strata; this changes the within-genotype estimand.",
             "- Across-genotype Spearman analysis here uses plant-averaged growth and fixed U5 coverage, which may differ from earlier E4 feature definitions."]
    (REPORT / "p1_2_growth_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nP1.2 EXPLORATORY RESULTS")
    for _, r in result.iterrows():
        print(f"{int(r.horizon_hours)}h | {r.subset:12} | eligible={int(r.genotypes_eligible)}/21 | paired plants={int(r.paired_plants_total)} | windows={int(r.retained_unique_windows):,}")
        print(f"  high-low median={r.median_high_minus_low:.5f} [{r.median_ci_lower:.5f}, {r.median_ci_upper:.5f}] | exploratory Wilcoxon={r.wilcoxon_p_exploratory:.5g}")
        print(f"  across-genotype rho={r.spearman_rho_across_genotypes:.4f} [{r.spearman_ci_lower:.4f}, {r.spearman_ci_upper:.4f}]")
    print(f"Saved outputs: {OUT}")


if __name__ == "__main__":
    analyse()
