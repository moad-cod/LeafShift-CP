
#!/usr/bin/env python3
"""
P0.4a — CQR threshold-clipping sensitivity.

Compares:
- U4 pooled CQR: nonnegative vs signed thresholds
- U5 balanced CQR: nonnegative vs signed thresholds

Uses the same trained models, calibration data, and
held-out genotype test data for each paired comparison.

No E3 outputs are overwritten.
"""

import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "E3"))

import run_cqr_baselines as cqr


OUT = ROOT / "experiments" / "P0" / "P0_4a"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

for directory in (TABLES, FIGURES, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

ALPHA = 0.10
HORIZONS = (24, 48)

FOLD_FILE = ROOT / "configs" / "E1_1_genotype_folds.json"


def save_csv(path, rows):
    if not rows:
        return

    pd.DataFrame(rows).to_csv(path, index=False)


def plant_metrics(y, lower, upper, plants):
    """
    Evaluate using equal weight per test plant,
    rather than treating all windows as independent.

    Returns plant-averaged coverage, width, interval score.
    """
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    plants = np.asarray(plants)

    covered = ((y >= lower) & (y <= upper)).astype(float)
    width = upper - lower

    score = (
        width
        + (2.0 / ALPHA) * np.maximum(lower - y, 0.0)
        + (2.0 / ALPHA) * np.maximum(y - upper, 0.0)
    )

    values = []
    for plant in np.unique(plants):
        mask = plants == plant
        values.append((
            covered[mask].mean(),
            width[mask].mean(),
            score[mask].mean(),
        ))

    array = np.asarray(values, dtype=float)

    return {
        "coverage": float(array[:, 0].mean()),
        "width": float(array[:, 1].mean()),
        "interval_score": float(array[:, 2].mean()),
        "test_plants": len(values),
        "test_windows": len(y),
    }


def construct_intervals(lo, hi, threshold):
    """
    Follow E3.2 endpoint conventions:
    - Sort preliminary quantiles before calibration
    - Nonnegative area lower bound
    - Upper bound cannot fall below lower bound

    When signed thresholds are very negative, report
    endpoint collapse separately.
    """
    raw_lower = lo - threshold
    raw_upper = hi + threshold

    lower = np.maximum(0.0, raw_lower)
    upper = np.maximum(lower, raw_upper)

    collapsed = raw_upper < lower

    return lower, upper, int(np.sum(collapsed))


def evaluate_split(d, outer, inner, horizon):
    fit_genotypes = set(inner["train_genotypes"])
    cal_genotypes = set(inner["validation_genotypes"])
    test_genotype = outer["test_genotype"]

    if len(fit_genotypes) != 15:
        raise ValueError("Expected 15 fitting genotypes")

    if len(cal_genotypes) != 5:
        raise ValueError("Expected 5 calibration genotypes")

    if (
        fit_genotypes & cal_genotypes
        or test_genotype in fit_genotypes
        or test_genotype in cal_genotypes
    ):
        raise ValueError("Genotype leakage")

    g = d["g"]
    fit = np.isin(g, list(fit_genotypes))
    cal = np.isin(g, list(cal_genotypes))
    test = g == test_genotype

    if set(g[fit]) != fit_genotypes:
        raise ValueError("Missing fitting genotype")

    if set(g[cal]) != cal_genotypes:
        raise ValueError("Missing calibration genotype")

    if set(g[test]) != {test_genotype}:
        raise ValueError("Missing test genotype")

    weights_fit = cqr.balanced_weights(
        d["g"][fit], d["p"][fit]
    )

    # Fit once per split, then reuse for both variants.
    model_lo = cqr.fit_quantile_model(
        d["X"][fit],
        d["y"][fit],
        weights_fit,
        0.05,
    )

    model_hi = cqr.fit_quantile_model(
        d["X"][fit],
        d["y"][fit],
        weights_fit,
        0.95,
    )

    cal_lo_raw = model_lo.predict(d["X"][cal])
    cal_hi_raw = model_hi.predict(d["X"][cal])

    cal_lo = np.minimum(cal_lo_raw, cal_hi_raw)
    cal_hi = np.maximum(cal_lo_raw, cal_hi_raw)

    # Canonical signed CQR conformity score.
    scores = np.maximum(
        cal_lo - d["y"][cal],
        d["y"][cal] - cal_hi,
    )

    weights_cal = cqr.balanced_weights(
        d["g"][cal],
        d["p"][cal],
    )

    test_lo_raw = model_lo.predict(d["X"][test])
    test_hi_raw = model_hi.predict(d["X"][test])

    test_lo = np.minimum(test_lo_raw, test_hi_raw)
    test_hi = np.maximum(test_lo_raw, test_hi_raw)

    y_test = d["y"][test]
    plants_test = d["p"][test]

    quantile_crossing_cal = int(
        np.sum(cal_lo_raw > cal_hi_raw)
    )
    quantile_crossing_test = int(
        np.sum(test_lo_raw > test_hi_raw)
    )

    common = {
        "horizon_hours": horizon,
        "outer_fold": int(outer["outer_fold"]),
        "inner_fold": int(inner["inner_fold"]),
        "test_genotype": test_genotype,
    }

    result_rows = []
    diagnostic_rows = []

    methods = {
        "U4_CQR_pooled": cqr.pooled_quantile(scores),
        "U5_CQR_balanced": cqr.weighted_quantile(
            scores, weights_cal
        ),
    }

    for method, q_signed in methods.items():
        q_clipped = max(0.0, float(q_signed))

        diagnostic_rows.append({
            **common,
            "method": method,
            "threshold_signed": float(q_signed),
            "threshold_clipped": q_clipped,
            "negative_threshold": int(q_signed < 0),
            "threshold_difference": q_clipped - q_signed,
            "calibration_windows": int(cal.sum()),
            "calibration_plants": len(
                np.unique(d["p"][cal])
            ),
            "negative_score_fraction": float(
                np.mean(scores < 0)
            ),
            "crossed_calibration_quantiles": (
                quantile_crossing_cal
            ),
            "crossed_test_quantiles": (
                quantile_crossing_test
            ),
        })

        for variant, threshold in (
            ("original_nonnegative", q_clipped),
            ("unrestricted_signed", float(q_signed)),
        ):
            lower, upper, collapsed = construct_intervals(
                test_lo, test_hi, threshold
            )

            metrics = plant_metrics(
                y_test, lower, upper, plants_test
            )

            result_rows.append({
                **common,
                "method": method,
                "variant": variant,
                "threshold": threshold,
                "collapsed_intervals": collapsed,
                **metrics,
            })

    return result_rows, diagnostic_rows


def summarize(results, diagnostics):
    keys = [
        "horizon_hours",
        "outer_fold",
        "inner_fold",
        "test_genotype",
        "method",
    ]

    pairs = results.pivot(
        index=keys,
        columns="variant",
        values=["coverage", "width", "interval_score"],
    )

    pairs.columns = [
        f"{metric}__{variant}"
        for metric, variant in pairs.columns
    ]

    pairs = pairs.reset_index()

    for metric in ("coverage", "width", "interval_score"):
        pairs[f"delta_{metric}"] = (
            pairs[f"{metric}__unrestricted_signed"]
            - pairs[f"{metric}__original_nonnegative"]
        )

    genotype_pairs = (
        pairs.groupby(
            ["horizon_hours", "test_genotype", "method"],
            as_index=False,
        )[
            ["delta_coverage", "delta_width",
             "delta_interval_score"]
        ]
        .mean()
    )

    summary_rows = []

    for (horizon, method), group in results.groupby(
        ["horizon_hours", "method"]
    ):
        diag = diagnostics[
            (diagnostics["horizon_hours"] == horizon)
            & (diagnostics["method"] == method)
        ]

        row = {
            "horizon_hours": horizon,
            "method": method,
            "assignments": len(diag),
            "negative_threshold_assignments": int(
                diag["negative_threshold"].sum()
            ),
            "negative_threshold_fraction": float(
                diag["negative_threshold"].mean()
            ),
        }

        for variant in (
            "original_nonnegative",
            "unrestricted_signed",
        ):
            subset = group[group["variant"] == variant]

            # First average the four assignments per genotype,
            # then macro-average over the 21 genotypes.
            per_genotype = (
                subset.groupby("test_genotype")[
                    ["coverage", "width", "interval_score"]
                ]
                .mean()
            )

            for metric in (
                "coverage",
                "width",
                "interval_score",
            ):
                row[f"{variant}_{metric}"] = float(
                    per_genotype[metric].mean()
                )

            row[
                f"{variant}_genotypes_below_90"
            ] = int(
                (per_genotype["coverage"] < 0.90).sum()
            )

        for metric in (
            "coverage",
            "width",
            "interval_score",
        ):
            row[f"delta_{metric}"] = (
                row[f"unrestricted_signed_{metric}"]
                - row[f"original_nonnegative_{metric}"]
            )

        summary_rows.append(row)

    return pairs, genotype_pairs, pd.DataFrame(summary_rows)


def make_figures(genotype_pairs, diagnostics):
    for horizon in HORIZONS:
        subset = genotype_pairs[
            genotype_pairs["horizon_hours"] == horizon
        ]

        fig, ax = plt.subplots(figsize=(9, 5))

        methods = ["U4_CQR_pooled", "U5_CQR_balanced"]

        for idx, method in enumerate(methods):
            d = subset[subset["method"] == method]
            values = d["delta_interval_score"].to_numpy()

            ax.scatter(
                np.full(len(values), idx, dtype=float),
                values,
                alpha=0.7,
                label=method,
            )

            ax.scatter(
                [idx],
                [values.mean()],
                marker="D",
                s=100,
                edgecolors="black",
                linewidths=0.8,
            )

        ax.axhline(0, linestyle="--", color="gray")
        ax.set_xticks(range(len(methods)), methods)
        ax.set_ylabel(
            "Signed minus clipped interval score (pixels)"
        )
        ax.set_title(
            f"P0.4a — CQR clipping sensitivity ({horizon}h)"
        )
        ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            FIGURES /
            f"P0_4a_score_sensitivity_{horizon}h.png",
            dpi=180,
        )
        plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, horizon in zip(axes, HORIZONS):
        subset = diagnostics[
            diagnostics["horizon_hours"] == horizon
        ]

        counts = (
            subset.groupby("method")["negative_threshold"]
            .sum()
        )

        ax.bar(counts.index, counts.values)
        ax.set_title(f"{horizon}h: Negative CQR thresholds")
        ax.set_ylabel("Number of assignments")
        ax.tick_params(axis="x", rotation=15)

    fig.tight_layout()
    fig.savefig(
        FIGURES / "P0_4a_negative_threshold_counts.png",
        dpi=180,
    )
    plt.close(fig)


def write_report(summary, diagnostics, results):
    lines = [
        "# P0.4a — CQR Threshold-Clipping Sensitivity",
        "",
        "## Experiment",
        "",
        "Compared nonnegative-expansion CQR with signed CQR.",
        "The same quantile models and calibration scores",
        "were used for each paired comparison.",
        "",
        "The original E3.2 endpoint clipping to nonnegative",
        "leaf area was preserved for both variants.",
        "",
        "## Results",
        "",
        "| Horizon | Method | Negative q assignments | "
        "Delta coverage | Delta width | Delta score |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for _, row in summary.iterrows():
        lines.append(
            f"| {int(row['horizon_hours'])}h "
            f"| {row['method']} "
            f"| {int(row['negative_threshold_assignments'])}/84 "
            f"| {row['delta_coverage']:+.5f} "
            f"| {row['delta_width']:+.2f} "
            f"| {row['delta_interval_score']:+.2f} |"
        )

    lines.extend([
        "",
        "Deltas are signed minus original clipped.",
        "Negative interval-score delta means signed CQR",
        "performed better on the reported metric.",
        "",
        "## Interpretation",
        "",
        "If no thresholds were negative, clipping had",
        "no effect on the evaluated predictions.",
        "",
        "If negative thresholds occurred, the change",
        "must be evaluated at the genotype level.",
        "",
        "A negative signed threshold can shrink the",
        "preliminary quantile interval.",
        "",
        "## Limitations",
        "",
        "- Genotypes and test plants are reused across",
        "  four calibration assignments.",
        "- Forecasting windows within plants overlap.",
        "- Held-out genotype shift does not satisfy",
        "  standard conformal assumptions automatically.",
        "- The signed comparison retains E3.2's",
        "  nonnegative-area endpoint convention.",
        "- The sensitivity analysis uses previously",
        "  inspected test genotypes and is exploratory.",
        "",
        "## Next",
        "",
        "Check whether the clipped implementation",
        "materially changes our conclusions.",
        "Then proceed to P0.4b weighted-quantile",
        "finite-sample sensitivity.",
    ])

    path = REPORTS / "P0_4a_cqr_clipping_sensitivity.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    print("=" * 72)
    print("P0.4a — CQR THRESHOLD-CLIPPING SENSITIVITY")
    print("=" * 72)

    if not FOLD_FILE.exists():
        raise FileNotFoundError(FOLD_FILE)

    with FOLD_FILE.open(encoding="utf-8") as f:
        folds = json.load(f)["folds"]

    if len(folds) != 21:
        raise ValueError("Expected 21 outer folds")

    data = cqr.load_data()

    results = []
    diagnostics = []

    for horizon in HORIZONS:
        print(
            f"\n{horizon}h | "
            f"{len(data[horizon]['y']):,} windows",
            flush=True,
        )

        for outer_number, outer in enumerate(folds, 1):
            for inner in outer["inner_folds"]:
                rows, diag = evaluate_split(
                    data[horizon],
                    outer,
                    inner,
                    horizon,
                )
                results.extend(rows)
                diagnostics.extend(diag)

            print(
                f"  Completed genotype {outer_number:02d}/21 "
                f"| {outer['test_genotype']}",
                flush=True,
            )

        save_csv(
            TABLES / "P0_4a_fold_metrics_checkpoint.csv",
            results,
        )

    result_df = pd.DataFrame(results)
    diagnostic_df = pd.DataFrame(diagnostics)

    pairs, genotype_pairs, summary = summarize(
        result_df, diagnostic_df
    )

    save_csv(
        TABLES / "P0_4a_fold_metrics.csv",
        results,
    )
    save_csv(
        TABLES / "P0_4a_threshold_diagnostics.csv",
        diagnostics,
    )
    pairs.to_csv(
        TABLES / "P0_4a_paired_fold_differences.csv",
        index=False,
    )
    genotype_pairs.to_csv(
        TABLES / "P0_4a_genotype_differences.csv",
        index=False,
    )
    summary.to_csv(
        TABLES / "P0_4a_summary.csv",
        index=False,
    )

    make_figures(genotype_pairs, diagnostic_df)
    write_report(summary, diagnostic_df, result_df)

    print("\n" + "=" * 72)
    print("P0.4a — SUMMARY")
    print("=" * 72)

    for _, row in summary.iterrows():
        print(
            f"\n{int(row['horizon_hours'])}h "
            f"| {row['method']}"
            f"\n  Negative q: "
            f"{int(row['negative_threshold_assignments'])}/84"
            f"\n  Coverage change: "
            f"{row['delta_coverage']:+.5f}"
            f"\n  Width change: "
            f"{row['delta_width']:+.2f}"
            f"\n  Interval score change: "
            f"{row['delta_interval_score']:+.2f}"
        )

    print("\nSaved results under:", OUT)
    print("\nP0.4a COMPLETE")


if __name__ == "__main__":
    main()
