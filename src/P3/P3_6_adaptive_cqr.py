
#!/usr/bin/env python3
"""
P3.6 — Size-adaptive and size-growth-adaptive CQR.

Uses the original E3.2 models, folds and weights.

Methods:
    U5       original balanced CQR
    SA_CQR   size-adaptive CQR
    SGA_CQR  size + historical-growth-adaptive CQR

Default: one-split pilot, with strict U5 reproduction.
Use --full for all 21 x 4 x 2 configurations.

No existing experimental results are overwritten.
All adaptive parameters are fixed or computed from
fitting genotypes only. All results are retrospective.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/E3"))

import run_cqr_baselines as cqr

OUT = ROOT / "experiments/P3/P3_6"
OUT.mkdir(parents=True, exist_ok=True)

REFERENCE = (
    ROOT / "experiments/E3/tables/E3_2_fold_metrics.csv"
)

METHODS = ("U5", "SA_CQR", "SGA_CQR")
ALPHA = 0.10


def historical_scales(X, growth_reference):
    """All features come from the six past observations.

    Original feature indices:
        0:6    historical leaf areas
        17:22  historical adjacent growth rates
        5      final historical area (the sixth value in the 0:6 area block)

    The growth reference is estimated from fitting
    genotypes only, never from the test genotype.
    """
    X = np.asarray(X, dtype=float)

    area = X[:, 5]
    rates = X[:, 17:22]

    if np.any(~np.isfinite(area)):
        raise ValueError("Invalid historical leaf area")

    if np.any(area <= 0):
        raise ValueError("Nonpositive historical area")

    # Fixed floor, disclosed as a design choice.
    size_scale = np.sqrt(np.maximum(area, 500.0))

    # Absolute historical change-rate magnitude,
    # normalized by historical leaf area.
    relative_dynamics = (
        np.median(np.abs(rates), axis=1)
        / np.maximum(area, 500.0)
    )

    if not np.isfinite(relative_dynamics).all():
        raise ValueError("Nonfinite historical dynamics")

    normalized = np.minimum(
        relative_dynamics / growth_reference, 3.0
    )

    size_growth_scale = size_scale * (1.0 + normalized)

    return {
        "U5": np.ones(len(X), dtype=float),
        "SA_CQR": size_scale,
        "SGA_CQR": size_growth_scale,
    }


def fitting_growth_reference(X_fit):
    """Calculate the growth normalizer on fitting data.

    Fixed rule: empirical 75th percentile of relative
    historical growth magnitude, with a positive floor.
    No test targets or test features are used.
    """
    X_fit = np.asarray(X_fit, dtype=float)

    area = np.maximum(X_fit[:, 5], 500.0)
    rates = X_fit[:, 17:22]

    relative = np.median(
        np.abs(rates), axis=1
    ) / area

    value = float(np.quantile(relative, 0.75))

    return max(value, 1e-8)


def interval_metrics(y, lower, upper, plants):
    """Equal-weight window metrics within each plant.

    Then equally average plants in the test genotype.
    """
    y = np.asarray(y, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    plants = np.asarray(plants).astype(str)

    if not (
        len(y) == len(lower) == len(upper) == len(plants)
    ):
        raise ValueError("Mismatched evaluation lengths")

    if np.any(lower > upper):
        raise ValueError("Crossed prediction intervals")

    covered = (
        (y >= lower) & (y <= upper)
    ).astype(float)

    lower_miss = (y < lower).astype(float)
    upper_miss = (y > upper).astype(float)

    width = upper - lower

    score = (
        width
        + (2 / ALPHA) * (lower - y)
          * (y < lower)
        + (2 / ALPHA) * (y - upper)
          * (y > upper)
    )

    table = pd.DataFrame({
        "plant": plants,
        "coverage": covered,
        "lower_miss": lower_miss,
        "upper_miss": upper_miss,
        "width": width,
        "score": score,
    })

    per_plant = table.groupby("plant").mean(
        numeric_only=True
    )

    result = per_plant.mean().to_dict()
    result["plants"] = len(per_plant)
    result["windows"] = len(table)

    return result


def evaluate_method(
    y_cal, cal_lo, cal_hi,
    w_cal, scale_cal,
    y_test, test_lo, test_hi,
    scale_test, plants_test
):
    """Empirical genotype–plant-balanced CQR."""
    scores = np.maximum(
        cal_lo - y_cal,
        y_cal - cal_hi
    )

    scaled_scores = scores / scale_cal

    threshold_signed = cqr.weighted_quantile(
        scaled_scores, w_cal
    )

    # Match original U5 nonnegative expansion policy.
    threshold = max(0.0, float(threshold_signed))

    lower = np.maximum(
        0.0, test_lo - threshold * scale_test
    )
    upper = test_hi + threshold * scale_test

    metrics = interval_metrics(
        y_test, lower, upper, plants_test
    )

    metrics["threshold"] = threshold
    metrics["threshold_signed"] = float(threshold_signed)

    return metrics


def check_original_u5(row, reference):
    """Strictly verify that the reconstructed U5 agrees
    with the archived E3.2 fold result.
    """
    selected = reference.loc[
        (reference["horizon_hours"] == row["horizon_hours"])
        & (reference["test_genotype"] == row["test_genotype"])
        & (reference["inner_fold"] == row["inner_fold"])
        & (reference["method"] == "U5_CQR_balanced")
    ]

    if len(selected) != 1:
        raise ValueError(
            "Cannot identify unique archived U5 fold: "
            f"{row['test_genotype']}, "
            f"{row['inner_fold']}, "
            f"{row['horizon_hours']}h"
        )

    expected = selected.iloc[0]

    tolerances = {
        "threshold": 1e-3,
        "coverage": 1e-6,
        "width": 1e-3,
        "score": 1e-3,
    }

    for metric, tolerance in tolerances.items():
        actual = float(row[metric])
        original = float(expected[metric])

        if not np.isclose(
            actual, original,
            atol=tolerance,
            rtol=1e-7
        ):
            raise RuntimeError(
                "U5 REPRODUCTION FAILED\n"
                f"Metric: {metric}\n"
                f"Actual: {actual}\n"
                f"Expected: {original}\n"
                "Stop and inspect the original E3.2 "
                "interval construction before proceeding."
            )


def evaluate_split(d, outer, inner, horizon, reference):
    test_genotype = str(outer["test_genotype"])

    fitting_genotypes = set(inner["train_genotypes"])
    calibration_genotypes = set(
        inner["validation_genotypes"]
    )

    if len(fitting_genotypes) != 15:
        raise ValueError("Expected 15 fitting genotypes")

    if len(calibration_genotypes) != 5:
        raise ValueError("Expected 5 calibration genotypes")

    if (
        fitting_genotypes & calibration_genotypes
        or test_genotype in fitting_genotypes
        or test_genotype in calibration_genotypes
    ):
        raise ValueError("Genotype leakage")

    g = d["g"]
    p = d["p"]
    X = d["X"]
    y = d["y"]

    fit = np.isin(g, list(fitting_genotypes))
    cal = np.isin(g, list(calibration_genotypes))
    test = g == test_genotype

    if set(g[fit]) != fitting_genotypes:
        raise ValueError("Missing fitting genotype")

    if set(g[cal]) != calibration_genotypes:
        raise ValueError("Missing calibration genotype")

    if set(g[test]) != {test_genotype}:
        raise ValueError("Missing held-out genotype")

    # Original fitting and calibration weights.
    w_fit = cqr.balanced_weights(
        g[fit], p[fit]
    )
    w_cal = cqr.balanced_weights(
        g[cal], p[cal]
    )

    # Original base quantile models.
    model_lo = cqr.fit_quantile_model(
        X[fit], y[fit], w_fit, 0.05
    )
    model_hi = cqr.fit_quantile_model(
        X[fit], y[fit], w_fit, 0.95
    )

    cal_a = model_lo.predict(X[cal])
    cal_b = model_hi.predict(X[cal])

    test_a = model_lo.predict(X[test])
    test_b = model_hi.predict(X[test])

    # Match original quantile-crossing handling.
    cal_lo = np.minimum(cal_a, cal_b)
    cal_hi = np.maximum(cal_a, cal_b)

    test_lo = np.minimum(test_a, test_b)
    test_hi = np.maximum(test_a, test_b)

    growth_ref = fitting_growth_reference(X[fit])

    scales_cal = historical_scales(
        X[cal], growth_ref
    )
    scales_test = historical_scales(
        X[test], growth_ref
    )

    results = []

    for method in METHODS:
        metrics = evaluate_method(
            y[cal], cal_lo, cal_hi,
            w_cal, scales_cal[method],
            y[test], test_lo, test_hi,
            scales_test[method], p[test]
        )

        row = {
            "horizon_hours": int(horizon),
            "outer_fold": outer["outer_fold"],
            "inner_fold": inner["inner_fold"],
            "test_genotype": test_genotype,
            "method": method,
            "growth_reference": growth_ref,
            **metrics,
        }

        if method == "U5":
            check_original_u5(row, reference)

        results.append(row)

    return results


def summarize_full(results):
    """Average assignments within genotype first.

    Then average genotypes equally.
    """
    genotype = (
        results.groupby(
            ["horizon_hours", "method", "test_genotype"]
        )[
            [
                "coverage", "lower_miss", "upper_miss",
                "width", "score"
            ]
        ]
        .mean()
        .reset_index()
    )

    genotype.to_csv(
        OUT / "P3_6_genotype_metrics.csv",
        index=False
    )

    summary = (
        genotype.groupby(["horizon_hours", "method"])
        .agg(
            genotypes=("test_genotype", "nunique"),
            coverage=("coverage", "mean"),
            lower_miss=("lower_miss", "mean"),
            upper_miss=("upper_miss", "mean"),
            width=("width", "mean"),
            score=("score", "mean"),
            worst_coverage=("coverage", "min"),
            genotypes_below_90=(
                "coverage",
                lambda x: int((x < .90).sum())
            ),
        )
        .reset_index()
    )

    summary.to_csv(
        OUT / "P3_6_method_summary.csv",
        index=False
    )

    for h in (24, 48):
        part = summary[
            summary.horizon_hours == h
        ]

        if len(part) != 3 or not part.genotypes.eq(21).all():
            raise ValueError(
                "Incomplete genotype-macro summary"
            )

        original = part[part.method == "U5"].iloc[0]
        expected = .899712 if h == 24 else .9004006

        if abs(original.coverage - expected) > .00015:
            raise ValueError(
                "Original macro coverage mismatch"
            )

        print(f"\n{'=' * 65}")
        print(f"{h}h METHOD COMPARISON")
        print(f"{'=' * 65}")
        print(
            part[
                [
                    "method", "coverage", "width", "score",
                    "lower_miss", "upper_miss",
                    "worst_coverage", "genotypes_below_90"
                ]
            ].to_string(index=False)
        )

    return genotype, summary


def make_figures(summary):
    for h in (24, 48):
        sub = summary[
            summary.horizon_hours == h
        ].set_index("method").loc[list(METHODS)]

        for metric, ylabel, multiplier in [
            ("coverage", "Genotype-macro coverage (%)", 100),
            ("width", "Mean interval width (px²)", 1),
            ("score", "Mean interval score (px²)", 1),
        ]:
            fig, ax = plt.subplots(figsize=(7, 4.5))

            ax.bar(
                list(METHODS),
                sub[metric].to_numpy() * multiplier
            )

            if metric == "coverage":
                ax.axhline(
                    90,
                    linestyle="--",
                    color="black",
                    linewidth=1
                )

            ax.set_ylabel(ylabel)
            ax.set_title(
                f"P3.6 — {metric}, {h}h"
            )
            ax.grid(axis="y", alpha=.15)
            fig.tight_layout()

            stem = f"P3_6_{metric}_{h}h"
            fig.savefig(
                OUT / f"{stem}.png",
                dpi=250,
                bbox_inches="tight"
            )
            fig.savefig(
                OUT / f"{stem}.pdf",
                bbox_inches="tight"
            )
            plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--full",
        action="store_true",
        help="Evaluate all 168 genotype/assignment/horizon cases"
    )
    args = parser.parse_args()

    if not REFERENCE.is_file():
        raise FileNotFoundError(REFERENCE)

    reference = pd.read_csv(REFERENCE)

    with cqr.FOLDS.open(encoding="utf-8") as f:
        folds = json.load(f)["folds"]

    if len(folds) != 21:
        raise ValueError("Expected 21 outer folds")

    data = cqr.load_data()

    rows = []

    if args.full:
        tasks = [
            (h, outer, inner)
            for h in (24, 48)
            for outer in folds
            for inner in outer["inner_folds"]
        ]
    else:
        # Pilot: one 24h split only.
        tasks = [
            (24, folds[0], folds[0]["inner_folds"][0])
        ]

    for index, (h, outer, inner) in enumerate(tasks, 1):
        result = evaluate_split(
            data[h], outer, inner, h, reference
        )
        rows.extend(result)

        print(
            f"[{index}/{len(tasks)}] "
            f"{h}h | {outer['test_genotype']} "
            f"| assignment {inner['inner_fold']} "
            "| U5 PASS",
            flush=True
        )

        # New experiment checkpoint only.
        pd.DataFrame(rows).to_csv(
            OUT / (
                "P3_6_full_fold_metrics.csv"
                if args.full
                else "P3_6_pilot_fold_metrics.csv"
            ),
            index=False
        )

    result_df = pd.DataFrame(rows)

    if not args.full:
        print("\nPILOT COMPLETE")
        print(
            result_df[
                ["method", "coverage", "width", "score"]
            ].to_string(index=False)
        )
        print("\nU5 archived-fold verification: PASS")
        return

    if len(result_df) != 168 * len(METHODS):
        raise ValueError("Incomplete full experiment")

    _, summary = summarize_full(result_df)
    make_figures(summary)

    report = [
        "# P3.6 — Adaptive CQR Benchmark",
        "",
        "All 168 original configurations evaluated.",
        "U5 checked against the archived E3.2 fold metrics.",
        "",
        "Methods:",
        "- U5: original balanced CQR.",
        "- SA-CQR: historical-size normalization.",
        "- SGA-CQR: size and relative historical-dynamics normalization.",
        "",
        "Growth reference computed from fitting genotypes.",
        "Calibration thresholds use calibration genotypes only.",
        "",
        "No independent prospective genotype validation is claimed.",
        "No formal genotype-conditional coverage guarantee is claimed.",
        "",
        "## Results",
        "",
        summary.to_string(index=False),
        "",
        "## Interpretation",
        "",
        "Retrospective methodological comparison.",
        "No method should be selected as prospectively validated",
        "based solely on these previously examined test genotypes.",
    ]

    (OUT / "P3_6_results.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("\nFULL BENCHMARK COMPLETE")
    print("Results saved to:", OUT)


if __name__ == "__main__":
    main()
