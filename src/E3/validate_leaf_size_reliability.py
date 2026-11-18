
from pathlib import Path
from collections import defaultdict
import csv
import math

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# E3.0 — LEAF-SIZE RELIABILITY VALIDATION
#
# Input: E2.4 saved leaf-size metrics.
# No retraining. No recalibration. No test-driven tuning.
#
# Primary contrast:
#   medium-leaf coverage minus large-leaf coverage
#
# Paired by held-out genotype and calibration assignment.
#
# Statistical unit: genotype, NOT forecasting window.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "experiments/E2/tables/"
    "E2_4_size_metrics.csv"
)

BASE = ROOT / "experiments/E3"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

for directory in (TABLES, FIGURES, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

FOLD_OUTPUT = TABLES / "E3_0_fold_size_summary.csv"
PAIRED_OUTPUT = TABLES / "E3_0_paired_size_gaps.csv"
SUMMARY_OUTPUT = TABLES / "E3_0_size_reliability_summary.csv"
REPORT_OUTPUT = REPORTS / "E3_0_leaf_size_reliability.md"

HORIZONS = (24, 48)
FOLDS = (1, 2, 3, 4)
METHODS = ("U0_pooled", "U1_balanced")

SIZES = (
    "small_lt500",
    "medium_500to1999",
    "large_ge2000",
)

NOMINAL = 0.90
MIN_PLANTS = 2
MIN_WINDOWS = 30

BOOTSTRAPS = 10000
SEED = 20261108


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows: {path}")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def read_input():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    lookup = {}

    with INPUT.open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            h = int(row["horizon_hours"])
            fold = int(row["inner_fold"])
            genotype = row["test_genotype"]
            method = row["method"]
            size = row["leaf_size"]

            if (
                h not in HORIZONS
                or fold not in FOLDS
                or method not in METHODS
                or size not in SIZES
            ):
                raise ValueError(
                    f"Unexpected E2.4 row: {row}"
                )

            key = (
                h, fold, genotype, method, size
            )

            if key in lookup:
                raise ValueError(
                    f"Duplicate size metric: {key}"
                )

            parsed = {
                "plants": int(row["plants"]),
                "windows": int(row["windows"]),
                "coverage": float(row["coverage"]),
                "width": float(row["width"]),
                "score": float(row["score"]),
                "mae": float(row["mae"]),
            }

            if not all(
                math.isfinite(parsed[k])
                for k in (
                    "coverage",
                    "width",
                    "score",
                    "mae",
                )
            ):
                raise ValueError(
                    f"Nonfinite metrics: {key}"
                )

            if not 0 <= parsed["coverage"] <= 1:
                raise ValueError(
                    f"Invalid coverage: {key}"
                )

            if (
                parsed["plants"] <= 0
                or parsed["windows"] <= 0
                or parsed["width"] < 0
                or parsed["score"] < 0
            ):
                raise ValueError(
                    f"Invalid group support: {key}"
                )

            lookup[key] = parsed

    return lookup


def eligible(row):
    return (
        row is not None
        and row["plants"] >= MIN_PLANTS
        and row["windows"] >= MIN_WINDOWS
    )


def genotype_bootstrap_ci(values, seed):
    """
    Resample held-out genotypes.

    Each genotype contributes one value averaged
    across its four predefined calibration folds.
    """
    values = np.asarray(
        values, dtype=np.float64
    )

    if len(values) < 2:
        return float("nan"), float("nan")

    rng = np.random.default_rng(seed)
    n = len(values)
    means = np.empty(BOOTSTRAPS)

    for i in range(BOOTSTRAPS):
        indices = rng.integers(0, n, size=n)
        means[i] = values[indices].mean()

    low, high = np.quantile(
        means, [0.025, 0.975]
    )

    return float(low), float(high)


def paired_contrasts(lookup, genotypes):
    """
    Complete-case comparison:
      same genotype,
      same calibration fold,
      same horizon,
      same method,
      medium and large leaves.

    Only genotypes eligible for BOTH categories in
    ALL FOUR calibration assignments are included.

    This avoids treating the 4 repeated assignments
    as 4 independent biological replications.
    """
    paired_rows = []
    summaries = []

    for h in HORIZONS:
        for method in METHODS:
            genotype_gaps = []

            for genotype in genotypes:
                fold_records = []

                for fold in FOLDS:
                    medium = lookup.get((
                        h, fold, genotype, method,
                        "medium_500to1999"
                    ))
                    large = lookup.get((
                        h, fold, genotype, method,
                        "large_ge2000"
                    ))

                    if not (
                        eligible(medium)
                        and eligible(large)
                    ):
                        fold_records = []
                        break

                    fold_records.append({
                        "horizon_hours": h,
                        "method": method,
                        "genotype": genotype,
                        "inner_fold": fold,
                        "medium_coverage": medium["coverage"],
                        "large_coverage": large["coverage"],
                        "coverage_gap_pp": 100 * (
                            medium["coverage"]
                            - large["coverage"]
                        ),
                        "medium_score": medium["score"],
                        "large_score": large["score"],
                        "medium_width": medium["width"],
                        "large_width": large["width"],
                        "large_below_90": int(
                            large["coverage"] < NOMINAL
                        ),
                    })

                if len(fold_records) == 4:
                    paired_rows.extend(fold_records)

                    genotype_gaps.append(float(
                        np.mean([
                            r["coverage_gap_pp"]
                            for r in fold_records
                        ])
                    ))

            if not genotype_gaps:
                raise ValueError(
                    f"No complete paired genotypes: "
                    f"{h}h {method}"
                )

            low, high = genotype_bootstrap_ci(
                genotype_gaps,
                SEED + h
                + sum(ord(c) for c in method),
            )

            relevant = [
                r for r in paired_rows
                if (
                    r["horizon_hours"] == h
                    and r["method"] == method
                )
            ]

            fold_means = [
                float(np.mean([
                    r["coverage_gap_pp"]
                    for r in relevant
                    if r["inner_fold"] == fold
                ]))
                for fold in FOLDS
            ]

            n = len(genotype_gaps)

            summaries.append({
                "horizon_hours": h,
                "method": method,
                "paired_genotypes": n,
                "mean_medium_minus_large_pp": float(
                    np.mean(genotype_gaps)
                ),
                "bootstrap_CI_low_pp": low,
                "bootstrap_CI_high_pp": high,
                "genotypes_positive_gap": int(
                    np.sum(
                        np.asarray(genotype_gaps) > 0
                    )
                ),
                "folds_positive_gap": sum(
                    value > 0
                    for value in fold_means
                ),
                "mean_medium_coverage": float(
                    np.mean([
                        r["medium_coverage"]
                        for r in relevant
                    ])
                ),
                "mean_large_coverage": float(
                    np.mean([
                        r["large_coverage"]
                        for r in relevant
                    ])
                ),
                "large_below_90_genotype_folds": sum(
                    r["large_below_90"]
                    for r in relevant
                ),
                "total_paired_genotype_folds": len(
                    relevant
                ),
            })

    return paired_rows, summaries


def summarize_folds(lookup, genotypes):
    rows = []

    for h in HORIZONS:
        for fold in FOLDS:
            for method in METHODS:
                for size in SIZES:
                    group = []

                    for genotype in genotypes:
                        entry = lookup.get((
                            h, fold, genotype,
                            method, size
                        ))

                        if eligible(entry):
                            group.append(entry)

                    if not group:
                        continue

                    rows.append({
                        "horizon_hours": h,
                        "inner_fold": fold,
                        "method": method,
                        "leaf_size": size,
                        "eligible_genotypes": len(group),
                        "mean_coverage": float(
                            np.mean([
                                r["coverage"]
                                for r in group
                            ])
                        ),
                        "mean_width": float(
                            np.mean([
                                r["width"]
                                for r in group
                            ])
                        ),
                        "mean_score": float(
                            np.mean([
                                r["score"]
                                for r in group
                            ])
                        ),
                        "mean_mae": float(
                            np.mean([
                                r["mae"]
                                for r in group
                            ])
                        ),
                        "genotypes_below_90": sum(
                            r["coverage"] < NOMINAL
                            for r in group
                        ),
                        "genotypes_below_80": sum(
                            r["coverage"] < 0.80
                            for r in group
                        ),
                    })

    return rows


def make_figures(fold_rows, paired_rows, genotypes):
    for h in HORIZONS:
        # Figure 1: subgroup coverage by calibration fold
        fig, ax = plt.subplots(figsize=(10, 6))

        for method in METHODS:
            for size in (
                "medium_500to1999",
                "large_ge2000",
            ):
                subset = [
                    r for r in fold_rows
                    if (
                        r["horizon_hours"] == h
                        and r["method"] == method
                        and r["leaf_size"] == size
                    )
                ]

                subset.sort(
                    key=lambda r: r["inner_fold"]
                )

                ax.plot(
                    [r["inner_fold"] for r in subset],
                    [
                        100 * r["mean_coverage"]
                        for r in subset
                    ],
                    marker="o",
                    label=f"{method}: {size}",
                )

        ax.axhline(
            90.0, linestyle="--",
            label="Nominal 90%"
        )
        ax.set_xticks(FOLDS)
        ax.set_xlabel("Calibration assignment")
        ax.set_ylabel("Empirical coverage (%)")
        ax.set_title(
            f"E3.0 — {h}h leaf-size coverage stability"
        )
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.2)
        fig.tight_layout()

        fig.savefig(
            FIGURES / f"E3_0_size_coverage_{h}h.png",
            dpi=200,
        )
        plt.close(fig)

        # Figure 2: paired medium-large gap for U1
        selected = [
            r for r in paired_rows
            if (
                r["horizon_hours"] == h
                and r["method"] == "U1_balanced"
            )
        ]

        by_genotype = defaultdict(list)

        for r in selected:
            by_genotype[r["genotype"]].append(
                r["coverage_gap_pp"]
            )

        labels = sorted(
            by_genotype,
            key=lambda g: np.mean(
                by_genotype[g]
            ),
        )

        values = [
            float(np.mean(by_genotype[g]))
            for g in labels
        ]

        fig, ax = plt.subplots(figsize=(11, 7))
        ax.barh(labels, values)
        ax.axvline(0, linestyle="--")
        ax.set_xlabel(
            "Medium minus large coverage "
            "(percentage points)"
        )
        ax.set_title(
            f"E3.0 — {h}h paired U1 leaf-size gap"
        )
        fig.tight_layout()

        fig.savefig(
            FIGURES / f"E3_0_paired_gap_{h}h.png",
            dpi=200,
        )
        plt.close(fig)


def main():
    print("=" * 70)
    print("E3.0 — LEAF-SIZE RELIABILITY VALIDATION")
    print("=" * 70)

    lookup = read_input()

    genotypes = sorted({
        key[2] for key in lookup
    })

    if len(genotypes) != 21:
        raise ValueError(
            f"Expected 21 genotypes, got {len(genotypes)}"
        )

    # The method comparison must refer to the
    # same genotype, horizon, assignment, size.
    # E2.4 should contain both methods wherever
    # a subgroup exists.
    for (h, fold, g, method, size) in lookup:
        other = (
            "U1_balanced"
            if method == "U0_pooled"
            else "U0_pooled"
        )

        if (h, fold, g, other, size) not in lookup:
            raise ValueError(
                "U0/U1 subgroup pairing mismatch"
            )

    fold_rows = summarize_folds(
        lookup, genotypes
    )
    paired_rows, summary_rows = paired_contrasts(
        lookup, genotypes
    )

    write_csv(FOLD_OUTPUT, fold_rows)
    write_csv(PAIRED_OUTPUT, paired_rows)
    write_csv(SUMMARY_OUTPUT, summary_rows)

    make_figures(
        fold_rows, paired_rows, genotypes
    )

    report = [
        "# E3.0 — Leaf-Size Reliability Validation",
        "",
        "## Question",
        "",
        "Does large-leaf prediction-interval undercoverage "
        "persist across four calibration assignments?",
        "",
        "## Design",
        "",
        "- Reuses saved E2.4 metrics.",
        "- No retraining or recalibration.",
        "- Nominal coverage: 90%.",
        "- Leaf-size boundaries frozen from E2.3.",
        "- Eligible groups: >=2 plants and >=30 windows.",
        "- Primary contrast: medium minus large coverage.",
        "- Complete-case paired genotype analysis.",
        "- Four folds averaged within genotype.",
        "- Bootstrap resampling unit: genotype.",
        "",
        "## Main paired results",
        "",
        "| Horizon | Method | Genotypes | Gap (pp) | "
        "95% bootstrap CI | Positive folds |",
        "|---|---|---:|---:|---|---:|",
    ]

    for r in summary_rows:
        report.append(
            f"| {r['horizon_hours']}h "
            f"| {r['method']} "
            f"| {r['paired_genotypes']} "
            f"| {r['mean_medium_minus_large_pp']:.2f} "
            f"| [{r['bootstrap_CI_low_pp']:.2f}, "
            f"{r['bootstrap_CI_high_pp']:.2f}] "
            f"| {r['folds_positive_gap']}/4 |"
        )

    report.extend([
        "",
        "## Interpretation limits",
        "",
        "- The four assignments reuse test plants.",
        "- Fitting/calibration composition changes together.",
        "- Bootstrap intervals are descriptive.",
        "- Subgroups are defined from historical leaf area.",
        "- The fixed absolute-residual interval may have "
        "scale-dependent errors.",
        "- Differences do not establish causal mechanisms.",
        "- Observed subgroup coverage is not a formal "
        "conditional-coverage guarantee.",
        "- Annotation errors and acquisition schedules "
        "remain potential confounders.",
        "- These results have already informed method "
        "development, so outer tests are not pristine "
        "confirmatory evaluation sets.",
        "",
        "## Next",
        "",
        "E3.1 — Compare against established normalized "
        "residual conformal prediction.",
    ])

    REPORT_OUTPUT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print(
        f"\nInput genotype-size results: {len(lookup)}"
    )
    print(
        f"Fold-size summary rows: {len(fold_rows)}"
    )
    print(
        f"Complete paired rows: {len(paired_rows)}"
    )

    for r in summary_rows:
        print(
            f"\n{r['horizon_hours']}h "
            f"| {r['method']}"
        )
        print(
            "  Paired genotypes: "
            f"{r['paired_genotypes']}/21"
        )
        print(
            "  Mean medium-minus-large coverage gap: "
            f"{r['mean_medium_minus_large_pp']:+.2f} pp"
        )
        print(
            "  95% genotype-bootstrap interval: "
            f"[{r['bootstrap_CI_low_pp']:.2f}, "
            f"{r['bootstrap_CI_high_pp']:.2f}] pp"
        )
        print(
            "  Folds with positive gap: "
            f"{r['folds_positive_gap']}/4"
        )
        print(
            "  Genotypes with positive mean gap: "
            f"{r['genotypes_positive_gap']}/"
            f"{r['paired_genotypes']}"
        )
        print(
            "  Matched-subset medium coverage: "
            f"{r['mean_medium_coverage']:.3f}"
        )
        print(
            "  Matched-subset large coverage: "
            f"{r['mean_large_coverage']:.3f}"
        )

    print("\nSaved:")
    print(FOLD_OUTPUT)
    print(PAIRED_OUTPUT)
    print(SUMMARY_OUTPUT)
    print(REPORT_OUTPUT)

    print("\nE3.0 COMPLETE")


if __name__ == "__main__":
    main()