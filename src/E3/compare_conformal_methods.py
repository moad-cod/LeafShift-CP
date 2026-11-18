
from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# E3.4 — COMPARATIVE RELIABILITY DECISION GATE
#
# Uses existing E3.1, E3.2 and E3.3 metrics.
# No model fitting and no recalibration.
#
# Aggregation:
#   window -> plant -> four assignments -> genotype
#
# Genotypes, not repeated folds, are bootstrap units.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/E3"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

for p in (TABLES, FIGURES, REPORTS):
    p.mkdir(parents=True, exist_ok=True)

INPUTS = {
    "E3_1": (
        TABLES / "E3_1_fold_metrics.csv",
        TABLES / "E3_1_size_metrics.csv",
    ),
    "E3_2": (
        TABLES / "E3_2_fold_metrics.csv",
        TABLES / "E3_2_size_metrics.csv",
    ),
    "E3_3": (
        TABLES / "E3_3_fold_metrics.csv",
        TABLES / "E3_3_size_metrics.csv",
    ),
}

METHODS = [
    "U1_balanced",
    "U3_normalized_balanced",
    "U5_CQR_balanced",
    "U7_Mondrian_balanced",
]

HORIZONS = (24, 48)
SIZES = (
    "small_lt500",
    "medium_500to1999",
    "large_ge2000",
)

BOOTSTRAPS = 10000
SEED = 20261108

OVERALL_OUT = TABLES / "E3_4_method_comparison.csv"
PAIRED_OUT = TABLES / "E3_4_paired_comparisons.csv"
SIZE_OUT = TABLES / "E3_4_subgroup_comparison.csv"
GENOTYPE_OUT = TABLES / "E3_4_genotype_reliability.csv"
REPORT_OUT = REPORTS / "E3_4_decision_gate.md"


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def load_results():
    overall = {}
    size = {}

    for _, (fold_file, size_file) in INPUTS.items():
        for r in read_csv(fold_file):
            if r["method"] not in METHODS:
                continue

            key = (
                int(r["horizon_hours"]),
                r["method"],
                r["test_genotype"],
                int(r["inner_fold"]),
            )

            if key in overall:
                raise ValueError(f"Duplicate fold result: {key}")

            overall[key] = {
                metric: float(r[metric])
                for metric in ("coverage", "width", "score")
            }

        for r in read_csv(size_file):
            if r["method"] not in METHODS:
                continue

            key = (
                int(r["horizon_hours"]),
                r["method"],
                r["test_genotype"],
                int(r["inner_fold"]),
                r["leaf_size"],
            )

            if key in size:
                raise ValueError(f"Duplicate subgroup result: {key}")

            size[key] = {
                "coverage": float(r["coverage"]),
                "width": float(r["width"]),
                "score": float(r["score"]),
                "plants": int(r["plants"]),
                "windows": int(r["windows"]),
            }

    genotypes = sorted({
        key[2] for key in overall
    })

    if len(genotypes) != 21:
        raise ValueError("Expected 21 genotypes")

    # Require identical genotype and fold membership
    # across all methods.
    for h in HORIZONS:
        for method in METHODS:
            for g in genotypes:
                for fold in (1, 2, 3, 4):
                    key = (h, method, g, fold)
                    if key not in overall:
                        raise ValueError(f"Missing result: {key}")

    return overall, size, genotypes


def bootstrap_ci(values, seed):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)

    samples = rng.integers(
        0, n, size=(BOOTSTRAPS, n)
    )
    estimates = np.mean(values[samples], axis=1)

    return tuple(
        float(x) for x in
        np.quantile(estimates, [0.025, 0.975])
    )


def main():
    overall, size, genotypes = load_results()

    method_rows = []
    paired_rows = []
    subgroup_rows = []
    genotype_rows = []

    # Average the four assignment results
    # within each genotype first.
    by_genotype = {}

    for h in HORIZONS:
        for method in METHODS:
            for g in genotypes:
                values = [
                    overall[(h, method, g, fold)]
                    for fold in (1, 2, 3, 4)
                ]

                averaged = {
                    metric: float(np.mean([
                        r[metric] for r in values
                    ]))
                    for metric in (
                        "coverage", "width", "score"
                    )
                }

                by_genotype[(h, method, g)] = averaged

                genotype_rows.append({
                    "horizon_hours": h,
                    "method": method,
                    "genotype": g,
                    **averaged,
                    "coverage_below_90": int(
                        averaged["coverage"] < 0.90
                    ),
                    "coverage_below_80": int(
                        averaged["coverage"] < 0.80
                    ),
                })

            group = [
                by_genotype[(h, method, g)]
                for g in genotypes
            ]

            method_rows.append({
                "horizon_hours": h,
                "method": method,
                "coverage": float(np.mean([
                    x["coverage"] for x in group
                ])),
                "width": float(np.mean([
                    x["width"] for x in group
                ])),
                "score": float(np.mean([
                    x["score"] for x in group
                ])),
                "genotypes_below_90": sum(
                    x["coverage"] < 0.90 for x in group
                ),
                "genotypes_below_80": sum(
                    x["coverage"] < 0.80 for x in group
                ),
                "worst_genotype_coverage": float(
                    min(x["coverage"] for x in group)
                ),
            })

    # Paired comparisons with U1 as reference,
    # plus direct U3 vs U5 and U5 vs U7.
    comparisons = [
        ("U1_balanced", "U3_normalized_balanced"),
        ("U1_balanced", "U5_CQR_balanced"),
        ("U1_balanced", "U7_Mondrian_balanced"),
        ("U3_normalized_balanced", "U5_CQR_balanced"),
        ("U5_CQR_balanced", "U7_Mondrian_balanced"),
    ]

    for h in HORIZONS:
        for reference, candidate in comparisons:
            for metric in ("score", "coverage", "width"):
                differences = np.asarray([
                    by_genotype[(h, candidate, g)][metric]
                    - by_genotype[(h, reference, g)][metric]
                    for g in genotypes
                ])

                ci_low, ci_high = bootstrap_ci(
                    differences,
                    SEED + h
                    + sum(map(ord, reference + candidate + metric)),
                )

                paired_rows.append({
                    "horizon_hours": h,
                    "reference": reference,
                    "candidate": candidate,
                    "metric": metric,
                    "candidate_minus_reference": float(
                        np.mean(differences)
                    ),
                    "bootstrap_ci_low": ci_low,
                    "bootstrap_ci_high": ci_high,
                    "candidate_lower_count": int(
                        np.sum(differences < 0)
                    ),
                    "candidate_higher_count": int(
                        np.sum(differences > 0)
                    ),
                    "paired_genotypes": len(genotypes),
                })

    # Complete-case subgroup summaries. Eligibility
    # is based only on subgroup sample support.
    for h in HORIZONS:
        for method in METHODS:
            for category in SIZES:
                eligible = []

                for g in genotypes:
                    records = [
                        size.get((h, method, g, fold, category))
                        for fold in (1, 2, 3, 4)
                    ]

                    if any(
                        r is None
                        or r["plants"] < 2
                        or r["windows"] < 30
                        for r in records
                    ):
                        continue

                    eligible.append({
                        metric: float(np.mean([
                            r[metric] for r in records
                        ]))
                        for metric in (
                            "coverage", "width", "score"
                        )
                    })

                if not eligible:
                    raise ValueError(
                        f"No eligible subgroup: "
                        f"{h}h {method} {category}"
                    )

                subgroup_rows.append({
                    "horizon_hours": h,
                    "method": method,
                    "leaf_size": category,
                    "eligible_genotypes": len(eligible),
                    "coverage": float(np.mean([
                        x["coverage"] for x in eligible
                    ])),
                    "width": float(np.mean([
                        x["width"] for x in eligible
                    ])),
                    "score": float(np.mean([
                        x["score"] for x in eligible
                    ])),
                })

    save_csv(OVERALL_OUT, method_rows)
    save_csv(PAIRED_OUT, paired_rows)
    save_csv(SIZE_OUT, subgroup_rows)
    save_csv(GENOTYPE_OUT, genotype_rows)

    # Representative figures.
    for h in HORIZONS:
        selected = [
            r for r in method_rows
            if r["horizon_hours"] == h
        ]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(
            [r["method"] for r in selected],
            [r["score"] for r in selected],
        )
        ax.set_ylabel("90% interval score (lower is better)")
        ax.set_title(
            f"E3.4 — {h}h uncertainty method comparison"
        )
        ax.tick_params(axis="x", labelrotation=18)
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"E3_4_interval_score_{h}h.png",
            dpi=200,
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(SIZES))
        width = 0.19

        lookup = {
            (r["method"], r["leaf_size"]): r
            for r in subgroup_rows
            if r["horizon_hours"] == h
        }

        for i, method in enumerate(METHODS):
            values = [
                100 * lookup[(method, category)]["coverage"]
                for category in SIZES
            ]

            ax.bar(
                x + (i - 1.5) * width,
                values,
                width,
                label=method,
            )

        ax.axhline(90, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(SIZES, rotation=12)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Empirical coverage (%)")
        ax.set_title(
            f"E3.4 — {h}h conditional reliability"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"E3_4_subgroup_coverage_{h}h.png",
            dpi=200,
        )
        plt.close(fig)

    report = [
        "# E3.4 — Comparative Reliability Decision Gate",
        "",
        "## Scope",
        "",
        "Comparison of U1, U3, U5, and U7 across "
        "21 test genotypes, four shared calibration "
        "assignments, and two forecasting horizons.",
        "",
        "No model retraining or recalibration was performed.",
        "",
        "## Overall results",
        "",
        "| Horizon | Method | Coverage | Width | Score | "
        "Genotypes below 90% |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for r in method_rows:
        report.append(
            f"| {r['horizon_hours']}h "
            f"| {r['method']} "
            f"| {r['coverage']:.3f} "
            f"| {r['width']:.1f} "
            f"| {r['score']:.1f} "
            f"| {r['genotypes_below_90']}/21 |"
        )

    report.extend([
        "",
        "## Paired genotype comparisons",
        "",
        "A negative candidate-minus-reference interval "
        "score indicates improvement.",
        "",
        "| Horizon | Reference | Candidate | Metric | "
        "Difference | Bootstrap 95% CI |",
        "|---|---|---|---|---:|---|",
    ])

    for r in paired_rows:
        report.append(
            f"| {r['horizon_hours']}h "
            f"| {r['reference']} "
            f"| {r['candidate']} "
            f"| {r['metric']} "
            f"| {r['candidate_minus_reference']:+.3f} "
            f"| [{r['bootstrap_ci_low']:.3f}, "
            f"{r['bootstrap_ci_high']:.3f}] |"
        )

    report.extend([
        "",
        "## Interpretation limits",
        "",
        "- All tested methods are established baselines.",
        "- The four assignments share outer test plants.",
        "- Bootstrap units are genotypes, not windows.",
        "- Subgroup support differs by leaf-size category.",
        "- The results are exploratory, because test outcomes "
        "have informed method-selection decisions.",
        "- The data do not demonstrate a genotype-shift "
        "coverage guarantee.",
        "- Genotype-specific failures may reflect annotation "
        "or acquisition differences.",
        "",
        "## Research decision",
        "",
        "Inspect the strongest methods for interval score, "
        "subgroup balance, and worst-genotype reliability "
        "before proposing any new calibration method.",
    ])

    REPORT_OUT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("=" * 70)
    print("E3.4 — COMPARATIVE RELIABILITY RESULTS")
    print("=" * 70)

    for r in method_rows:
        print(
            f"\n{r['horizon_hours']}h | {r['method']}"
            f"\n  Coverage: {r['coverage']:.3f}"
            f"\n  Width: {r['width']:.2f} px"
            f"\n  Score: {r['score']:.2f}"
            f"\n  Genotypes below 90%: "
            f"{r['genotypes_below_90']}/21"
            f"\n  Worst-genotype coverage: "
            f"{r['worst_genotype_coverage']:.3f}"
        )

    print("\nPAIRED INTERVAL-SCORE DIFFERENCES")
    for r in paired_rows:
        if r["metric"] != "score":
            continue
        print(
            f"{r['horizon_hours']}h | "
            f"{r['candidate']} vs {r['reference']} "
            f"| delta={r['candidate_minus_reference']:+.2f} "
            f"| CI=[{r['bootstrap_ci_low']:.2f}, "
            f"{r['bootstrap_ci_high']:.2f}]"
        )

    print("\nSAVED:")
    for path in (
        OVERALL_OUT, PAIRED_OUT, SIZE_OUT,
        GENOTYPE_OUT, REPORT_OUT
    ):
        print(path)

    print("\nE3.4 COMPLETE")


if __name__ == "__main__":
    main()
