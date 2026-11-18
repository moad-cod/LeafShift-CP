
from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# E3.5 — GENOTYPE-LEVEL RELIABILITY DIAGNOSTICS
#
# Inputs:
#   E3.4 genotype-level results
#   E1 forecasting manifest
#
# No model training or calibration.
#
# Analyses:
#   1. Persistent failures across U1/U3/U5/U7
#   2. Genotype-level historical trajectory summaries
#   3. Descriptive correlations with empirical coverage
#   4. Reproducible figures and Markdown report
#
# These are exploratory associations, NOT causality.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "data/processed/forecasting/"
    "forecast_windows.csv"
)

METRICS = (
    ROOT / "experiments/E3/tables/"
    "E3_4_genotype_reliability.csv"
)

BASE = ROOT / "experiments/E3"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

FEATURE_OUT = TABLES / "E3_5_genotype_features.csv"
FAILURE_OUT = TABLES / "E3_5_failure_overlap.csv"
ASSOCIATION_OUT = TABLES / "E3_5_feature_associations.csv"
REPORT_OUT = REPORTS / "E3_5_genotype_failure_diagnostics.md"

METHODS = (
    "U1_balanced",
    "U3_normalized_balanced",
    "U5_CQR_balanced",
    "U7_Mondrian_balanced",
)

HORIZONS = (24, 48)

FEATURES = (
    "last_area_mean",
    "large_leaf_fraction",
    "absolute_growth_mean",
    "relative_growth_mean",
    "historical_cv_mean",
    "history_span_mean",
    "history_skipped_fraction",
)

NOMINAL = 0.90


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty output: {path}")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return float(np.mean(values))


def build_genotype_features():
    # First aggregate windows within plants, so plants
    # with more overlapping windows do not dominate.
    by_plant = defaultdict(
        lambda: defaultdict(list)
    )

    for row in read_csv(MANIFEST):
        horizon = int(row["horizon_hours"])
        if horizon not in HORIZONS:
            continue

        areas = np.asarray(
            json.loads(row["history_areas_pixels"]),
            dtype=float,
        )

        times = np.asarray(
            json.loads(row["history_relative_hours"]),
            dtype=float,
        )

        if len(areas) != 6 or len(times) != 6:
            raise ValueError(
                "Expected six historical observations"
            )

        if np.any(~np.isfinite(areas)):
            raise ValueError("Invalid historical areas")

        if np.any(np.diff(times) <= 0):
            raise ValueError("Non-increasing history times")

        first = float(areas[0])
        last = float(areas[-1])

        growth = last - first
        relative_growth = (
            growth / max(abs(first), 1.0)
        )

        cv = float(
            np.std(areas) / max(np.mean(areas), 1.0)
        )

        skipped = (
            float(row["history_skipped_image_frames"]) > 0
        )

        key = (
            horizon,
            row["genotype"],
            row["plant_id"],
        )

        features = {
            "last_area_mean": last,
            "large_leaf_fraction": float(last >= 2000),
            "absolute_growth_mean": growth,
            "relative_growth_mean": relative_growth,
            "historical_cv_mean": cv,
            "history_span_mean": float(
                times[-1] - times[0]
            ),
            "history_skipped_fraction": float(skipped),
        }

        for feature, value in features.items():
            by_plant[key][feature].append(value)

    # Then average plant characteristics within genotype.
    by_genotype = defaultdict(list)

    for (h, g, plant), observations in by_plant.items():
        by_genotype[(h, g)].append({
            feature: mean(values)
            for feature, values in observations.items()
        })

    result = []

    for (h, g), plants in sorted(by_genotype.items()):
        result.append({
            "horizon_hours": h,
            "genotype": g,
            "plants": len(plants),
            **{
                feature: mean([
                    p[feature] for p in plants
                ])
                for feature in FEATURES
            },
        })

    if len(result) != 42:
        raise ValueError(
            f"Expected 42 genotype/horizon rows, "
            f"found {len(result)}"
        )

    return result


def load_coverage():
    result = {}

    for row in read_csv(METRICS):
        horizon = int(row["horizon_hours"])
        method = row["method"]
        genotype = row["genotype"]

        if method not in METHODS:
            continue

        key = (horizon, genotype, method)

        if key in result:
            raise ValueError(
                f"Duplicate coverage result: {key}"
            )

        result[key] = float(row["coverage"])

    if len(result) != 168:
        raise ValueError(
            f"Expected 168 coverage results, "
            f"found {len(result)}"
        )

    return result


def build_failure_overlap(features, coverage):
    result = []

    for row in features:
        h = row["horizon_hours"]
        g = row["genotype"]

        values = {
            m: coverage[(h, g, m)]
            for m in METHODS
        }

        below = sum(
            value < NOMINAL
            for value in values.values()
        )

        result.append({
            "horizon_hours": h,
            "genotype": g,
            "undercovered_methods": below,
            "undercovered_all_four": int(below == 4),
            "undercovered_three_or_more": int(
                below >= 3
            ),
            "min_method_coverage": min(values.values()),
            "max_method_coverage": max(values.values()),
            "mean_method_coverage": mean(
                list(values.values())
            ),
            **values,
        })

    return result


def rank_correlations(x, y):
    # Ties receive average ranks. Spearman's rho is
    # Pearson correlation of rank-transformed values.
    def ranks(values):
        values = np.asarray(values, dtype=float)

        order = np.argsort(
            values, kind="stable"
        )
        output = np.empty(len(values), dtype=float)

        sorted_values = values[order]
        i = 0

        while i < len(values):
            j = i + 1

            while (
                j < len(values)
                and sorted_values[j] == sorted_values[i]
            ):
                j += 1

            output[order[i:j]] = (
                (i + j - 1) / 2.0 + 1.0
            )
            i = j

        return output

    a = ranks(x)
    b = ranks(y)

    if (
        np.std(a) < 1e-12
        or np.std(b) < 1e-12
    ):
        return float("nan")

    return float(
        np.corrcoef(a, b)[0, 1]
    )


def analyze_associations(features, coverage):
    rows = []

    for h in HORIZONS:
        selected = [
            row for row in features
            if row["horizon_hours"] == h
        ]

        if len(selected) != 21:
            raise ValueError(
                "Expected 21 genotypes per horizon"
            )

        for method in METHODS:
            y = [
                coverage[(
                    h, row["genotype"], method
                )]
                for row in selected
            ]

            for feature in FEATURES:
                x = [
                    row[feature]
                    for row in selected
                ]

                rows.append({
                    "horizon_hours": h,
                    "method": method,
                    "feature": feature,
                    "spearman_rho": rank_correlations(
                        x, y
                    ),
                    "genotypes": len(selected),
                })

    return rows


def make_figures(overlap, features):
    for h in HORIZONS:
        selected = [
            row for row in overlap
            if row["horizon_hours"] == h
        ]

        selected.sort(
            key=lambda row: row[
                "mean_method_coverage"
            ]
        )

        fig, ax = plt.subplots(
            figsize=(12, 7)
        )

        x = np.arange(len(selected))
        width = 0.20

        for index, method in enumerate(METHODS):
            ax.bar(
                x + (index - 1.5) * width,
                [
                    100 * row[method]
                    for row in selected
                ],
                width,
                label=method,
            )

        ax.axhline(
            90, linestyle="--",
            label="Nominal 90%"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [row["genotype"] for row in selected],
            rotation=65,
            ha="right",
        )
        ax.set_ylabel("Empirical coverage (%)")
        ax.set_ylim(0, 105)
        ax.set_title(
            f"E3.5 — {h}h genotype reliability"
        )
        ax.legend(fontsize=8)
        fig.tight_layout()

        fig.savefig(
            FIGURES
            / f"E3_5_genotype_reliability_{h}h.png",
            dpi=200,
        )
        plt.close(fig)

        # Diagnostic scatter: a single prespecified
        # observable feature vs U5 coverage.
        selected_features = [
            row for row in features
            if row["horizon_hours"] == h
        ]

        cov = {
            row["genotype"]: row[
                "U5_CQR_balanced"
            ]
            for row in selected
        }

        fig, ax = plt.subplots(figsize=(8, 5))

        for row in selected_features:
            genotype = row["genotype"]

            ax.scatter(
                row["large_leaf_fraction"],
                100 * cov[genotype],
            )

            ax.annotate(
                genotype,
                (
                    row["large_leaf_fraction"],
                    100 * cov[genotype],
                ),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7,
            )

        ax.axhline(90, linestyle="--")
        ax.set_xlabel(
            "Mean fraction of large-leaf windows"
        )
        ax.set_ylabel(
            "U5 empirical genotype coverage (%)"
        )
        ax.set_title(
            f"E3.5 — {h}h size composition vs coverage"
        )
        fig.tight_layout()

        fig.savefig(
            FIGURES
            / f"E3_5_large_leaf_association_{h}h.png",
            dpi=200,
        )
        plt.close(fig)


def main():
    print("=" * 70)
    print("E3.5 — GENOTYPE FAILURE DIAGNOSTICS")
    print("=" * 70)

    features = build_genotype_features()
    coverage = load_coverage()

    overlap = build_failure_overlap(
        features, coverage
    )

    associations = analyze_associations(
        features, coverage
    )

    write_csv(FEATURE_OUT, features)
    write_csv(FAILURE_OUT, overlap)
    write_csv(ASSOCIATION_OUT, associations)

    make_figures(overlap, features)

    report = [
        "# E3.5 — Genotype Failure Diagnostics",
        "",
        "## Scientific objective",
        "",
        "Investigate whether persistent uncertainty "
        "undercoverage across conformal methods is "
        "associated with historical growth characteristics.",
        "",
        "## Failure overlap",
        "",
        "| Horizon | Genotype | Failed methods | "
        "Mean coverage |",
        "|---|---|---:|---:|",
    ]

    for row in sorted(
        overlap,
        key=lambda r: (
            r["horizon_hours"],
            -r["undercovered_methods"],
            r["mean_method_coverage"],
        ),
    ):
        if row["undercovered_methods"] < 3:
            continue

        report.append(
            f"| {row['horizon_hours']}h "
            f"| {row['genotype']} "
            f"| {row['undercovered_methods']}/4 "
            f"| {row['mean_method_coverage']:.3f} |"
        )

    report.extend([
        "",
        "## Exploratory feature associations",
        "",
        "| Horizon | Method | Feature | Spearman rho |",
        "|---|---|---|---:|",
    ])

    ordered = sorted(
        associations,
        key=lambda r: abs(r["spearman_rho"])
        if np.isfinite(r["spearman_rho"])
        else -1.0,
        reverse=True,
    )

    for row in ordered:
        report.append(
            f"| {row['horizon_hours']}h "
            f"| {row['method']} "
            f"| {row['feature']} "
            f"| {row['spearman_rho']:+.3f} |"
        )

    report.extend([
        "",
        "## Scientific limitations",
        "",
        "- Only 21 independent genotype-level units.",
        "- Exploratory associations, not causal effects.",
        "- Multiple features and methods are examined.",
        "- Genotype-level traits derive from overlapping "
        "forecasting windows.",
        "- Historical features are known at the cutoff, "
        "but genotype-level summaries are diagnostic "
        "rather than deployable predictors.",
        "- Target-genotype coverage itself is unknown "
        "at deployment time.",
        "- Acquisition conditions and annotation quality "
        "may confound associations.",
        "- No distribution-free coverage guarantee "
        "under genotype shift is established.",
        "",
        "## Decision",
        "",
        "Use these results to determine whether observable "
        "growth characteristics plausibly explain "
        "persistent calibration failures. Do not "
        "construct a new method solely from correlations "
        "selected on the outer-test genotypes.",
    ])

    REPORT_OUT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print(
        "\nGENOTYPES UNDERCOVERED BY ALL FOUR METHODS"
    )

    for h in HORIZONS:
        failures = sorted([
            row["genotype"]
            for row in overlap
            if (
                row["horizon_hours"] == h
                and row["undercovered_all_four"]
            )
        ])

        print(
            f"  {h}h: {len(failures)}/21 "
            f"— {', '.join(failures) if failures else 'none'}"
        )

    print(
        "\nSTRONGEST EXPLORATORY ASSOCIATIONS"
    )

    for h in HORIZONS:
        selected = sorted(
            [
                row for row in associations
                if (
                    row["horizon_hours"] == h
                    and row["method"] == "U5_CQR_balanced"
                    and np.isfinite(row["spearman_rho"])
                )
            ],
            key=lambda r: abs(r["spearman_rho"]),
            reverse=True,
        )

        print(f"\n{h}h — U5:")
        for row in selected[:4]:
            print(
                f"  {row['feature']:28s} "
                f"rho={row['spearman_rho']:+.3f}"
            )

    print("\nSaved:")
    print(FEATURE_OUT)
    print(FAILURE_OUT)
    print(ASSOCIATION_OUT)
    print(REPORT_OUT)
    print("\nE3.5 COMPLETE")


if __name__ == "__main__":
    main()
