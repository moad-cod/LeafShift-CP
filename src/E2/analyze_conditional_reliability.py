
from pathlib import Path
from collections import defaultdict
import csv
import json
import math

import numpy as np

# ============================================================
# E2.3 — CONDITIONAL UNCERTAINTY RELIABILITY
#
# Input: E1.0 manifest and E2.1 held-out intervals.
#
# No models are fitted.
# No calibration thresholds are changed.
# No target-derived subgroup definitions.
#
# Subgroups use information available by cutoff time.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "data/processed/forecasting/"
    "forecast_windows.csv"
)
PREDICTIONS = (
    ROOT / "experiments/E2/tables/"
    "E2_1_interval_predictions.csv"
)

BASE = ROOT / "experiments/E2"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

GROUP_METRICS = (
    TABLES / "E2_3_conditional_genotype_metrics.csv"
)
SUMMARY = (
    TABLES / "E2_3_conditional_summary.csv"
)
REPORT = (
    REPORTS / "E2_3_conditional_reliability.md"
)

HORIZONS = (24, 48)
METHODS = ("U0_pooled", "U1_balanced")
NOMINAL = 0.90
ALPHA = 0.10

# Fixed exploratory thresholds; not selected
# by inspecting test coverage.
SIZE_SMALL_MAX = 500.0
SIZE_MEDIUM_MAX = 2000.0

GROWTH_STABLE_THRESHOLD = 0.10

CV_LOW_MAX = 0.10
CV_MEDIUM_MAX = 0.30

# For reporting a genotype's subgroup result.
MIN_WINDOWS_PER_GENOTYPE_GROUP = 30
MIN_PLANTS_PER_GENOTYPE_GROUP = 2


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No output rows: {path}")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def classify_history(areas, skipped_frames):
    if len(areas) != 6:
        raise ValueError(
            "Expected six historical leaf areas"
        )

    a = np.asarray(areas, dtype=np.float64)

    if not np.isfinite(a).all() or np.any(a <= 0):
        raise ValueError("Invalid historical area")

    last = float(a[-1])
    first = float(a[0])

    relative_growth = (last - first) / first
    cv = float(np.std(a) / np.mean(a))

    if last < SIZE_SMALL_MAX:
        size = "small_lt500"
    elif last < SIZE_MEDIUM_MAX:
        size = "medium_500to1999"
    else:
        size = "large_ge2000"

    if relative_growth < -GROWTH_STABLE_THRESHOLD:
        growth = "decreasing"
    elif relative_growth > GROWTH_STABLE_THRESHOLD:
        growth = "increasing"
    else:
        growth = "approximately_stable"

    if cv < CV_LOW_MAX:
        variability = "low_lt0.10"
    elif cv < CV_MEDIUM_MAX:
        variability = "moderate_0.10to0.30"
    else:
        variability = "high_ge0.30"

    history_gap = (
        "with_skipped_frames"
        if skipped_frames > 0
        else "no_skipped_frames"
    )

    return {
        "leaf_size": size,
        "relative_growth": growth,
        "historical_variability": variability,
        "history_gap": history_gap,
    }


def load_groups():
    """
    One set of historical group labels per window.
    Target values are not accessed.
    """
    groups = {}

    with MANIFEST.open(
        newline="", encoding="utf-8"
    ) as file:
        for row in csv.DictReader(file):
            wid = row["window_id"]

            if wid in groups:
                raise ValueError(
                    f"Duplicate manifest ID: {wid}"
                )

            areas = json.loads(
                row["history_areas_pixels"]
            )

            categories = classify_history(
                areas,
                int(row["history_skipped_image_frames"])
            )

            groups[wid] = {
                "genotype": row["genotype"],
                "plant_id": row["plant_id"],
                "horizon": int(
                    row["horizon_hours"]
                ),
                "categories": categories,
            }

    if len(groups) != 228645:
        raise ValueError(
            f"Unexpected manifest size: {len(groups)}"
        )

    return groups


def accumulator():
    return {
        "n": 0,
        "covered": 0,
        "width_sum": 0.0,
        "score_sum": 0.0,
        "mae_sum": 0.0,
    }


def add_metrics(acc, target, prediction, lower, upper):
    covered = lower <= target <= upper

    score = (
        upper - lower
        + (2 / ALPHA) * max(lower - target, 0.0)
        + (2 / ALPHA) * max(target - upper, 0.0)
    )

    acc["n"] += 1
    acc["covered"] += int(covered)
    acc["width_sum"] += upper - lower
    acc["score_sum"] += score
    acc["mae_sum"] += abs(target - prediction)

    return int(covered)


def main():
    print("=" * 68)
    print("E2.3 — CONDITIONAL RELIABILITY ANALYSIS")
    print("=" * 68)

    groups = load_groups()

    # Key:
    # (horizon, dimension, category, genotype,
    #  plant, method)
    aggregate = defaultdict(accumulator)
    seen = set()
    count = 0

    with PREDICTIONS.open(
        newline="", encoding="utf-8"
    ) as file:
        for row in csv.DictReader(file):
            wid = row["window_id"]
            method = row["method"]
            horizon = int(row["horizon_hours"])
            genotype = row["genotype"]
            plant = row["plant_id"]

            if wid not in groups:
                raise ValueError(
                    f"Unknown prediction window: {wid}"
                )

            if method not in METHODS:
                raise ValueError(
                    f"Unknown calibration method: {method}"
                )

            if horizon not in HORIZONS:
                raise ValueError(
                    f"Unexpected horizon: {horizon}"
                )

            key_id = (wid, method)

            if key_id in seen:
                raise ValueError(
                    f"Duplicate prediction: {key_id}"
                )

            seen.add(key_id)

            info = groups[wid]

            if (
                info["genotype"] != genotype
                or info["plant_id"] != plant
                or info["horizon"] != horizon
            ):
                raise ValueError(
                    f"Manifest/prediction mismatch: {wid}"
                )

            target = float(row["target_area_pixels"])
            prediction = float(
                row["predicted_area_pixels"]
            )
            lower = float(row["lower_pixels"])
            upper = float(row["upper_pixels"])

            if not all(
                math.isfinite(x)
                for x in (
                    target, prediction, lower, upper
                )
            ):
                raise ValueError(
                    "Nonfinite interval values"
                )

            if lower < 0 or upper < lower:
                raise ValueError("Invalid prediction interval")

            actual_coverage = int(
                lower <= target <= upper
            )

            if actual_coverage != int(row["covered"]):
                raise ValueError(
                    "Saved coverage flag is inconsistent"
                )

            for dimension, category in (
                info["categories"].items()
            ):
                key = (
                    horizon,
                    dimension,
                    category,
                    genotype,
                    plant,
                    method,
                )

                add_metrics(
                    aggregate[key],
                    target,
                    prediction,
                    lower,
                    upper,
                )

            count += 1

    expected_count = len(groups) * len(METHODS)

    if count != expected_count:
        raise ValueError(
            f"Expected {expected_count} prediction rows, "
            f"found {count}"
        )

    # Aggregate windows within a plant, then
    # plants within a genotype for each subgroup.
    by_genotype = defaultdict(list)

    for key, acc in aggregate.items():
        (
            horizon, dimension, category,
            genotype, plant, method
        ) = key

        n = acc["n"]

        by_genotype[
            (
                horizon, dimension, category,
                genotype, method
            )
        ].append({
            "plant_id": plant,
            "windows": n,
            "coverage": acc["covered"] / n,
            "width": acc["width_sum"] / n,
            "score": acc["score_sum"] / n,
            "mae": acc["mae_sum"] / n,
        })

    genotype_rows = []

    for key, plant_rows in sorted(
        by_genotype.items()
    ):
        (
            horizon, dimension, category,
            genotype, method
        ) = key

        genotype_rows.append({
            "horizon_hours": horizon,
            "dimension": dimension,
            "category": category,
            "genotype": genotype,
            "method": method,
            "plants": len(plant_rows),
            "windows": sum(
                p["windows"] for p in plant_rows
            ),
            "plant_mean_coverage": float(
                np.mean([
                    p["coverage"] for p in plant_rows
                ])
            ),
            "plant_mean_width": float(
                np.mean([
                    p["width"] for p in plant_rows
                ])
            ),
            "plant_mean_interval_score": float(
                np.mean([
                    p["score"] for p in plant_rows
                ])
            ),
            "plant_mean_mae": float(
                np.mean([
                    p["mae"] for p in plant_rows
                ])
            ),
        })

    write_csv(GROUP_METRICS, genotype_rows)

    # A genotype contributes to a category-level
    # macro average only when that subgroup has at
    # least two plants and 30 windows.
    #
    # This eligibility criterion describes support,
    # not prediction quality.
    by_group = defaultdict(dict)

    for row in genotype_rows:
        key = (
            row["horizon_hours"],
            row["dimension"],
            row["category"],
            row["method"],
        )

        if (
            row["plants"] >= MIN_PLANTS_PER_GENOTYPE_GROUP
            and row["windows"]
            >= MIN_WINDOWS_PER_GENOTYPE_GROUP
        ):
            by_group[key][row["genotype"]] = row

    summary_rows = []

    # Pair U0 and U1 on the same eligible genotypes.
    group_keys = sorted({
        key[:3] for key in by_group
    })

    for horizon, dimension, category in group_keys:
        u0 = by_group.get(
            (horizon, dimension, category, "U0_pooled"),
            {}
        )
        u1 = by_group.get(
            (horizon, dimension, category, "U1_balanced"),
            {}
        )

        shared = sorted(set(u0) & set(u1))

        for method, data in (
            ("U0_pooled", u0),
            ("U1_balanced", u1),
        ):
            rows = [data[g] for g in shared]

            if not rows:
                continue

            def avg(field):
                return float(np.mean([
                    r[field] for r in rows
                ]))

            summary_rows.append({
                "horizon_hours": horizon,
                "dimension": dimension,
                "category": category,
                "method": method,
                "eligible_genotypes": len(rows),
                "sum_eligible_plants": sum(
                    r["plants"] for r in rows
                ),
                "sum_eligible_windows": sum(
                    r["windows"] for r in rows
                ),
                "macro_coverage": avg(
                    "plant_mean_coverage"
                ),
                "coverage_gap_pp": (
                    100 * (
                        avg("plant_mean_coverage")
                        - NOMINAL
                    )
                ),
                "macro_width_pixels": avg(
                    "plant_mean_width"
                ),
                "macro_interval_score": avg(
                    "plant_mean_interval_score"
                ),
                "macro_mae_pixels": avg(
                    "plant_mean_mae"
                ),
                "genotypes_below_90": sum(
                    r["plant_mean_coverage"] < NOMINAL
                    for r in rows
                ),
                "genotypes_below_80": sum(
                    r["plant_mean_coverage"] < 0.80
                    for r in rows
                ),
            })

    write_csv(SUMMARY, summary_rows)

    # ----------------------------------------------------
    # Representative figures for the four dimensions.
    # ----------------------------------------------------

    try:
        import matplotlib.pyplot as plt

        dimensions = [
            "leaf_size",
            "relative_growth",
            "historical_variability",
            "history_gap",
        ]

        for horizon in HORIZONS:
            for dimension in dimensions:
                rows = [
                    r for r in summary_rows
                    if (
                        r["horizon_hours"] == horizon
                        and r["dimension"] == dimension
                    )
                ]

                categories = sorted({
                    r["category"] for r in rows
                })

                if not categories:
                    continue

                lookup = {
                    (r["category"], r["method"]): r
                    for r in rows
                }

                x = np.arange(len(categories))
                width = 0.38

                fig, ax = plt.subplots(
                    figsize=(max(8, len(categories) * 2.5), 5)
                )

                for i, method in enumerate(METHODS):
                    values = [
                        lookup.get(
                            (cat, method), {}
                        ).get("macro_coverage", np.nan)
                        for cat in categories
                    ]

                    ax.bar(
                        x + (i - 0.5) * width,
                        values,
                        width,
                        label=method,
                    )

                ax.axhline(
                    NOMINAL,
                    linestyle="--",
                    label="Nominal 90%",
                )

                ax.set_xticks(x)
                ax.set_xticklabels(
                    categories,
                    rotation=20,
                    ha="right",
                )
                ax.set_ylim(0, 1.05)
                ax.set_ylabel(
                    "Genotype-macro plant coverage"
                )
                ax.set_title(
                    f"E2.3 — {horizon}h: {dimension}"
                )
                ax.legend()
                fig.tight_layout()

                fig.savefig(
                    FIGURES
                    / f"E2_3_{dimension}_{horizon}h.png",
                    dpi=200,
                )
                plt.close(fig)

    except ImportError:
        print(
            "Matplotlib unavailable: figures skipped."
        )

    # ----------------------------------------------------
    # Report
    # ----------------------------------------------------

    lines = [
        "# E2.3 — Conditional Reliability",
        "",
        "## Protocol",
        "",
        "- Existing E2.1 held-out intervals only.",
        "- No new training or recalibration.",
        "- All categories use historical observations.",
        "- Subgroup boundaries were specified before "
        "this experiment.",
        "- Windows are averaged within plants, "
        "then plants within genotypes.",
        "- Only genotype subgroups with at least two "
        "plants and 30 windows enter macro summaries.",
        "- Subgroup metrics are descriptive; no "
        "conditional coverage guarantees.",
        "",
        "## Results",
        "",
        "| Horizon | Dimension | Category | Method | "
        "Genotypes | Coverage | Width (px) | Score |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]

    for r in summary_rows:
        lines.append(
            f"| {r['horizon_hours']}h "
            f"| {r['dimension']} "
            f"| {r['category']} "
            f"| {r['method']} "
            f"| {r['eligible_genotypes']} "
            f"| {r['macro_coverage']:.3f} "
            f"| {r['macro_width_pixels']:.2f} "
            f"| {r['macro_interval_score']:.2f} |"
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Categories use exploratory fixed thresholds.",
        "- Different categories can contain different "
        "numbers of eligible genotypes.",
        "- Subgroup coverage estimates can be unstable.",
        "- These are visible-area targets, not true "
        "anatomical leaf areas.",
        "- Historical color tracking may include "
        "unresolved annotation inconsistencies.",
        "- Genotype and acquisition schedules may "
        "remain confounded.",
        "- Repeated overlapping windows are correlated.",
        "",
        "## Next",
        "",
        "E2.4 — Validate findings across the four "
        "predefined calibration genotype assignments.",
    ])

    REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print(f"\nAnalyzed prediction rows: {count:,}")
    print(f"Genotype-group rows: {len(genotype_rows):,}")
    print(f"Summary rows: {len(summary_rows)}")

    for horizon in HORIZONS:
        print(f"\n{horizon}h — U1 conditional coverage:")

        for r in summary_rows:
            if (
                r["horizon_hours"] == horizon
                and r["method"] == "U1_balanced"
            ):
                print(
                    f"  {r['dimension']:23s} "
                    f"{r['category']:25s} "
                    f"coverage={r['macro_coverage']:.3f} "
                    f"genotypes={r['eligible_genotypes']}"
                )

    print("\nSaved:")
    print(GROUP_METRICS)
    print(SUMMARY)
    print(REPORT)
    print("\nE2.3 COMPLETE")


if __name__ == "__main__":
    main()
