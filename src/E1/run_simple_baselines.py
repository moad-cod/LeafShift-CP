
from pathlib import Path
from collections import defaultdict
import csv
import json
import math
import statistics

# ============================================================
# E1.2 — PERSISTENCE AND LINEAR FORECASTING BASELINES
#
# B0: Persistence
# B1: Time-aware least-squares linear extrapolation
#
# Models have no fitted cross-plant parameters.
# Every prediction uses only the six historical
# observations and the requested forecast horizon.
#
# No raw images or E1.0 manifest are modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "data/processed/forecasting"
    / "forecast_windows.csv"
)

FOLD_CONFIG = (
    ROOT / "configs/E1_1_genotype_folds.json"
)

EXP = ROOT / "experiments/E1"
TABLES = EXP / "tables"
FIGURES = EXP / "figures"
REPORTS = EXP / "reports"

for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

PREDICTIONS = TABLES / "E1_2_baseline_predictions.csv"
PLANT_METRICS = TABLES / "E1_2_plant_metrics.csv"
GENOTYPE_METRICS = TABLES / "E1_2_genotype_metrics.csv"
SUMMARY = TABLES / "E1_2_summary.csv"
REPORT = REPORTS / "E1_2_simple_baselines.md"

MODELS = ("B0_persistence", "B1_linear")
HORIZONS = (24, 48)

# Sensitivity comparisons are prespecified.
SUBSETS = (
    "ALL",
    "EXCLUDE_PALETTE_FLAG",
    "CONTIGUOUS_HISTORY",
    "BOTH_QC_FILTERS",
)

PREDICTION_COLUMNS = [
    "window_id",
    "genotype",
    "plant_id",
    "leaf_id",
    "horizon_hours",
    "target_area_pixels",
    "B0_persistence",
    "B1_linear",
    "flagged_palette_plant",
    "history_skipped_image_frames",
]


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def linear_forecast(times, areas, horizon):
    """
    Least-squares fit y = intercept + slope * t.

    Historical times are relative to the cutoff,
    with the final historical observation at t=0.

    The future is predicted at the nominal horizon,
    not at the observed target timestamp.

    Negative predicted area is clipped to zero.
    """
    n = len(times)

    mean_t = sum(times) / n
    mean_y = sum(areas) / n

    denominator = sum(
        (t - mean_t) ** 2 for t in times
    )

    if denominator <= 1e-12:
        return max(0.0, areas[-1])

    numerator = sum(
        (t - mean_t) * (y - mean_y)
        for t, y in zip(times, areas)
    )

    slope = numerator / denominator
    intercept = mean_y - slope * mean_t

    return max(
        0.0,
        intercept + slope * horizon
    )


class ErrorAccumulator:
    def __init__(self):
        self.count = 0
        self.absolute_error_sum = 0.0
        self.squared_error_sum = 0.0
        self.target_sum = 0.0

    def add(self, target, prediction):
        error = prediction - target
        self.count += 1
        self.absolute_error_sum += abs(error)
        self.squared_error_sum += error * error
        self.target_sum += target

    @property
    def mae(self):
        return (
            self.absolute_error_sum / self.count
            if self.count else float("nan")
        )

    @property
    def rmse(self):
        return (
            math.sqrt(
                self.squared_error_sum / self.count
            )
            if self.count else float("nan")
        )


def belongs_to_subset(row, subset):
    flagged = int(row["flagged_palette_plant"]) > 0
    skipped = (
        int(row["history_skipped_image_frames"]) > 0
    )

    if subset == "ALL":
        return True

    if subset == "EXCLUDE_PALETTE_FLAG":
        return not flagged

    if subset == "CONTIGUOUS_HISTORY":
        return not skipped

    if subset == "BOTH_QC_FILTERS":
        return not flagged and not skipped

    raise ValueError(subset)


def main():
    print("=" * 68)
    print("E1.2 — SIMPLE FORECASTING BASELINES")
    print("=" * 68)

    with FOLD_CONFIG.open(encoding="utf-8") as f:
        fold_config = json.load(f)

    folds = fold_config["folds"]
    outer_test_genotypes = [
        fold["test_genotype"] for fold in folds
    ]

    if len(folds) != 21:
        raise RuntimeError(
            "Expected 21 leave-one-genotype-out folds."
        )

    if len(set(outer_test_genotypes)) != 21:
        raise RuntimeError(
            "Duplicate outer-test genotype."
        )

    # Since these two baselines have no across-plant
    # training parameters, each genotype is evaluated
    # directly as its corresponding LOGO test fold.
    test_fold_id = {
        genotype: fold["outer_fold"]
        for genotype, fold in [
            (f["test_genotype"], f) for f in folds
        ]
    }

    # One accumulator per plant, horizon, model, subset.
    plant_errors = defaultdict(ErrorAccumulator)

    # Pooled window errors are included for reference,
    # but not used as the primary study metric.
    pooled_errors = defaultdict(ErrorAccumulator)

    total_windows = 0
    seen_genotypes = set()

    examples = []

    with MANIFEST.open(
        newline="", encoding="utf-8"
    ) as src, PREDICTIONS.open(
        "w", newline="", encoding="utf-8"
    ) as dst:

        reader = csv.DictReader(src)
        writer = csv.DictWriter(
            dst, fieldnames=PREDICTION_COLUMNS
        )
        writer.writeheader()

        for row in reader:
            genotype = row["genotype"]

            if genotype not in test_fold_id:
                raise ValueError(
                    f"Missing LOGO fold: {genotype}"
                )

            plant = row["plant_id"]
            horizon = int(row["horizon_hours"])
            target = float(row["target_area_pixels"])

            areas = [
                float(x)
                for x in json.loads(
                    row["history_areas_pixels"]
                )
            ]
            times = [
                float(x)
                for x in json.loads(
                    row["history_relative_hours"]
                )
            ]

            if len(times) != 6 or len(areas) != 6:
                raise ValueError(
                    "Expected six historical observations."
                )

            if horizon not in HORIZONS:
                raise ValueError(
                    f"Unexpected horizon: {horizon}"
                )

            if abs(times[-1]) > 1e-5:
                raise ValueError(
                    "Last historical time is not the cutoff."
                )

            b0 = max(0.0, areas[-1])
            b1 = linear_forecast(
                times, areas, horizon
            )

            predictions = {
                "B0_persistence": b0,
                "B1_linear": b1,
            }

            writer.writerow({
                "window_id": row["window_id"],
                "genotype": genotype,
                "plant_id": plant,
                "leaf_id": row["leaf_id"],
                "horizon_hours": horizon,
                "target_area_pixels": target,
                "B0_persistence": round(b0, 5),
                "B1_linear": round(b1, 5),
                "flagged_palette_plant": (
                    row["flagged_palette_plant"]
                ),
                "history_skipped_image_frames": (
                    row["history_skipped_image_frames"]
                ),
            })

            for subset in SUBSETS:
                if not belongs_to_subset(row, subset):
                    continue

                for model, prediction in predictions.items():
                    key = (
                        subset, horizon, genotype,
                        plant, model
                    )
                    plant_errors[key].add(
                        target, prediction
                    )

                    pooled_errors[
                        (subset, horizon, model)
                    ].add(target, prediction)

            if len(examples) < 1500:
                examples.append({
                    "genotype": genotype,
                    "plant_id": plant,
                    "horizon": horizon,
                    "target": target,
                    "B0": b0,
                    "B1": b1,
                })

            total_windows += 1
            seen_genotypes.add(genotype)

    if total_windows != 228645:
        raise RuntimeError(
            f"Unexpected window count: {total_windows}"
        )

    if set(seen_genotypes) != set(test_fold_id):
        raise RuntimeError(
            "Not all outer-test genotypes were evaluated."
        )

    # ----------------------------------------------------
    # Plant-level metrics
    # ----------------------------------------------------

    plant_rows = []

    for key, acc in sorted(plant_errors.items()):
        subset, horizon, genotype, plant, model = key

        plant_rows.append({
            "subset": subset,
            "horizon_hours": horizon,
            "genotype": genotype,
            "outer_fold": test_fold_id[genotype],
            "plant_id": plant,
            "model": model,
            "windows": acc.count,
            "mae_pixels": round(acc.mae, 5),
            "rmse_pixels": round(acc.rmse, 5),
        })

    save_csv(
        PLANT_METRICS,
        plant_rows,
        [
            "subset", "horizon_hours", "genotype",
            "outer_fold", "plant_id", "model",
            "windows", "mae_pixels", "rmse_pixels",
        ],
    )

    # ----------------------------------------------------
    # Genotype-level metrics
    #
    # Equal-weight plants, NOT equal-weight windows.
    # ----------------------------------------------------

    genotype_groups = defaultdict(list)

    for row in plant_rows:
        key = (
            row["subset"],
            row["horizon_hours"],
            row["genotype"],
            row["model"],
        )
        genotype_groups[key].append(row)

    genotype_rows = []

    for key, rows in sorted(genotype_groups.items()):
        subset, horizon, genotype, model = key

        genotype_rows.append({
            "subset": subset,
            "horizon_hours": horizon,
            "genotype": genotype,
            "outer_fold": test_fold_id[genotype],
            "model": model,
            "plants": len(rows),
            "windows": sum(r["windows"] for r in rows),
            "plant_mean_mae_pixels": round(
                statistics.mean(
                    r["mae_pixels"] for r in rows
                ), 5
            ),
            "plant_mean_rmse_pixels": round(
                statistics.mean(
                    r["rmse_pixels"] for r in rows
                ), 5
            ),
        })

    save_csv(
        GENOTYPE_METRICS,
        genotype_rows,
        [
            "subset", "horizon_hours", "genotype",
            "outer_fold", "model", "plants",
            "windows", "plant_mean_mae_pixels",
            "plant_mean_rmse_pixels",
        ],
    )

    # ----------------------------------------------------
    # Macro-genotype summary.
    # Both genotype and plant get equal weighting
    # at their respective aggregation stage.
    # ----------------------------------------------------

    summary_groups = defaultdict(list)

    for row in genotype_rows:
        key = (
            row["subset"],
            row["horizon_hours"],
            row["model"],
        )
        summary_groups[key].append(row)

    summary_rows = []

    for key, rows in sorted(summary_groups.items()):
        subset, horizon, model = key
        pooled = pooled_errors[(subset, horizon, model)]

        summary_rows.append({
            "subset": subset,
            "horizon_hours": horizon,
            "model": model,
            "genotypes": len(rows),
            "plants": sum(r["plants"] for r in rows),
            "windows": sum(r["windows"] for r in rows),
            "macro_genotype_plant_mae_pixels": round(
                statistics.mean(
                    r["plant_mean_mae_pixels"]
                    for r in rows
                ), 5
            ),
            "mean_plant_rmse_pixels": round(
                statistics.mean(
                    r["plant_mean_rmse_pixels"]
                    for r in rows
                ), 5
            ),
            "pooled_window_mae_pixels": round(
                pooled.mae, 5
            ),
            "pooled_window_rmse_pixels": round(
                pooled.rmse, 5
            ),
        })

    save_csv(
        SUMMARY,
        summary_rows,
        [
            "subset", "horizon_hours", "model",
            "genotypes", "plants", "windows",
            "macro_genotype_plant_mae_pixels",
            "mean_plant_rmse_pixels",
            "pooled_window_mae_pixels",
            "pooled_window_rmse_pixels",
        ],
    )

    # ----------------------------------------------------
    # Representative figure — per-genotype MAE.
    # ----------------------------------------------------

    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(
            2, 1, figsize=(12, 12)
        )

        for ax, horizon in zip(axes, HORIZONS):
            rows = [
                r for r in genotype_rows
                if (
                    r["subset"] == "ALL"
                    and r["horizon_hours"] == horizon
                )
            ]

            genotypes = sorted({
                r["genotype"] for r in rows
            })

            b0 = {
                r["genotype"]: r[
                    "plant_mean_mae_pixels"
                ]
                for r in rows
                if r["model"] == "B0_persistence"
            }

            b1 = {
                r["genotype"]: r[
                    "plant_mean_mae_pixels"
                ]
                for r in rows
                if r["model"] == "B1_linear"
            }

            x = list(range(len(genotypes)))

            ax.plot(
                x,
                [b0[g] for g in genotypes],
                marker="o",
                label="B0 Persistence",
            )

            ax.plot(
                x,
                [b1[g] for g in genotypes],
                marker="s",
                label="B1 Linear",
            )

            ax.set_xticks(x)
            ax.set_xticklabels(
                genotypes, rotation=65, ha="right"
            )
            ax.set_ylabel(
                "Plant-averaged MAE (pixels)"
            )
            ax.set_title(
                f"{horizon}h — Held-out genotype errors"
            )
            ax.legend()
            ax.grid(axis="y", alpha=0.25)

        fig.tight_layout()
        fig.savefig(
            FIGURES / "E1_2_baseline_genotype_mae.png",
            dpi=200,
        )
        plt.close(fig)

    except ImportError:
        print(
            "Matplotlib not installed; figure skipped."
        )

    # ----------------------------------------------------
    # Markdown report
    # ----------------------------------------------------

    report = [
        "# E1.2 — Simple Forecasting Baselines",
        "",
        "## Models",
        "",
        "- B0: Last-observation persistence.",
        "- B1: Time-aware least-squares extrapolation.",
        "- Both models predict the nominal horizon.",
        "- Negative predicted area is clipped to zero.",
        "- No population-trained parameters are used.",
        "",
        "## Main results: ALL windows",
        "",
        "| Horizon | Model | Genotypes | Plants | "
        "Macro plant-averaged MAE (px) |",
        "|---|---|---:|---:|---:|",
    ]

    for r in summary_rows:
        if r["subset"] == "ALL":
            report.append(
                f"| {r['horizon_hours']}h "
                f"| {r['model']} "
                f"| {r['genotypes']} "
                f"| {r['plants']} "
                f"| {r['macro_genotype_plant_mae_pixels']:.2f} |"
            )

    report.extend([
        "",
        "## Methodological notes",
        "",
        "These are deterministic per-window baselines.",
        "No model is fitted across genotypes.",
        "",
        "For each outer evaluation, the corresponding "
        "genotype is considered the test genotype.",
        "",
        "All inputs are restricted to historical observations.",
        "",
        "Genotype-level MAE is computed by first averaging "
        "window absolute errors within each plant, then "
        "averaging across plants.",
        "",
        "The main summary averages genotype-level results "
        "with equal weight.",
        "",
        "The reported mean plant RMSE is not mathematically "
        "identical to pooled RMSE.",
        "",
        "Sensitivity subsets are descriptive. They do not "
        "establish that flagged observations are erroneous.",
        "",
        "## Limitations",
        "",
        "Visible projected leaf area is not true leaf area.",
        "The target is observed within +/-3h of nominal horizon.",
        "Only observable future targets contribute windows.",
        "Some individual color tracks may be inconsistent.",
        "Overlapping windows are highly correlated.",
        "",
        "## Next",
        "",
        "Compare per-genotype B0/B1 errors, inspect large "
        "failures, and determine whether trained models "
        "should be introduced under LOGO validation.",
    ])

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    # Terminal output
    print("\n" + "=" * 68)
    print("E1.2 — MACRO GENOTYPE RESULTS")
    print("=" * 68)

    for horizon in HORIZONS:
        print(f"\n{horizon}-hour forecasting:")

        for row in summary_rows:
            if (
                row["subset"] == "ALL"
                and row["horizon_hours"] == horizon
            ):
                print(
                    f"  {row['model']:16s} "
                    f"MAE={row['macro_genotype_plant_mae_pixels']:.2f} px "
                    f"pooled_RMSE={row['pooled_window_rmse_pixels']:.2f} px"
                )

    print("\nQC sensitivity — macro genotype MAE:")

    for row in summary_rows:
        print(
            f"  {row['subset']:22s} "
            f"{row['horizon_hours']:2d}h "
            f"{row['model']:16s} "
            f"{row['macro_genotype_plant_mae_pixels']:.2f} px "
            f"({row['windows']:,} windows)"
        )

    print("\nSaved:")
    print(PREDICTIONS)
    print(PLANT_METRICS)
    print(GENOTYPE_METRICS)
    print(SUMMARY)
    print(REPORT)

    print("\nE1.2 complete.")


if __name__ == "__main__":
    main()