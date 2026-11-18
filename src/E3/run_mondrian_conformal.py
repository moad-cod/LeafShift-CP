
from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# Reuse E3.1 feature engineering, model configuration,
# calibration quantiles, and evaluation metrics.
import run_normalized_conformal as baseline


# ============================================================
# E3.3 — MONDRIAN CONFORMAL CALIBRATION
#
# U6: Pooled residual quantile per leaf-size category
# U7: Genotype/plant-balanced quantile per category
#
# 21 outer genotypes x 4 calibration assignments
# x 2 forecasting horizons = 168 fitted models.
#
# No held-out genotype used in model fitting
# or calibration.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

FOLDS_PATH = ROOT / "configs/E1_1_genotype_folds.json"

BASE = ROOT / "experiments/E3"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

for directory in (TABLES, FIGURES, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

FOLD_OUT = TABLES / "E3_3_fold_metrics.csv"
SIZE_OUT = TABLES / "E3_3_size_metrics.csv"
SUMMARY_OUT = TABLES / "E3_3_summary.csv"
SIZE_SUMMARY_OUT = TABLES / "E3_3_size_summary.csv"
REPORT_OUT = REPORTS / "E3_3_mondrian_conformal.md"

HORIZONS = (24, 48)

METHODS = (
    "U6_Mondrian_pooled",
    "U7_Mondrian_balanced",
)

SIZES = (
    "small_lt500",
    "medium_500to1999",
    "large_ge2000",
)

NOMINAL = 0.90
EXPECTED_OUTER_FOLDS = 21
EXPECTED_INNER_FOLDS = 4


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to save: {path}")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=list(rows[0].keys()),
        )
        writer.writeheader()
        writer.writerows(rows)


def validate_split(d, outer, inner):
    fit_g = set(inner["train_genotypes"])
    cal_g = set(inner["validation_genotypes"])
    test_g = outer["test_genotype"]

    if len(fit_g) != 15:
        raise ValueError("Expected 15 fitting genotypes")

    if len(cal_g) != 5:
        raise ValueError("Expected 5 calibration genotypes")

    if fit_g & cal_g:
        raise ValueError("Fitting/calibration overlap")

    if test_g in fit_g or test_g in cal_g:
        raise ValueError("Held-out genotype leakage")

    fit = np.isin(d["g"], list(fit_g))
    cal = np.isin(d["g"], list(cal_g))
    test = d["g"] == test_g

    if not (fit.any() and cal.any() and test.any()):
        raise ValueError("Empty split partition")

    if set(d["g"][fit]) != fit_g:
        raise ValueError("Incomplete fitting genotypes")

    if set(d["g"][cal]) != cal_g:
        raise ValueError("Incomplete calibration genotypes")

    if set(d["g"][test]) != {test_g}:
        raise ValueError("Incorrect held-out genotype")

    if (
        np.any(fit & cal)
        or np.any(fit & test)
        or np.any(cal & test)
    ):
        raise ValueError("Split membership overlap")

    return fit, cal, test


def compute_category_thresholds(
    residuals,
    sizes,
    genotypes,
    plants,
):
    thresholds = {}
    support = {}

    for category in SIZES:
        mask = sizes == category

        n_windows = int(np.sum(mask))
        n_genotypes = len(np.unique(genotypes[mask]))
        n_plants = len(np.unique(plants[mask]))

        if n_windows == 0:
            raise ValueError(
                f"Empty calibration category: {category}"
            )

        if n_genotypes < 2 or n_plants < 2:
            raise ValueError(
                f"Insufficient calibration support: "
                f"{category}"
            )

        category_residuals = residuals[mask]

        # U6: Conventional finite-sample pooled
        # conformal residual quantile.
        q_pooled = baseline.pooled_quantile(
            category_residuals
        )

        # U7: Equal total weight per genotype,
        # per plant within genotype, and per window
        # within plant.
        category_weights = baseline.balanced_weights(
            genotypes[mask],
            plants[mask],
        )

        q_balanced = baseline.balanced_quantile(
            category_residuals,
            category_weights,
        )

        thresholds[category] = {
            "U6_Mondrian_pooled": float(q_pooled),
            "U7_Mondrian_balanced": float(q_balanced),
        }

        support[category] = {
            "calibration_windows": n_windows,
            "calibration_genotypes": n_genotypes,
            "calibration_plants": n_plants,
        }

    return thresholds, support


def evaluate(d, outer, inner, horizon):
    fit, cal, test = validate_split(
        d, outer, inner
    )

    test_genotype = outer["test_genotype"]

    # ----------------------------------------------------
    # 1. Fit fixed B3 point forecasting model.
    # ----------------------------------------------------

    train_weights = baseline.balanced_weights(
        d["g"][fit],
        d["p"][fit],
    )

    model = HistGradientBoostingRegressor(
        **baseline.PARAMS
    )

    model.fit(
        d["X"][fit],
        d["y"][fit],
        sample_weight=train_weights * len(train_weights),
    )

    # ----------------------------------------------------
    # 2. Calibration predictions and residuals.
    # ----------------------------------------------------

    cal_predictions = np.maximum(
        model.predict(d["X"][cal]),
        0.0,
    )

    residuals = np.abs(
        d["y"][cal] - cal_predictions
    )

    thresholds, support = compute_category_thresholds(
        residuals=residuals,
        sizes=d["size"][cal],
        genotypes=d["g"][cal],
        plants=d["p"][cal],
    )

    # ----------------------------------------------------
    # 3. Evaluate on entirely unseen genotype.
    # ----------------------------------------------------

    test_predictions = np.maximum(
        model.predict(d["X"][test]),
        0.0,
    )

    y_test = d["y"][test]
    plants_test = d["p"][test]
    sizes_test = d["size"][test]

    common = {
        "horizon_hours": horizon,
        "outer_fold": int(outer["outer_fold"]),
        "inner_fold": int(inner["inner_fold"]),
        "test_genotype": test_genotype,
    }

    fold_rows = []
    size_rows = []

    for method in METHODS:
        halfwidth = np.asarray(
            [
                thresholds[category][method]
                for category in sizes_test
            ],
            dtype=np.float64,
        )

        lower = np.maximum(
            0.0,
            test_predictions - halfwidth,
        )

        upper = test_predictions + halfwidth

        overall_metrics = baseline.plant_aggregated_metrics(
            y_test,
            test_predictions,
            lower,
            upper,
            plants_test,
        )

        fold_rows.append({
            **common,
            "method": method,
            **overall_metrics,
        })

        for category in SIZES:
            mask = sizes_test == category

            if not np.any(mask):
                continue

            category_metrics = (
                baseline.plant_aggregated_metrics(
                    y_test[mask],
                    test_predictions[mask],
                    lower[mask],
                    upper[mask],
                    plants_test[mask],
                )
            )

            size_rows.append({
                **common,
                "method": method,
                "leaf_size": category,
                "calibration_threshold_pixels": (
                    thresholds[category][method]
                ),
                **support[category],
                **category_metrics,
            })

    return fold_rows, size_rows


def summarize(fold_rows, size_rows):
    overall = []
    conditional = []

    # ----------------------------------------------------
    # Overall aggregation:
    # windows -> plants -> calibration assignments
    # -> genotypes.
    # ----------------------------------------------------

    for horizon in HORIZONS:
        for method in METHODS:
            rows = [
                r for r in fold_rows
                if (
                    r["horizon_hours"] == horizon
                    and r["method"] == method
                )
            ]

            by_genotype = defaultdict(list)

            for row in rows:
                by_genotype[row["test_genotype"]].append(
                    row
                )

            if len(by_genotype) != EXPECTED_OUTER_FOLDS:
                raise ValueError(
                    "Expected 21 held-out genotypes"
                )

            for genotype, items in by_genotype.items():
                assignments = sorted(
                    r["inner_fold"] for r in items
                )

                if assignments != [1, 2, 3, 4]:
                    raise ValueError(
                        f"Incomplete assignments: {genotype}"
                    )

            def macro_average(field):
                return float(
                    np.mean([
                        np.mean([
                            item[field] for item in items
                        ])
                        for items in by_genotype.values()
                    ])
                )

            per_genotype_coverage = [
                float(np.mean([
                    item["coverage"] for item in items
                ]))
                for items in by_genotype.values()
            ]

            overall.append({
                "horizon_hours": horizon,
                "method": method,
                "macro_coverage": macro_average("coverage"),
                "macro_width": macro_average("width"),
                "macro_interval_score": macro_average("score"),
                "macro_mae": macro_average("mae"),
                "genotypes_below_90": sum(
                    coverage < NOMINAL
                    for coverage in per_genotype_coverage
                ),
            })

            # --------------------------------------------
            # Conditional size-group aggregation.
            # --------------------------------------------

            for category in SIZES:
                matched = [
                    r for r in size_rows
                    if (
                        r["horizon_hours"] == horizon
                        and r["method"] == method
                        and r["leaf_size"] == category
                        and r["plants"] >= 2
                        and r["windows"] >= 30
                    )
                ]

                by_group = defaultdict(list)

                for row in matched:
                    by_group[row["test_genotype"]].append(
                        row
                    )

                complete = []

                for genotype, entries in by_group.items():
                    assignments = sorted(
                        r["inner_fold"] for r in entries
                    )

                    if assignments == [1, 2, 3, 4]:
                        complete.append(entries)

                if not complete:
                    continue

                def category_macro(field):
                    return float(
                        np.mean([
                            np.mean([
                                entry[field]
                                for entry in entries
                            ])
                            for entries in complete
                        ])
                    )

                conditional.append({
                    "horizon_hours": horizon,
                    "method": method,
                    "leaf_size": category,
                    "complete_genotypes": len(complete),
                    "macro_coverage": category_macro(
                        "coverage"
                    ),
                    "macro_width": category_macro(
                        "width"
                    ),
                    "macro_interval_score": category_macro(
                        "score"
                    ),
                })

    return overall, conditional


def make_figures(overall, conditional):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; figures skipped.")
        return

    for horizon in HORIZONS:
        # ------------------------------------------------
        # Figure 1: Subgroup coverage
        # ------------------------------------------------

        selected = [
            r for r in conditional
            if r["horizon_hours"] == horizon
        ]

        lookup = {
            (r["method"], r["leaf_size"]): r
            for r in selected
        }

        x = np.arange(len(SIZES))
        width = 0.36

        fig, ax = plt.subplots(figsize=(10, 5))

        for i, method in enumerate(METHODS):
            values = [
                100 * lookup[(method, category)]["macro_coverage"]
                if (method, category) in lookup
                else np.nan
                for category in SIZES
            ]

            ax.bar(
                x + (i - 0.5) * width,
                values,
                width,
                label=method,
            )

        ax.axhline(
            90.0,
            linestyle="--",
            label="Nominal 90%",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(SIZES, rotation=12)
        ax.set_ylabel("Empirical coverage (%)")
        ax.set_ylim(0, 105)
        ax.set_title(
            f"E3.3 — {horizon}h Mondrian leaf-size coverage"
        )
        ax.legend()
        fig.tight_layout()

        figure_path = (
            FIGURES
            / f"E3_3_size_coverage_{horizon}h.png"
        )

        fig.savefig(
            figure_path,
            dpi=200,
        )
        plt.close(fig)

        # ------------------------------------------------
        # Figure 2: Interval score by leaf-size group
        # ------------------------------------------------

        fig, ax = plt.subplots(figsize=(10, 5))

        for i, method in enumerate(METHODS):
            values = [
                lookup[(method, category)]["macro_interval_score"]
                if (method, category) in lookup
                else np.nan
                for category in SIZES
            ]

            ax.bar(
                x + (i - 0.5) * width,
                values,
                width,
                label=method,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(SIZES, rotation=12)
        ax.set_ylabel("Mean 90% interval score (pixels)")
        ax.set_title(
            f"E3.3 — {horizon}h Mondrian interval efficiency"
        )
        ax.legend()
        fig.tight_layout()

        figure_path = (
            FIGURES
            / f"E3_3_size_interval_score_{horizon}h.png"
        )

        fig.savefig(
            figure_path,
            dpi=200,
        )
        plt.close(fig)


def main():
    print("=" * 70)
    print("E3.3 — MONDRIAN CONFORMAL CALIBRATION")
    print("=" * 70)

    with FOLDS_PATH.open(encoding="utf-8") as f:
        folds = json.load(f)["folds"]

    if len(folds) != EXPECTED_OUTER_FOLDS:
        raise ValueError("Expected 21 outer folds")

    data = baseline.load_data()

    # Ensure the reused E3.1 loader contains
    # the expected categorical information.
    for horizon in HORIZONS:
        if "size" not in data[horizon]:
            raise ValueError(
                "E3.1 data loader missing leaf-size categories"
            )

    fold_rows = []
    size_rows = []

    for horizon in HORIZONS:
        print(
            f"\n{horizon}h — "
            f"{len(data[horizon]['y']):,} windows",
            flush=True,
        )

        for outer in folds:
            inner_folds = outer["inner_folds"]

            if len(inner_folds) != EXPECTED_INNER_FOLDS:
                raise ValueError(
                    "Expected four inner assignments"
                )

            for inner in inner_folds:
                result, subgroup_result = evaluate(
                    data[horizon],
                    outer,
                    inner,
                    horizon,
                )

                fold_rows.extend(result)
                size_rows.extend(subgroup_result)

                by_method = {
                    row["method"]: row
                    for row in result
                }

                print(
                    f"  Outer {outer['outer_fold']:02d}/21 "
                    f"| inner {inner['inner_fold']}/4 "
                    f"| {outer['test_genotype']:11s} "
                    f"| U6={by_method[METHODS[0]]['coverage']:.3f} "
                    f"| U7={by_method[METHODS[1]]['coverage']:.3f}",
                    flush=True,
                )

        # Save checkpoint after each horizon.
        save_csv(FOLD_OUT, fold_rows)
        save_csv(SIZE_OUT, size_rows)

    overall, conditional = summarize(
        fold_rows,
        size_rows,
    )

    save_csv(SUMMARY_OUT, overall)
    save_csv(SIZE_SUMMARY_OUT, conditional)

    make_figures(
        overall,
        conditional,
    )

    report = [
        "# E3.3 — Mondrian Conformal Calibration",
        "",
        "## Experimental protocol",
        "",
        "- 21 held-out genotypes.",
        "- Four calibration assignments per genotype.",
        "- Forecast horizons: 24h and 48h.",
        "- Model fitting: 15 genotypes.",
        "- Calibration: 5 separate genotypes.",
        "- Testing: 1 entirely unseen genotype.",
        "- Fixed B3 histogram gradient-boosting model.",
        "- Nominal interval coverage: 90%.",
        "",
        "## Methods",
        "",
        "- U6: pooled residual quantile per leaf-size group.",
        "- U7: genotype/plant-balanced residual quantile "
        "per leaf-size group.",
        "- Small: area below 500 pixels.",
        "- Medium: area from 500 to below 2000 pixels.",
        "- Large: area at least 2000 pixels.",
        "- Categories determined from historical leaf area.",
        "",
        "## Overall results",
        "",
        "| Horizon | Method | Coverage | Width (px) | "
        "Interval score |",
        "|---|---|---:|---:|---:|",
    ]

    for row in overall:
        report.append(
            f"| {row['horizon_hours']}h "
            f"| {row['method']} "
            f"| {row['macro_coverage']:.3f} "
            f"| {row['macro_width']:.2f} "
            f"| {row['macro_interval_score']:.2f} |"
        )

    report.extend([
        "",
        "## Scientific limitations",
        "",
        "- Leaf-size boundaries were used in earlier "
        "exploratory analyses.",
        "- Correlated windows are not independent samples.",
        "- Calibration and test genotypes may not be "
        "exchangeable.",
        "- Weighted calibration has no automatic "
        "finite-sample coverage guarantee.",
        "- Observed group coverage is not a formal "
        "conditional-coverage guarantee.",
        "- Outer test data have informed the exploratory "
        "research direction.",
        "- Annotation quality remains a potential "
        "source of error.",
        "",
        "## Next step",
        "",
        "Compare U6/U7 against normalized conformal "
        "prediction (U3) and balanced CQR (U5). "
        "Use coverage, interval width, interval score, "
        "and leaf-size subgroup reliability.",
    ])

    REPORT_OUT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("E3.3 — FINAL RESULTS")
    print("=" * 70)

    for row in overall:
        print(
            f"\n{row['horizon_hours']}h | {row['method']}"
            f"\n  Coverage: {row['macro_coverage']:.3f}"
            f"\n  Mean width: {row['macro_width']:.2f} px"
            f"\n  Interval score: "
            f"{row['macro_interval_score']:.2f}"
            f"\n  Genotypes below 90%: "
            f"{row['genotypes_below_90']}/21"
        )

    for row in conditional:
        print(
            f"\n{row['horizon_hours']}h "
            f"| {row['method']} "
            f"| {row['leaf_size']}"
            f"\n  Coverage: {row['macro_coverage']:.3f}"
            f"\n  Interval score: "
            f"{row['macro_interval_score']:.2f}"
            f"\n  Complete genotypes: "
            f"{row['complete_genotypes']}/21"
        )

    print("\nSaved:")
    for path in (
        FOLD_OUT,
        SIZE_OUT,
        SUMMARY_OUT,
        SIZE_SUMMARY_OUT,
        REPORT_OUT,
    ):
        print(path)

    print("\nE3.3 COMPLETE")


if __name__ == "__main__":
    main()
