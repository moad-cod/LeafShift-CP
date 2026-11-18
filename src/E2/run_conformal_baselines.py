
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================================================
# E2.1 — UNCERTAINTY CALIBRATION BASELINES
#
# U0: Pooled-window absolute-residual split conformal
# U1: Genotype/plant-balanced empirical residual quantile
#
# 21 LOGO outer folds, 24h and 48h.
# Uses the E2.0 primary calibration split (inner fold 1).
#
# Test genotype NEVER used in fitting or calibration.
# No raw data or earlier experiment outputs modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "data/processed/forecasting/forecast_windows.csv"
)
PROTOCOL = ROOT / "configs/E2_0_uncertainty_protocol.json"

EXP = ROOT / "experiments/E2"
TABLES = EXP / "tables"
FIGURES = EXP / "figures"
REPORTS = EXP / "reports"

for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

PRED_FILE = TABLES / "E2_1_interval_predictions.csv"
PLANT_FILE = TABLES / "E2_1_plant_metrics.csv"
GENOTYPE_FILE = TABLES / "E2_1_genotype_metrics.csv"
SUMMARY_FILE = TABLES / "E2_1_summary.csv"
THRESHOLD_FILE = TABLES / "E2_1_calibration_thresholds.csv"
REPORT_FILE = REPORTS / "E2_1_uncertainty_baselines.md"

HORIZONS = (24, 48)
EXPECTED = {24: 121431, 48: 107214}
TARGET_COVERAGE = 0.90
ALPHA = 1.0 - TARGET_COVERAGE

MODEL_PARAMS = {
    "max_iter": 150,
    "max_leaf_nodes": 31,
    "learning_rate": 0.05,
    "l2_regularization": 1.0,
    "max_bins": 64,
    "min_samples_leaf": 40,
    "early_stopping": False,
    "random_state": 20261108,
}

METHODS = ("U0_pooled", "U1_balanced")


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty output: {path}")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def make_features(times, areas, horizon):
    """Reproduce E1.5 feature engineering."""
    if len(times) != 6 or len(areas) != 6:
        raise ValueError("Expected six historical observations")

    gaps = np.diff(times)
    changes = np.diff(areas)

    if np.any(gaps <= 0):
        raise ValueError("Invalid observation times")

    rates = changes / gaps

    mean_t = np.mean(times)
    mean_y = np.mean(areas)
    centered = times - mean_t
    denominator = float(np.dot(centered, centered))

    slope = (
        float(np.dot(centered, areas - mean_y))
        / denominator
        if denominator > 1e-12
        else 0.0
    )

    intercept = mean_y - slope * mean_t
    linear = max(
        0.0,
        float(intercept + slope * horizon)
    )

    features = [
        *areas.tolist(),
        *times.tolist(),
        *changes.tolist(),
        *rates.tolist(),
        linear,
        float(areas[-1]),
        float(np.mean(areas)),
        float(times[-1] - times[0]),
        float(np.max(gaps)),
        float(np.mean(rates)),
        float(areas[-1] / max(areas[0], 1.0)),
    ]

    return np.asarray(features, dtype=np.float32)


def load_manifest():
    data = {
        h: {
            "X": [],
            "y": [],
            "g": [],
            "p": [],
            "id": [],
        }
        for h in HORIZONS
    }

    with MANIFEST.open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            h = int(row["horizon_hours"])

            if h not in HORIZONS:
                raise ValueError(f"Unexpected horizon {h}")

            times = np.asarray(
                json.loads(row["history_relative_hours"]),
                dtype=np.float64,
            )
            areas = np.asarray(
                json.loads(row["history_areas_pixels"]),
                dtype=np.float64,
            )

            if abs(times[-1]) > 1e-5:
                raise ValueError("Invalid cutoff timestamp")

            feature = make_features(
                times, areas, h
            )

            genotype = row["genotype"]
            plant = row["plant_id"]

            if not plant.startswith(genotype + "/"):
                raise ValueError(
                    f"Plant/genotype mismatch: {plant}"
                )

            d = data[h]
            d["X"].append(feature)
            d["y"].append(
                float(row["target_area_pixels"])
            )
            d["g"].append(genotype)
            d["p"].append(plant)
            d["id"].append(row["window_id"])

    for h in HORIZONS:
        d = data[h]

        d["X"] = np.stack(d["X"])
        d["y"] = np.asarray(d["y"], dtype=np.float64)
        d["g"] = np.asarray(d["g"])
        d["p"] = np.asarray(d["p"])

        if len(d["y"]) != EXPECTED[h]:
            raise ValueError(
                f"Unexpected {h}h window count"
            )

        if not np.isfinite(d["X"]).all():
            raise ValueError("Nonfinite features")
        if not np.isfinite(d["y"]).all():
            raise ValueError("Nonfinite targets")

    return data


def balanced_weights(genotypes, plants):
    """
    Equal total weight per genotype, then per plant,
    then per window. Matches E1.5 training weights.
    """
    genotype_plants = defaultdict(set)
    plant_counts = Counter(plants)

    for g, p in zip(genotypes, plants):
        genotype_plants[str(g)].add(str(p))

    n_genotypes = len(genotype_plants)

    weights = np.asarray([
        1.0 / (
            n_genotypes
            * len(genotype_plants[str(g)])
            * plant_counts[str(p)]
        )
        for g, p in zip(genotypes, plants)
    ], dtype=np.float64)

    return weights * len(weights) / weights.sum()


def fit_model(d, train_mask):
    model = HistGradientBoostingRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        d["X"][train_mask],
        d["y"][train_mask],
        sample_weight=balanced_weights(
            d["g"][train_mask],
            d["p"][train_mask],
        ),
    )

    return model


def predict(model, X):
    return np.maximum(
        model.predict(X),
        0.0,
    )


def pooled_conformal_quantile(residuals):
    """
    Conventional finite-sample split conformal rank:
        k = ceil((n + 1) * (1 - alpha))

    If k > n, the conformal threshold is infinite.
    Here calibration sets are large enough that
    finite thresholds are expected.

    The exchangeability assumption is NOT guaranteed
    across calibration and test genotypes.
    """
    residuals = np.asarray(
        residuals, dtype=np.float64
    )

    n = len(residuals)

    if n == 0:
        raise ValueError("Empty calibration residuals")

    k = math.ceil(
        (n + 1) * TARGET_COVERAGE
    )

    if k > n:
        return float("inf")

    return float(
        np.partition(residuals, k - 1)[k - 1]
    )


def balanced_empirical_quantile(
    residuals, genotypes, plants
):
    """
    Weighted empirical 90th percentile, assigning:
      equal total weight per calibration genotype
      equal total weight per plant within genotype
      equal total weight per window within plant

    This is NOT a finite-sample conformal guarantee.
    """
    residuals = np.asarray(
        residuals, dtype=np.float64
    )

    weights = balanced_weights(
        genotypes, plants
    )

    weights = weights / weights.sum()

    order = np.argsort(residuals)
    sorted_r = residuals[order]
    sorted_w = weights[order]

    cumulative = np.cumsum(sorted_w)

    index = int(
        np.searchsorted(
            cumulative,
            TARGET_COVERAGE,
            side="left",
        )
    )

    index = min(index, len(sorted_r) - 1)

    return float(sorted_r[index])


def evaluate_plant(
    targets, predictions, lower, upper
):
    covered = (
        (targets >= lower)
        & (targets <= upper)
    )

    errors = predictions - targets
    widths = upper - lower

    return {
        "windows": len(targets),
        "coverage": float(np.mean(covered)),
        "mean_width": float(np.mean(widths)),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(
            np.sqrt(np.mean(errors ** 2))
        ),
    }


def run_evaluation(d, split, horizon):
    fit_g = set(split["fit_genotypes"])
    cal_g = set(split["calibration_genotypes"])
    test_g = set(split["test_genotypes"])

    assert len(fit_g) == 15
    assert len(cal_g) == 5
    assert len(test_g) == 1

    assert fit_g.isdisjoint(cal_g)
    assert fit_g.isdisjoint(test_g)
    assert cal_g.isdisjoint(test_g)

    fit_mask = np.isin(d["g"], list(fit_g))
    cal_mask = np.isin(d["g"], list(cal_g))
    test_mask = np.isin(d["g"], list(test_g))

    if not (
        fit_mask.any()
        and cal_mask.any()
        and test_mask.any()
    ):
        raise ValueError("Empty fitting/calibration/test set")

    assert set(d["g"][fit_mask]) == fit_g
    assert set(d["g"][cal_mask]) == cal_g
    assert set(d["g"][test_mask]) == test_g

    model = fit_model(d, fit_mask)

    cal_pred = predict(
        model, d["X"][cal_mask]
    )
    cal_y = d["y"][cal_mask]
    residuals = np.abs(
        cal_y - cal_pred
    )

    thresholds = {
        "U0_pooled": pooled_conformal_quantile(
            residuals
        ),
        "U1_balanced": balanced_empirical_quantile(
            residuals,
            d["g"][cal_mask],
            d["p"][cal_mask],
        ),
    }

    test_indices = np.flatnonzero(test_mask)

    y_test = d["y"][test_mask]
    pred_test = predict(
        model, d["X"][test_mask]
    )
    plants_test = d["p"][test_mask]

    threshold_rows = []

    for method, q in thresholds.items():
        threshold_rows.append({
            "outer_fold": split["outer_fold"],
            "horizon_hours": horizon,
            "test_genotype": split["test_genotype"],
            "method": method,
            "calibration_genotypes": len(cal_g),
            "calibration_plants": len(
                set(d["p"][cal_mask])
            ),
            "calibration_windows": len(residuals),
            "residual_threshold_pixels": q,
        })

    prediction_rows = []
    plant_rows = []
    genotype_rows = []

    for method, q in thresholds.items():
        lower = np.maximum(
            0.0,
            pred_test - q
        )
        upper = pred_test + q

        covered = (
            (y_test >= lower)
            & (y_test <= upper)
        )

        for j, index in enumerate(test_indices):
            prediction_rows.append({
                "outer_fold": split["outer_fold"],
                "horizon_hours": horizon,
                "genotype": split["test_genotype"],
                "plant_id": str(plants_test[j]),
                "window_id": d["id"][index],
                "method": method,
                "target_area_pixels": float(y_test[j]),
                "predicted_area_pixels": float(pred_test[j]),
                "lower_pixels": float(lower[j]),
                "upper_pixels": float(upper[j]),
                "covered": int(covered[j]),
            })

        plant_scores = []

        for plant in sorted(set(plants_test)):
            mask = plants_test == plant

            result = evaluate_plant(
                y_test[mask],
                pred_test[mask],
                lower[mask],
                upper[mask],
            )

            plant_rows.append({
                "outer_fold": split["outer_fold"],
                "horizon_hours": horizon,
                "genotype": split["test_genotype"],
                "plant_id": str(plant),
                "method": method,
                "windows": result["windows"],
                "coverage": result["coverage"],
                "mean_width_pixels": result["mean_width"],
                "mae_pixels": result["mae"],
                "rmse_pixels": result["rmse"],
            })

            plant_scores.append(result)

        genotype_rows.append({
            "outer_fold": split["outer_fold"],
            "horizon_hours": horizon,
            "genotype": split["test_genotype"],
            "method": method,
            "plants": len(plant_scores),
            "windows": len(y_test),
            "pooled_window_coverage": float(
                np.mean(covered)
            ),
            "plant_mean_coverage": float(
                np.mean([
                    x["coverage"]
                    for x in plant_scores
                ])
            ),
            "plant_mean_width_pixels": float(
                np.mean([
                    x["mean_width"]
                    for x in plant_scores
                ])
            ),
            "plant_mean_mae_pixels": float(
                np.mean([
                    x["mae"]
                    for x in plant_scores
                ])
            ),
            "calibration_threshold_pixels": q,
        })

    return (
        threshold_rows,
        prediction_rows,
        plant_rows,
        genotype_rows,
    )


def create_figures(genotype_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; figures skipped")
        return

    for horizon in HORIZONS:
        rows = [
            r for r in genotype_rows
            if r["horizon_hours"] == horizon
        ]

        genotypes = sorted({
            r["genotype"] for r in rows
        })

        lookup = {
            (r["genotype"], r["method"]):
            r["plant_mean_coverage"]
            for r in rows
        }

        x = np.arange(len(genotypes))
        width = 0.38

        fig, ax = plt.subplots(figsize=(13, 6))

        for index, method in enumerate(METHODS):
            values = [
                lookup[(genotype, method)]
                for genotype in genotypes
            ]

            ax.bar(
                x + (index - 0.5) * width,
                values,
                width,
                label=method,
            )

        ax.axhline(
            TARGET_COVERAGE,
            linestyle="--",
            label="Nominal 90%",
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            genotypes, rotation=65, ha="right"
        )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(
            "Plant-averaged empirical coverage"
        )
        ax.set_title(
            f"E2.1 — {horizon}h unseen-genotype coverage"
        )
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            FIGURES / f"E2_1_coverage_{horizon}h.png",
            dpi=200,
        )
        plt.close(fig)


def main():
    print("=" * 70)
    print("E2.1 — UNCERTAINTY CALIBRATION BASELINES")
    print("=" * 70)

    with PROTOCOL.open(encoding="utf-8") as f:
        protocol = json.load(f)

    splits = protocol["primary_splits"]

    if len(splits) != 21:
        raise ValueError(
            "Expected 21 primary calibration splits"
        )

    if not math.isclose(
        float(protocol["nominal_coverage"]),
        TARGET_COVERAGE,
    ):
        raise ValueError("Coverage target mismatch")

    data = load_manifest()

    for horizon in HORIZONS:
        print(
            f"Loaded {horizon}h: "
            f"{len(data[horizon]['y']):,} windows"
        )

    thresholds_all = []
    predictions_all = []
    plants_all = []
    genotypes_all = []

    for horizon in HORIZONS:
        print(f"\n{'=' * 70}")
        print(f"{horizon}-HOUR UNCERTAINTY CALIBRATION")
        print("=" * 70)

        d = data[horizon]

        for split in splits:
            (
                thresholds,
                predictions,
                plants,
                genotypes,
            ) = run_evaluation(
                d, split, horizon
            )

            thresholds_all.extend(thresholds)
            predictions_all.extend(predictions)
            plants_all.extend(plants)
            genotypes_all.extend(genotypes)

            by_method = {
                r["method"]: r
                for r in genotypes
            }

            print(
                f"Fold {split['outer_fold']:02d}/21 "
                f"| {split['test_genotype']:11s} "
                f"| U0={by_method['U0_pooled']['plant_mean_coverage']:.3f} "
                f"| U1={by_method['U1_balanced']['plant_mean_coverage']:.3f}"
            )

    save_csv(THRESHOLD_FILE, thresholds_all)
    save_csv(PRED_FILE, predictions_all)
    save_csv(PLANT_FILE, plants_all)
    save_csv(GENOTYPE_FILE, genotypes_all)

    summary_rows = []

    for horizon in HORIZONS:
        for method in METHODS:
            rows = [
                r for r in genotypes_all
                if (
                    r["horizon_hours"] == horizon
                    and r["method"] == method
                )
            ]

            if len(rows) != 21:
                raise ValueError(
                    "Incomplete genotype evaluation"
                )

            coverage = float(np.mean([
                r["plant_mean_coverage"]
                for r in rows
            ]))

            width = float(np.mean([
                r["plant_mean_width_pixels"]
                for r in rows
            ]))

            mae = float(np.mean([
                r["plant_mean_mae_pixels"]
                for r in rows
            ]))

            summary_rows.append({
                "horizon_hours": horizon,
                "method": method,
                "nominal_coverage": TARGET_COVERAGE,
                "macro_genotype_plant_coverage": coverage,
                "macro_genotype_plant_interval_width_pixels": width,
                "macro_genotype_plant_mae_pixels": mae,
                "genotypes_below_90pct": sum(
                    r["plant_mean_coverage"]
                    < TARGET_COVERAGE
                    for r in rows
                ),
                "genotypes_below_80pct": sum(
                    r["plant_mean_coverage"] < 0.80
                    for r in rows
                ),
                "minimum_genotype_coverage": min(
                    r["plant_mean_coverage"]
                    for r in rows
                ),
                "maximum_genotype_coverage": max(
                    r["plant_mean_coverage"]
                    for r in rows
                ),
            })

    save_csv(SUMMARY_FILE, summary_rows)
    create_figures(genotypes_all)

    report = [
        "# E2.1 — Uncertainty Calibration Baselines",
        "",
        "## Protocol",
        "",
        "- 21 outer held-out genotypes.",
        "- 15 fitting genotypes, 5 calibration genotypes.",
        "- E2.0 inner fold 1 used for calibration.",
        "- Fixed B3 model configuration.",
        "- U0: pooled absolute-residual conformal threshold.",
        "- U1: genotype/plant-balanced empirical threshold.",
        "- Nominal target coverage: 90%.",
        "- Interval lower bound clipped to zero.",
        "",
        "## Results",
        "",
        "| Horizon | Method | Macro coverage | "
        "Mean width (px) | Genotypes below 90% |",
        "|---|---|---:|---:|---:|",
    ]

    for r in summary_rows:
        report.append(
            f"| {r['horizon_hours']}h "
            f"| {r['method']} "
            f"| {r['macro_genotype_plant_coverage']:.3f} "
            f"| {r['macro_genotype_plant_interval_width_pixels']:.2f} "
            f"| {r['genotypes_below_90pct']}/21 |"
        )

    report.extend([
        "",
        "## Important limitations",
        "",
        "- Calibration and test genotypes may differ.",
        "- Conventional pooled-window conformal guarantees "
        "cannot be assumed under this shift.",
        "- U1 has no automatic conformal coverage guarantee.",
        "- Within-plant forecasting windows are correlated.",
        "- Macro coverage does not imply every genotype "
        "achieves nominal coverage.",
        "- Only one predefined calibration split is evaluated.",
        "- Biological leaf identities remain subject to "
        "unresolved annotation-quality concerns.",
        "",
        "## Next",
        "",
        "E2.2 — Analyze interval coverage by held-out genotype, "
        "plant, horizon, and calibration method.",
    ])

    REPORT_FILE.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("E2.1 — FINAL UNCERTAINTY RESULTS")
    print("=" * 70)

    for r in summary_rows:
        print(
            f"\n{r['horizon_hours']}h "
            f"| {r['method']}"
        )
        print(
            "  Macro plant-averaged coverage: "
            f"{r['macro_genotype_plant_coverage']:.3f}"
        )
        print(
            "  Mean interval width: "
            f"{r['macro_genotype_plant_interval_width_pixels']:.2f} px"
        )
        print(
            "  Point prediction MAE: "
            f"{r['macro_genotype_plant_mae_pixels']:.2f} px"
        )
        print(
            "  Genotypes below 90% coverage: "
            f"{r['genotypes_below_90pct']}/21"
        )
        print(
            "  Genotypes below 80% coverage: "
            f"{r['genotypes_below_80pct']}/21"
        )
        print(
            "  Coverage range: "
            f"{r['minimum_genotype_coverage']:.3f} "
            f"to {r['maximum_genotype_coverage']:.3f}"
        )

    print("\nSaved:")
    print(THRESHOLD_FILE)
    print(PRED_FILE)
    print(PLANT_FILE)
    print(GENOTYPE_FILE)
    print(SUMMARY_FILE)
    print(REPORT_FILE)

    print("\nE2.1 COMPLETE")


if __name__ == "__main__":
    main()
