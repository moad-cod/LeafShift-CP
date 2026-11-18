
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import math

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# ============================================================
# E1.4 — RIDGE REGRESSION WITH NESTED GENOTYPE CV
#
# 21 LOGO outer folds
# 4 genotype-disjoint inner folds per outer fold
# Separate models for 24h and 48h horizons
#
# Ridge is fitted on development genotypes only.
# Scalers are fitted on training genotypes only.
# Hyperparameters are selected using inner validation.
#
# No raw data, E0 outputs, or E1.0 manifest modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "data/processed/forecasting"
    / "forecast_windows.csv"
)

FOLD_PATH = ROOT / "configs/E1_1_genotype_folds.json"

EXP = ROOT / "experiments/E1"
TABLES = EXP / "tables"
FIGURES = EXP / "figures"
REPORTS = EXP / "reports"

for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

PREDICTIONS = TABLES / "E1_4_ridge_predictions.csv"
GENOTYPE_RESULTS = TABLES / "E1_4_genotype_metrics.csv"
SUMMARY = TABLES / "E1_4_summary.csv"
ALPHA_RESULTS = TABLES / "E1_4_alpha_selection.csv"
REPORT = REPORTS / "E1_4_ridge_logo_report.md"

HORIZONS = (24, 48)
ALPHAS = (
    0.0001,
    0.001,
    0.01,
    0.1,
    1.0,
    10.0,
    100.0,
    1000.0,
)

EXPECTED_WINDOWS = {
    24: 121431,
    48: 107214,
}

FEATURE_NAMES = [
    *[f"area_t{i}" for i in range(6)],
    *[f"time_t{i}" for i in range(6)],
    *[f"delta_area_{i}" for i in range(5)],
    *[f"growth_rate_{i}" for i in range(5)],
    "linear_extrapolation",
    "last_area",
    "mean_area",
    "history_span_hours",
    "max_observation_gap_hours",
    "mean_growth_rate",
    "last_to_first_area_ratio",
]


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No results for {path}")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def linear_prediction(times, areas, horizon):
    """Reproduce the E1.2 B1 prediction rule."""
    mt = np.mean(times)
    my = np.mean(areas)

    dt = times - mt
    denominator = float(np.dot(dt, dt))

    if denominator <= 1e-12:
        return max(0.0, float(areas[-1]))

    slope = float(
        np.dot(dt, areas - my) / denominator
    )

    intercept = float(my - slope * mt)

    return max(
        0.0,
        intercept + slope * horizon
    )


def make_features(times, areas, horizon):
    """
    Only six historical observations and the
    requested nominal horizon are used.

    Never uses the future target timestamp,
    future area, genotype label, or QC outcome.
    """
    gaps = np.diff(times)
    changes = np.diff(areas)

    if np.any(gaps <= 0):
        raise ValueError(
            "Historical timestamps must increase."
        )

    rates = changes / gaps

    b1 = linear_prediction(
        times, areas, horizon
    )

    features = [
        *areas.tolist(),
        *times.tolist(),
        *changes.tolist(),
        *rates.tolist(),
        b1,
        float(areas[-1]),
        float(np.mean(areas)),
        float(times[-1] - times[0]),
        float(np.max(gaps)),
        float(np.mean(rates)),
        float(
            areas[-1] / max(areas[0], 1.0)
        ),
    ]

    if len(features) != len(FEATURE_NAMES):
        raise RuntimeError(
            "Unexpected feature vector length."
        )

    return np.asarray(features, dtype=np.float64), b1


def load_data():
    data = {
        horizon: {
            "X": [],
            "y": [],
            "b1": [],
            "genotype": [],
            "plant": [],
            "window_id": [],
            "leaf_id": [],
        }
        for horizon in HORIZONS
    }

    with MANIFEST.open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            horizon = int(row["horizon_hours"])

            if horizon not in HORIZONS:
                raise ValueError(
                    f"Unexpected horizon: {horizon}"
                )

            times = np.asarray(
                json.loads(
                    row["history_relative_hours"]
                ),
                dtype=np.float64,
            )

            areas = np.asarray(
                json.loads(
                    row["history_areas_pixels"]
                ),
                dtype=np.float64,
            )

            if len(times) != 6 or len(areas) != 6:
                raise ValueError(
                    "Each window needs six observations."
                )

            if abs(times[-1]) > 1e-5:
                raise ValueError(
                    "Cutoff does not match last observation."
                )

            features, b1 = make_features(
                times, areas, horizon
            )

            d = data[horizon]
            d["X"].append(features)
            d["y"].append(
                float(row["target_area_pixels"])
            )
            d["b1"].append(b1)
            d["genotype"].append(row["genotype"])
            d["plant"].append(row["plant_id"])
            d["window_id"].append(row["window_id"])
            d["leaf_id"].append(row["leaf_id"])

    for horizon in HORIZONS:
        d = data[horizon]

        d["X"] = np.vstack(d["X"])
        d["y"] = np.asarray(
            d["y"], dtype=np.float64
        )
        d["b1"] = np.asarray(
            d["b1"], dtype=np.float64
        )
        d["genotype"] = np.asarray(
            d["genotype"], dtype=str
        )
        d["plant"] = np.asarray(
            d["plant"], dtype=str
        )

        if len(d["y"]) != EXPECTED_WINDOWS[horizon]:
            raise RuntimeError(
                f"{horizon}h: unexpected window count."
            )

        if not np.isfinite(d["X"]).all():
            raise ValueError(
                "Nonfinite input features."
            )

        if not np.isfinite(d["y"]).all():
            raise ValueError(
                "Nonfinite target values."
            )

    return data


def calculate_weights(genotypes, plants):
    """
    Each genotype has equal total training weight.
    Within genotype, each plant has equal weight.
    Within plant, each window has equal weight.

    Normalized to mean sample weight = 1.
    """
    n = len(plants)

    if n == 0:
        raise ValueError("Empty training set.")

    genotype_to_plants = defaultdict(set)

    for g, p in zip(genotypes, plants):
        genotype_to_plants[g].add(p)

    plant_counts = Counter(plants)

    n_genotypes = len(genotype_to_plants)

    weights = np.asarray([
        1.0 / (
            n_genotypes
            * len(genotype_to_plants[g])
            * plant_counts[p]
        )
        for g, p in zip(genotypes, plants)
    ], dtype=np.float64)

    weights *= n / weights.sum()

    return weights


def fit_ridge(X, y, genotypes, plants, alpha):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    weights = calculate_weights(
        genotypes, plants
    )

    model = Ridge(
        alpha=alpha,
        solver="cholesky",
    )

    model.fit(
        X_scaled,
        y,
        sample_weight=weights,
    )

    return scaler, model


def predict_ridge(scaler, model, X):
    predictions = model.predict(
        scaler.transform(X)
    )

    # Projected area cannot be negative.
    return np.maximum(predictions, 0.0)


def genotype_macro_mae(
    targets, predictions, genotypes, plants
):
    """
    Mean absolute error:
      windows -> plant mean -> genotype mean

    Genotypes are equally weighted.
    Plants within a genotype are equally weighted.
    """
    errors = np.abs(predictions - targets)

    plant_sums = defaultdict(float)
    plant_counts = Counter()
    plant_genotype = {}

    for error, g, p in zip(
        errors, genotypes, plants
    ):
        plant_sums[p] += float(error)
        plant_counts[p] += 1

        if p in plant_genotype:
            if plant_genotype[p] != g:
                raise ValueError(
                    "Plant belongs to multiple genotypes."
                )
        else:
            plant_genotype[p] = g

    genotype_plant_maes = defaultdict(list)

    for plant, error_sum in plant_sums.items():
        g = plant_genotype[plant]

        genotype_plant_maes[g].append(
            error_sum / plant_counts[plant]
        )

    per_genotype = {
        g: float(np.mean(values))
        for g, values in genotype_plant_maes.items()
    }

    return (
        float(np.mean(list(per_genotype.values()))),
        per_genotype,
    )


def inner_cv_scores(d, fold):
    """
    Four inner validation folds, each with
    15 training and five validation genotypes.

    Returns mean genotype-balanced validation MAE
    for each candidate alpha.
    """
    alpha_scores = {
        alpha: [] for alpha in ALPHAS
    }

    X = d["X"]
    y = d["y"]
    g = d["genotype"]
    p = d["plant"]

    test_genotype = fold["test_genotype"]

    for inner in fold["inner_folds"]:
        training = set(inner["train_genotypes"])
        validation = set(
            inner["validation_genotypes"]
        )

        assert len(training) == 15
        assert len(validation) == 5
        assert training.isdisjoint(validation)
        assert test_genotype not in training
        assert test_genotype not in validation

        train_mask = np.isin(
            g, list(training)
        )
        val_mask = np.isin(
            g, list(validation)
        )

        if not train_mask.any() or not val_mask.any():
            raise ValueError(
                "Empty inner training/validation fold."
            )

        # Check explicitly that held-out genotype
        # is never fitted or validated.
        assert not np.any(
            g[train_mask] == test_genotype
        )
        assert not np.any(
            g[val_mask] == test_genotype
        )

        for alpha in ALPHAS:
            scaler, model = fit_ridge(
                X[train_mask],
                y[train_mask],
                g[train_mask],
                p[train_mask],
                alpha,
            )

            preds = predict_ridge(
                scaler, model, X[val_mask]
            )

            score, _ = genotype_macro_mae(
                y[val_mask],
                preds,
                g[val_mask],
                p[val_mask],
            )

            alpha_scores[alpha].append(score)

    return {
        alpha: float(np.mean(scores))
        for alpha, scores in alpha_scores.items()
    }


def compare_predictions(d, fold, alpha):
    """
    Fit on all 20 development genotypes;
    evaluate only the one held-out genotype.
    """
    test_genotype = fold["test_genotype"]
    development = set(
        fold["development_genotypes"]
    )

    assert len(development) == 20
    assert test_genotype not in development

    train_mask = np.isin(
        d["genotype"], list(development)
    )
    test_mask = (
        d["genotype"] == test_genotype
    )

    if not train_mask.any() or not test_mask.any():
        raise ValueError(
            "Empty outer training/test fold."
        )

    assert set(d["genotype"][train_mask]) == development
    assert set(d["genotype"][test_mask]) == {
        test_genotype
    }

    scaler, model = fit_ridge(
        d["X"][train_mask],
        d["y"][train_mask],
        d["genotype"][train_mask],
        d["plant"][train_mask],
        alpha,
    )

    ridge_predictions = predict_ridge(
        scaler,
        model,
        d["X"][test_mask],
    )

    b1_predictions = d["b1"][test_mask]

    y = d["y"][test_mask]
    g = d["genotype"][test_mask]
    p = d["plant"][test_mask]

    ridge_mae, _ = genotype_macro_mae(
        y, ridge_predictions, g, p
    )
    b1_mae, _ = genotype_macro_mae(
        y, b1_predictions, g, p
    )

    indices = np.flatnonzero(test_mask)

    prediction_rows = []

    for j, idx in enumerate(indices):
        prediction_rows.append({
            "outer_fold": fold["outer_fold"],
            "horizon_hours": d["horizon"],
            "genotype": test_genotype,
            "plant_id": str(p[j]),
            "window_id": d["window_id"][idx],
            "leaf_id": d["leaf_id"][idx],
            "target_area_pixels": float(y[j]),
            "B1_linear": float(b1_predictions[j]),
            "B2_ridge": float(ridge_predictions[j]),
            "selected_alpha": alpha,
        })

    result = {
        "outer_fold": fold["outer_fold"],
        "horizon_hours": d["horizon"],
        "test_genotype": test_genotype,
        "test_plants": len(set(p)),
        "test_windows": len(y),
        "selected_alpha": alpha,
        "B1_mae_pixels": round(b1_mae, 5),
        "B2_mae_pixels": round(ridge_mae, 5),
        "mae_improvement_pixels": round(
            b1_mae - ridge_mae, 5
        ),
        "winner": (
            "Ridge" if ridge_mae < b1_mae
            else "B1" if ridge_mae > b1_mae
            else "Tie"
        ),
    }

    return result, prediction_rows


def create_figures(results):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib missing: plots skipped.")
        return

    for horizon in HORIZONS:
        rows = sorted(
            (
                r for r in results
                if r["horizon_hours"] == horizon
            ),
            key=lambda r: r["test_genotype"],
        )

        names = [
            r["test_genotype"] for r in rows
        ]
        b1 = [
            r["B1_mae_pixels"] for r in rows
        ]
        ridge = [
            r["B2_mae_pixels"] for r in rows
        ]

        x = np.arange(len(names))
        width = 0.38

        fig, ax = plt.subplots(figsize=(13, 6))

        ax.bar(
            x - width / 2, b1, width,
            label="B1 Time-aware linear"
        )
        ax.bar(
            x + width / 2, ridge, width,
            label="B2 Ridge"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            names, rotation=65, ha="right"
        )
        ax.set_ylabel(
            "Plant-averaged test MAE (pixels)"
        )
        ax.set_title(
            f"E1.4 — {horizon}h forecasting "
            "on unseen genotypes"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.2)

        fig.tight_layout()
        fig.savefig(
            FIGURES
            / f"E1_4_ridge_vs_linear_{horizon}h.png",
            dpi=200,
        )
        plt.close(fig)


def main():
    print("=" * 68)
    print("E1.4 — NESTED LOGO RIDGE REGRESSION")
    print("=" * 68)

    with FOLD_PATH.open(encoding="utf-8") as f:
        config = json.load(f)

    folds = config["folds"]

    if len(folds) != 21:
        raise RuntimeError(
            "Expected exactly 21 outer folds."
        )

    test_genotypes = [
        fold["test_genotype"] for fold in folds
    ]

    if len(set(test_genotypes)) != 21:
        raise RuntimeError(
            "Each genotype must be tested once."
        )

    data = load_data()

    for horizon in HORIZONS:
        data[horizon]["horizon"] = horizon

        if set(data[horizon]["genotype"]) != set(
            test_genotypes
        ):
            raise RuntimeError(
                f"Genotype coverage mismatch: {horizon}h"
            )

        print(
            f"Loaded {horizon}h: "
            f"{len(data[horizon]['y']):,} windows"
        )

    results = []
    alpha_rows = []
    prediction_rows = []

    for horizon in HORIZONS:
        d = data[horizon]

        print(f"\n{'=' * 68}")
        print(f"{horizon}-HOUR RIDGE FORECASTING")
        print("=" * 68)

        for fold in folds:
            test = fold["test_genotype"]

            scores = inner_cv_scores(d, fold)

            selected_alpha = min(
                ALPHAS,
                key=lambda a: (scores[a], a)
            )

            for alpha, score in sorted(scores.items()):
                alpha_rows.append({
                    "horizon_hours": horizon,
                    "outer_fold": fold["outer_fold"],
                    "test_genotype": test,
                    "alpha": alpha,
                    "inner_macro_mae_pixels": round(
                        score, 5
                    ),
                    "selected": int(
                        alpha == selected_alpha
                    ),
                })

            result, predictions = compare_predictions(
                d, fold, selected_alpha
            )

            results.append(result)
            prediction_rows.extend(predictions)

            print(
                f"Fold {fold['outer_fold']:02d}/21 "
                f"| test={test:11s} "
                f"| alpha={selected_alpha:7g} "
                f"| B1={result['B1_mae_pixels']:8.2f} "
                f"| Ridge={result['B2_mae_pixels']:8.2f} "
                f"| winner={result['winner']}"
            )

    save_csv(
        GENOTYPE_RESULTS, results
    )
    save_csv(
        ALPHA_RESULTS, alpha_rows
    )
    save_csv(
        PREDICTIONS, prediction_rows
    )

    summaries = []

    for horizon in HORIZONS:
        rows = [
            r for r in results
            if r["horizon_hours"] == horizon
        ]

        b1 = float(np.mean([
            r["B1_mae_pixels"] for r in rows
        ]))

        ridge = float(np.mean([
            r["B2_mae_pixels"] for r in rows
        ]))

        improvement = b1 - ridge

        summaries.append({
            "horizon_hours": horizon,
            "genotypes": len(rows),
            "B1_macro_mae_pixels": round(b1, 5),
            "B2_ridge_macro_mae_pixels": round(
                ridge, 5
            ),
            "ridge_improvement_pixels": round(
                improvement, 5
            ),
            "ridge_improvement_pct": round(
                100 * improvement / b1, 3
            ),
            "ridge_wins": sum(
                r["winner"] == "Ridge" for r in rows
            ),
            "B1_wins": sum(
                r["winner"] == "B1" for r in rows
            ),
            "ties": sum(
                r["winner"] == "Tie" for r in rows
            ),
        })

    save_csv(SUMMARY, summaries)
    create_figures(results)

    report = [
        "# E1.4 — Nested LOGO Ridge Regression",
        "",
        "## Methodology",
        "",
        "- 21 outer genotype-held-out folds.",
        "- Four inner five-genotype validation folds.",
        f"- Candidate regularization values: {ALPHAS}.",
        "- Separate model per forecasting horizon.",
        "- Feature scaler fitted only on training data.",
        "- Per-genotype/per-plant sample weighting.",
        "- Alpha selected by macro genotype-plant MAE.",
        "- Refit on all 20 development genotypes.",
        "- Only past observations used as predictors.",
        "- Output areas clipped to zero.",
        "",
        "## Results",
        "",
        "| Horizon | B1 MAE | Ridge MAE | "
        "Ridge improvement | Ridge wins |",
        "|---|---:|---:|---:|---:|",
    ]

    for r in summaries:
        report.append(
            f"| {r['horizon_hours']}h "
            f"| {r['B1_macro_mae_pixels']:.2f} "
            f"| {r['B2_ridge_macro_mae_pixels']:.2f} "
            f"| {r['ridge_improvement_pct']:+.2f}% "
            f"| {r['ridge_wins']}/21 |"
        )

    report.extend([
        "",
        "## Important limitations",
        "",
        "- This is a trained-model unseen-genotype test.",
        "- A matched seen-genotype test is still pending.",
        "- Temporal acquisition schedules may confound "
        "genotype-based performance comparisons.",
        "- Forecast windows overlap within plants.",
        "- Future label visibility determines eligibility.",
        "- Annotation identity issues remain unresolved.",
        "- No predictive uncertainty is evaluated here.",
        "",
        "## Next",
        "",
        "Compare results with E1.3, inspect per-genotype "
        "failures, then evaluate gradient-boosted trees "
        "and a matched seen-genotype protocol.",
    ])

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("E1.4 — FINAL MACRO-GENOTYPE RESULTS")
    print("=" * 68)

    for r in summaries:
        print(f"\n{r['horizon_hours']}h:")
        print(
            f"  B1 Linear MAE: "
            f"{r['B1_macro_mae_pixels']:.2f} px"
        )
        print(
            f"  B2 Ridge MAE: "
            f"{r['B2_ridge_macro_mae_pixels']:.2f} px"
        )
        print(
            f"  Ridge improvement: "
            f"{r['ridge_improvement_pct']:+.2f}%"
        )
        print(
            f"  Ridge wins: {r['ridge_wins']}/21"
        )
        print(
            f"  B1 wins: {r['B1_wins']}/21"
        )

    print("\nSaved:")
    print(GENOTYPE_RESULTS)
    print(ALPHA_RESULTS)
    print(PREDICTIONS)
    print(SUMMARY)
    print(REPORT)
    print("\nE1.4 COMPLETE")


if __name__ == "__main__":
    main()
