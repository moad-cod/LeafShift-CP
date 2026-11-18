
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================================================
# E1.5 — GRADIENT-BOOSTED FORECASTING UNDER GENOTYPE SHIFT
#
# B3: Histogram Gradient Boosting
#
# 21 outer LOGO folds
# 4 genotype-disjoint inner folds per outer fold
# Separate models for 24h and 48h
#
# Compares against E1.4b Ridge predictions.
# No raw data or previous experiments modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "data/processed/forecasting/forecast_windows.csv"
)
FOLDS = ROOT / "configs/E1_1_genotype_folds.json"
RIDGE = (
    ROOT / "experiments/E1/tables/"
    "E1_4_ridge_predictions.csv"
)

EXP = ROOT / "experiments/E1"
TABLES = EXP / "tables"
FIGURES = EXP / "figures"
REPORTS = EXP / "reports"

for folder in (TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

PREDICTIONS = TABLES / "E1_5_boosting_predictions.csv"
METRICS = TABLES / "E1_5_genotype_metrics.csv"
SUMMARY = TABLES / "E1_5_summary.csv"
TUNING = TABLES / "E1_5_hyperparameter_selection.csv"
REPORT = REPORTS / "E1_5_boosting_report.md"

HORIZONS = (24, 48)
EXPECTED = {24: 121431, 48: 107214}

# Prespecified candidates; do not alter based on
# outer-test results.
CANDIDATES = [
    {
        "name": "compact",
        "max_iter": 100,
        "max_leaf_nodes": 15,
        "learning_rate": 0.05,
        "l2_regularization": 1.0,
    },
    {
        "name": "flexible",
        "max_iter": 150,
        "max_leaf_nodes": 31,
        "learning_rate": 0.05,
        "l2_regularization": 1.0,
    },
]


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def make_features(times, areas, horizon):
    gaps = np.diff(times)
    changes = np.diff(areas)

    if len(times) != 6 or np.any(gaps <= 0):
        raise ValueError("Invalid historical observations")

    rates = changes / gaps
    mean_t = np.mean(times)
    mean_y = np.mean(areas)
    centered = times - mean_t

    denominator = float(np.dot(centered, centered))
    slope = (
        float(np.dot(centered, areas - mean_y))
        / denominator
        if denominator > 1e-12 else 0.0
    )
    intercept = mean_y - slope * mean_t
    linear = max(0.0, intercept + slope * horizon)

    features = [
        *areas.tolist(),
        *times.tolist(),
        *changes.tolist(),
        *rates.tolist(),
        linear,
        float(areas[-1]),
        float(np.mean(areas)),
        float(times[-1] - times[0]),
        float(max(gaps)),
        float(np.mean(rates)),
        float(areas[-1] / max(areas[0], 1.0)),
    ]

    return np.asarray(features, dtype=np.float32), linear


def load_manifest():
    data = {
        h: {
            "X": [], "y": [], "b1": [],
            "g": [], "p": [], "id": [],
        }
        for h in HORIZONS
    }

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = int(row["horizon_hours"])
            if h not in HORIZONS:
                raise ValueError(f"Unknown horizon {h}")

            times = np.asarray(
                json.loads(row["history_relative_hours"]),
                dtype=np.float64
            )
            areas = np.asarray(
                json.loads(row["history_areas_pixels"]),
                dtype=np.float64
            )

            if abs(times[-1]) > 1e-5:
                raise ValueError("Incorrect prediction cutoff")

            features, linear = make_features(
                times, areas, h
            )

            d = data[h]
            d["X"].append(features)
            d["y"].append(float(row["target_area_pixels"]))
            d["b1"].append(linear)
            d["g"].append(row["genotype"])
            d["p"].append(row["plant_id"])
            d["id"].append(row["window_id"])

    for h, d in data.items():
        d["X"] = np.stack(d["X"])
        d["y"] = np.asarray(d["y"], dtype=np.float64)
        d["b1"] = np.asarray(d["b1"], dtype=np.float64)
        d["g"] = np.asarray(d["g"])
        d["p"] = np.asarray(d["p"])

        if len(d["y"]) != EXPECTED[h]:
            raise ValueError(f"Unexpected {h}h window count")
        if not np.isfinite(d["X"]).all():
            raise ValueError("Nonfinite input features")

    return data


def load_ridge():
    reference = {}

    with RIDGE.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            key = (
                int(row["horizon_hours"]),
                row["window_id"]
            )
            if key in reference:
                raise ValueError("Duplicate Ridge prediction")

            reference[key] = (
                row["genotype"],
                float(row["target_area_pixels"]),
                float(row["B2_ridge"]),
            )

    if len(reference) != sum(EXPECTED.values()):
        raise ValueError("Ridge predictions incomplete")

    return reference


def sample_weights(genotypes, plants):
    group_plants = defaultdict(set)
    plant_counts = Counter(plants)

    for g, p in zip(genotypes, plants):
        group_plants[g].add(p)

    n_genotypes = len(group_plants)

    weights = np.array([
        1.0 / (
            n_genotypes
            * len(group_plants[g])
            * plant_counts[p]
        )
        for g, p in zip(genotypes, plants)
    ])

    return weights * len(weights) / weights.sum()


def fit_model(d, train_mask, params):
    model = HistGradientBoostingRegressor(
        max_iter=params["max_iter"],
        max_leaf_nodes=params["max_leaf_nodes"],
        learning_rate=params["learning_rate"],
        l2_regularization=params["l2_regularization"],
        max_bins=64,
        min_samples_leaf=40,
        early_stopping=False,
        random_state=20261108,
    )

    model.fit(
        d["X"][train_mask],
        d["y"][train_mask],
        sample_weight=sample_weights(
            d["g"][train_mask],
            d["p"][train_mask]
        ),
    )
    return model


def predict(model, X):
    return np.maximum(model.predict(X), 0.0)


def genotype_macro_mae(y, predictions, genotypes, plants):
    plant_errors = defaultdict(list)
    plant_genotypes = {}

    for target, pred, g, p in zip(
        y, predictions, genotypes, plants
    ):
        plant_errors[p].append(abs(float(target - pred)))
        if p in plant_genotypes and plant_genotypes[p] != g:
            raise ValueError("Plant/genotype mismatch")
        plant_genotypes[p] = g

    genotype_errors = defaultdict(list)

    for p, errors in plant_errors.items():
        genotype_errors[plant_genotypes[p]].append(
            float(np.mean(errors))
        )

    return float(np.mean([
        np.mean(values)
        for values in genotype_errors.values()
    ]))


def choose_candidate(d, fold):
    scores = defaultdict(list)
    test_genotype = fold["test_genotype"]

    for inner in fold["inner_folds"]:
        train_g = set(inner["train_genotypes"])
        valid_g = set(inner["validation_genotypes"])

        assert len(train_g) == 15
        assert len(valid_g) == 5
        assert train_g.isdisjoint(valid_g)
        assert test_genotype not in train_g | valid_g

        train_mask = np.isin(d["g"], list(train_g))
        valid_mask = np.isin(d["g"], list(valid_g))

        for params in CANDIDATES:
            model = fit_model(d, train_mask, params)

            predictions = predict(
                model, d["X"][valid_mask]
            )

            score = genotype_macro_mae(
                d["y"][valid_mask],
                predictions,
                d["g"][valid_mask],
                d["p"][valid_mask]
            )

            scores[params["name"]].append(score)

    averages = {
        name: float(np.mean(values))
        for name, values in scores.items()
    }

    # Candidate order resolves exact ties.
    best = min(
        CANDIDATES,
        key=lambda p: averages[p["name"]]
    )

    return best, averages


def main():
    print("=" * 68)
    print("E1.5 — NESTED LOGO GRADIENT BOOSTING")
    print("=" * 68)

    with FOLDS.open(encoding="utf-8") as f:
        folds = json.load(f)["folds"]

    if len(folds) != 21:
        raise ValueError("Expected 21 outer folds")

    test_genotypes = [f["test_genotype"] for f in folds]

    if len(set(test_genotypes)) != 21:
        raise ValueError("Duplicate outer test genotype")

    data = load_manifest()
    ridge = load_ridge()

    results = []
    tuning = []
    prediction_rows = []

    for h in HORIZONS:
        d = data[h]
        print(f"\n{h}h: {len(d['y']):,} windows")

        if set(d["g"]) != set(test_genotypes):
            raise ValueError("Genotype coverage mismatch")

        for fold in folds:
            test = fold["test_genotype"]
            development = set(fold["development_genotypes"])

            assert len(development) == 20
            assert test not in development

            best, scores = choose_candidate(d, fold)

            for name, score in scores.items():
                tuning.append({
                    "horizon_hours": h,
                    "outer_fold": fold["outer_fold"],
                    "test_genotype": test,
                    "candidate": name,
                    "inner_macro_mae": score,
                    "selected": int(name == best["name"]),
                })

            train = np.isin(d["g"], list(development))
            test_mask = d["g"] == test

            model = fit_model(d, train, best)
            b3 = predict(model, d["X"][test_mask])

            ids = np.flatnonzero(test_mask)
            y = d["y"][test_mask]
            g = d["g"][test_mask]
            p = d["p"][test_mask]
            b1 = d["b1"][test_mask]

            b2 = np.empty(len(ids), dtype=np.float64)

            for j, idx in enumerate(ids):
                key = (h, d["id"][idx])

                if key not in ridge:
                    raise ValueError(f"Missing Ridge result: {key}")

                rg, ry, rp = ridge[key]

                if rg != test or abs(ry - y[j]) > 1e-5:
                    raise ValueError("Ridge/test alignment mismatch")

                b2[j] = rp

                prediction_rows.append({
                    "outer_fold": fold["outer_fold"],
                    "horizon_hours": h,
                    "genotype": test,
                    "plant_id": p[j],
                    "window_id": d["id"][idx],
                    "target_area_pixels": y[j],
                    "B1_linear": b1[j],
                    "B2_ridge": b2[j],
                    "B3_boosting": b3[j],
                    "selected_candidate": best["name"],
                })

            scores_test = {
                "B1": genotype_macro_mae(y, b1, g, p),
                "B2": genotype_macro_mae(y, b2, g, p),
                "B3": genotype_macro_mae(y, b3, g, p),
            }

            results.append({
                "outer_fold": fold["outer_fold"],
                "horizon_hours": h,
                "test_genotype": test,
                "plants": len(set(p)),
                "windows": len(y),
                "selected_candidate": best["name"],
                "B1_mae": scores_test["B1"],
                "B2_mae": scores_test["B2"],
                "B3_mae": scores_test["B3"],
                "B3_minus_B2_mae": (
                    scores_test["B3"] - scores_test["B2"]
                ),
                "winner": min(
                    scores_test,
                    key=scores_test.get
                ),
            })

            print(
                f"Fold {fold['outer_fold']:02d}/21 "
                f"| {test:11s} "
                f"| {best['name']:8s} "
                f"| Ridge={scores_test['B2']:8.2f} "
                f"| Boost={scores_test['B3']:8.2f}"
            )

    if len(prediction_rows) != sum(EXPECTED.values()):
        raise ValueError("Incomplete held-out predictions")

    save_csv(METRICS, results)
    save_csv(TUNING, tuning)
    save_csv(PREDICTIONS, prediction_rows)

    summaries = []

    for h in HORIZONS:
        rows = [
            r for r in results
            if r["horizon_hours"] == h
        ]

        b1 = float(np.mean([r["B1_mae"] for r in rows]))
        b2 = float(np.mean([r["B2_mae"] for r in rows]))
        b3 = float(np.mean([r["B3_mae"] for r in rows]))

        summaries.append({
            "horizon_hours": h,
            "genotypes": len(rows),
            "B1_macro_mae": b1,
            "B2_macro_mae": b2,
            "B3_macro_mae": b3,
            "B3_improvement_over_B2_pct": (
                100 * (b2 - b3) / b2
            ),
            "B3_wins_over_B2": sum(
                r["B3_mae"] < r["B2_mae"]
                for r in rows
            ),
            "B2_wins_over_B3": sum(
                r["B2_mae"] < r["B3_mae"]
                for r in rows
            ),
        })

    save_csv(SUMMARY, summaries)

    try:
        import matplotlib.pyplot as plt

        for h in HORIZONS:
            rows = sorted(
                [
                    r for r in results
                    if r["horizon_hours"] == h
                ],
                key=lambda r: r["test_genotype"]
            )

            x = np.arange(len(rows))
            width = 0.38

            fig, ax = plt.subplots(figsize=(13, 6))
            ax.bar(
                x - width / 2,
                [r["B2_mae"] for r in rows],
                width,
                label="B2 Ridge"
            )
            ax.bar(
                x + width / 2,
                [r["B3_mae"] for r in rows],
                width,
                label="B3 Boosting"
            )
            ax.set_xticks(x)
            ax.set_xticklabels(
                [r["test_genotype"] for r in rows],
                rotation=65, ha="right"
            )
            ax.set_ylabel("Plant-averaged MAE (pixels)")
            ax.set_title(
                f"E1.5 — {h}h unseen-genotype forecasting"
            )
            ax.legend()
            fig.tight_layout()
            fig.savefig(
                FIGURES / f"E1_5_ridge_vs_boosting_{h}h.png",
                dpi=200
            )
            plt.close(fig)

    except ImportError:
        print("Matplotlib unavailable; figures skipped")

    report = [
        "# E1.5 — Nested LOGO Gradient Boosting",
        "",
        "## Protocol",
        "- 21 held-out genotype outer folds",
        "- Four genotype-disjoint inner folds",
        "- Candidate selection by inner macro MAE",
        "- No early stopping on a hidden validation subset",
        "- Genotype/plant-balanced training weights",
        "- Historical leaf-area features only",
        "- Separate models for 24h and 48h",
        "",
        "## Results",
        "| Horizon | B1 MAE | B2 MAE | B3 MAE | "
        "B3 improvement vs B2 |",
        "|---|---:|---:|---:|---:|",
    ]

    for r in summaries:
        report.append(
            f"| {r['horizon_hours']}h "
            f"| {r['B1_macro_mae']:.2f} "
            f"| {r['B2_macro_mae']:.2f} "
            f"| {r['B3_macro_mae']:.2f} "
            f"| {r['B3_improvement_over_B2_pct']:+.2f}% |"
        )

    report.extend([
        "",
        "## Limitations",
        "- Outer-test results must not drive tuning decisions.",
        "- Genotype and acquisition schedule may be confounded.",
        "- Visible-area labels and leaf identities need QC.",
        "- Forecasting windows overlap within plants.",
        "- A matched seen-genotype evaluation is pending.",
        "- Predictive uncertainty is not yet evaluated.",
    ])

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("\n" + "=" * 68)
    print("E1.5 — FINAL MACRO-GENOTYPE RESULTS")
    print("=" * 68)

    for r in summaries:
        print(f"\n{r['horizon_hours']}h:")
        print(f"  B1 MAE: {r['B1_macro_mae']:.2f} px")
        print(f"  B2 MAE: {r['B2_macro_mae']:.2f} px")
        print(f"  B3 MAE: {r['B3_macro_mae']:.2f} px")
        print(
            f"  Boosting improvement vs Ridge: "
            f"{r['B3_improvement_over_B2_pct']:+.2f}%"
        )
        print(
            f"  Boosting wins: {r['B3_wins_over_B2']}/21"
        )
        print(
            f"  Ridge wins: {r['B2_wins_over_B3']}/21"
        )

    print("\nSaved:")
    print(METRICS)
    print(TUNING)
    print(PREDICTIONS)
    print(SUMMARY)
    print(REPORT)

    print("\nE1.5 COMPLETE")


if __name__ == "__main__":
    main()
