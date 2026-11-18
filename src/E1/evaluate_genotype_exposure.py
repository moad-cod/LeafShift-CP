from pathlib import Path
from collections import defaultdict, Counter
import csv
import hashlib
import json
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================================================
# E1.6 — GENOTYPE EXPOSURE EXPERIMENT
#
# A: Unseen genotype
# B: Additional same-genotype training plants
# C: Plant-count-matched same-genotype exposure
#
# Same test plants and test windows in all conditions.
# No test plant is ever included in training.
#
# Fixed B3 model configuration from E1.5.
# No hyperparameter tuning on E1.6 test labels.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "data/processed/forecasting/"
    "forecast_windows.csv"
)
FOLD_CONFIG = (
    ROOT / "configs/E1_1_genotype_folds.json"
)

EXP = ROOT / "experiments/E1"
TABLES = EXP / "tables"
FIGURES = EXP / "figures"
REPORTS = EXP / "reports"

for directory in (TABLES, FIGURES, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

SPLITS_FILE = TABLES / "E1_6_plant_splits.json"
METRICS_FILE = TABLES / "E1_6_genotype_metrics.csv"
PLANT_FILE = TABLES / "E1_6_plant_metrics.csv"
SUMMARY_FILE = TABLES / "E1_6_summary.csv"
REPORT_FILE = REPORTS / "E1_6_genotype_exposure.md"

HORIZONS = (24, 48)
EXPECTED_WINDOWS = {24: 121431, 48: 107214}

CONDITIONS = (
    "A_unseen",
    "B_exposed_extra",
    "C_exposed_matched",
)

TEST_FRACTION = 0.30

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


def stable_order(values, salt):
    """Deterministic ordering, independent of Python hash seed."""
    return sorted(
        values,
        key=lambda value: hashlib.sha256(
            f"{salt}|{value}".encode("utf-8")
        ).hexdigest(),
    )


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows available for {path}")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def build_features(times, areas, horizon):
    """
    Same feature definition as E1.5.
    Uses only historical observations.

    Future target timestamps and genotype labels
    are never included as predictive features.
    """
    if len(times) != 6 or len(areas) != 6:
        raise ValueError("Expected six observations")

    gaps = np.diff(times)
    changes = np.diff(areas)

    if np.any(gaps <= 0):
        raise ValueError("Non-increasing historical times")

    rates = changes / gaps

    mean_t = np.mean(times)
    mean_y = np.mean(areas)
    centered = times - mean_t

    denominator = float(
        np.dot(centered, centered)
    )

    slope = (
        float(np.dot(centered, areas - mean_y))
        / denominator
        if denominator > 1e-12
        else 0.0
    )

    intercept = mean_y - slope * mean_t

    linear = max(
        0.0, float(intercept + slope * horizon)
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

    return np.asarray(
        features, dtype=np.float32
    )


def load_manifest():
    data = {
        h: {
            "X": [],
            "y": [],
            "genotype": [],
            "plant": [],
            "window_id": [],
        }
        for h in HORIZONS
    }

    plants_by_genotype = defaultdict(set)

    with MANIFEST.open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            horizon = int(row["horizon_hours"])

            if horizon not in HORIZONS:
                raise ValueError(
                    f"Unexpected horizon {horizon}"
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

            if abs(times[-1]) > 1e-5:
                raise ValueError(
                    "Last historical time must be zero"
                )

            features = build_features(
                times, areas, horizon
            )

            genotype = row["genotype"]
            plant = row["plant_id"]

            if not plant.startswith(genotype + "/"):
                raise ValueError(
                    "Genotype/plant identifier mismatch"
                )

            d = data[horizon]

            d["X"].append(features)
            d["y"].append(
                float(row["target_area_pixels"])
            )
            d["genotype"].append(genotype)
            d["plant"].append(plant)
            d["window_id"].append(row["window_id"])

            plants_by_genotype[genotype].add(plant)

    for horizon, d in data.items():
        d["X"] = np.stack(d["X"])
        d["y"] = np.asarray(
            d["y"], dtype=np.float64
        )
        d["genotype"] = np.asarray(
            d["genotype"], dtype=str
        )
        d["plant"] = np.asarray(
            d["plant"], dtype=str
        )

        if len(d["y"]) != EXPECTED_WINDOWS[horizon]:
            raise ValueError(
                f"Unexpected {horizon}h window count"
            )

        if not np.isfinite(d["X"]).all():
            raise ValueError("Nonfinite model inputs")

        if not np.isfinite(d["y"]).all():
            raise ValueError("Nonfinite target areas")

    return data, plants_by_genotype


def create_plant_splits(
    folds, plants_by_genotype
):
    """
    Select test plants independently of target areas.

    For a genotype with n plants, use approximately
    30% as test plants, retaining at least one
    different plant for genotype exposure.

    The same plant split is used at both horizons.
    """
    result = []

    for fold in folds:
        genotype = fold["test_genotype"]
        development = set(
            fold["development_genotypes"]
        )

        plants = stable_order(
            plants_by_genotype[genotype],
            f"E1_6_TEST_{genotype}",
        )

        n_plants = len(plants)

        if n_plants < 2:
            raise ValueError(
                f"Insufficient plants: {genotype}"
            )

        n_test = max(
            1, int(round(n_plants * TEST_FRACTION))
        )
        n_test = min(n_test, n_plants - 1)

        test_plants = set(plants[:n_test])
        exposure_plants = (
            set(plants) - test_plants
        )

        development_plants = set().union(
            *[
                plants_by_genotype[g]
                for g in development
            ]
        )

        assert not (
            test_plants & development_plants
        )
        assert not (
            test_plants & exposure_plants
        )

        # For C, replace one development plant for
        # every added target-genotype plant.
        #
        # Do not remove the last plant of any
        # development genotype.
        donor_candidates = stable_order(
            development_plants,
            f"E1_6_DONORS_{genotype}",
        )

        donor_plants = set()
        remaining = {
            g: len(plants_by_genotype[g])
            for g in development
        }

        for plant in donor_candidates:
            donor_genotype = plant.split("/")[0]

            if remaining[donor_genotype] <= 1:
                continue

            donor_plants.add(plant)
            remaining[donor_genotype] -= 1

            if len(donor_plants) == len(
                exposure_plants
            ):
                break

        if len(donor_plants) != len(
            exposure_plants
        ):
            raise RuntimeError(
                f"Not enough eligible donor plants: {genotype}"
            )

        a_train = development_plants

        b_train = (
            development_plants | exposure_plants
        )

        c_train = (
            (development_plants - donor_plants)
            | exposure_plants
        )

        assert len(a_train) == len(c_train)
        assert len(b_train) == (
            len(a_train) + len(exposure_plants)
        )

        for training_set in (
            a_train, b_train, c_train
        ):
            assert training_set.isdisjoint(
                test_plants
            )

        result.append({
            "outer_fold": fold["outer_fold"],
            "test_genotype": genotype,
            "test_plants": sorted(test_plants),
            "exposure_plants": sorted(
                exposure_plants
            ),
            "removed_development_plants": sorted(
                donor_plants
            ),
            "training_plant_counts": {
                "A_unseen": len(a_train),
                "B_exposed_extra": len(b_train),
                "C_exposed_matched": len(c_train),
            },
            "training_plants": {
                "A_unseen": sorted(a_train),
                "B_exposed_extra": sorted(b_train),
                "C_exposed_matched": sorted(c_train),
            },
        })

    return result


def balanced_weights(genotypes, plants):
    """
    Equal total weight per genotype,
    equal total weight per plant within genotype,
    equal weight per window within plant.

    Same policy used in E1.5.
    """
    genotype_plants = defaultdict(set)
    plant_counts = Counter(plants)

    for genotype, plant in zip(
        genotypes, plants
    ):
        genotype_plants[genotype].add(plant)

    n_genotypes = len(genotype_plants)

    weights = np.asarray([
        1.0 / (
            n_genotypes
            * len(genotype_plants[g])
            * plant_counts[p]
        )
        for g, p in zip(genotypes, plants)
    ], dtype=np.float64)

    return weights * (
        len(weights) / weights.sum()
    )


def train_model(d, train_mask):
    model = HistGradientBoostingRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        d["X"][train_mask],
        d["y"][train_mask],
        sample_weight=balanced_weights(
            d["genotype"][train_mask],
            d["plant"][train_mask],
        ),
    )

    return model


def compute_plant_metrics(
    y, predictions, plants
):
    """
    Plant-level MAE and RMSE.

    Each plant's windows are aggregated first.
    """
    errors = defaultdict(list)

    for target, prediction, plant in zip(
        y, predictions, plants
    ):
        errors[str(plant)].append(
            float(prediction - target)
        )

    results = {}

    for plant, values in errors.items():
        a = np.asarray(
            values, dtype=np.float64
        )

        results[plant] = {
            "windows": len(a),
            "mae": float(np.mean(np.abs(a))),
            "rmse": float(
                np.sqrt(np.mean(a * a))
            ),
        }

    return results


def main():
    print("=" * 68)
    print("E1.6 — GENOTYPE EXPOSURE EVALUATION")
    print("=" * 68)

    with FOLD_CONFIG.open(
        encoding="utf-8"
    ) as f:
        folds = json.load(f)["folds"]

    if len(folds) != 21:
        raise ValueError(
            "Expected 21 outer genotype folds"
        )

    data, plants_by_genotype = load_manifest()

    splits = create_plant_splits(
        folds, plants_by_genotype
    )

    SPLITS_FILE.write_text(
        json.dumps({
            "experiment": "E1.6",
            "test_fraction": TEST_FRACTION,
            "model_parameters": MODEL_PARAMS,
            "splits": splits,
        }, indent=2),
        encoding="utf-8",
    )

    genotype_rows = []
    plant_rows = []

    for horizon in HORIZONS:
        d = data[horizon]

        print(
            f"\n{'=' * 68}\n"
            f"{horizon}-HOUR FORECASTING\n"
            f"{'=' * 68}"
        )

        for split in splits:
            genotype = split["test_genotype"]
            test_plants = set(
                split["test_plants"]
            )

            test_mask = np.isin(
                d["plant"], list(test_plants)
            )

            if not np.any(test_mask):
                raise ValueError(
                    f"No test windows: {genotype}"
                )

            assert set(
                d["genotype"][test_mask]
            ) == {genotype}

            y_test = d["y"][test_mask]
            p_test = d["plant"][test_mask]

            condition_metrics = {}

            for condition in CONDITIONS:
                training_plants = set(
                    split["training_plants"][
                        condition
                    ]
                )

                assert training_plants.isdisjoint(
                    test_plants
                )

                train_mask = np.isin(
                    d["plant"],
                    list(training_plants),
                )

                if not np.any(train_mask):
                    raise ValueError(
                        "Empty training set"
                    )

                if np.any(
                    train_mask & test_mask
                ):
                    raise AssertionError(
                        "Training/test plant leakage"
                    )

                if condition == "A_unseen":
                    if genotype in set(
                        d["genotype"][train_mask]
                    ):
                        raise AssertionError(
                            "Target genotype leaked into A"
                        )
                else:
                    if genotype not in set(
                        d["genotype"][train_mask]
                    ):
                        raise AssertionError(
                            "Target genotype absent from B/C"
                        )

                model = train_model(
                    d, train_mask
                )

                prediction = np.maximum(
                    model.predict(
                        d["X"][test_mask]
                    ),
                    0.0,
                )

                metrics = compute_plant_metrics(
                    y_test,
                    prediction,
                    p_test,
                )

                condition_metrics[condition] = metrics

                genotype_mae = float(
                    np.mean([
                        m["mae"]
                        for m in metrics.values()
                    ])
                )

                genotype_rmse = float(
                    np.mean([
                        m["rmse"]
                        for m in metrics.values()
                    ])
                )

                genotype_rows.append({
                    "horizon_hours": horizon,
                    "outer_fold": split["outer_fold"],
                    "test_genotype": genotype,
                    "condition": condition,
                    "test_plants": len(test_plants),
                    "test_windows": len(y_test),
                    "training_plants": len(
                        training_plants
                    ),
                    "training_windows": int(
                        np.sum(train_mask)
                    ),
                    "plant_mean_mae": genotype_mae,
                    "plant_mean_rmse": genotype_rmse,
                })

                for plant, m in metrics.items():
                    plant_rows.append({
                        "horizon_hours": horizon,
                        "test_genotype": genotype,
                        "plant_id": plant,
                        "condition": condition,
                        "windows": m["windows"],
                        "mae_pixels": m["mae"],
                        "rmse_pixels": m["rmse"],
                    })

            a = float(np.mean([
                x["mae"]
                for x in condition_metrics[
                    "A_unseen"
                ].values()
            ]))
            b = float(np.mean([
                x["mae"]
                for x in condition_metrics[
                    "B_exposed_extra"
                ].values()
            ]))
            c = float(np.mean([
                x["mae"]
                for x in condition_metrics[
                    "C_exposed_matched"
                ].values()
            ]))

            print(
                f"Fold {split['outer_fold']:02d}/21 "
                f"| {genotype:11s} "
                f"| unseen={a:8.2f} "
                f"| extra={b:8.2f} "
                f"| matched={c:8.2f}"
            )

    save_csv(METRICS_FILE, genotype_rows)
    save_csv(PLANT_FILE, plant_rows)

    summaries = []

    for horizon in HORIZONS:
        rows = [
            r for r in genotype_rows
            if r["horizon_hours"] == horizon
        ]

        by_condition = {
            condition: {
                r["test_genotype"]: r
                for r in rows
                if r["condition"] == condition
            }
            for condition in CONDITIONS
        }

        genotypes = sorted(
            by_condition["A_unseen"]
        )

        if len(genotypes) != 21:
            raise ValueError(
                "Expected 21 test genotypes"
            )

        for condition in CONDITIONS:
            if set(by_condition[condition]) != set(
                genotypes
            ):
                raise ValueError(
                    "Inconsistent genotype coverage"
                )

        # Paired comparisons: identical test plants
        # and windows in all three conditions.
        a = np.asarray([
            by_condition["A_unseen"][g][
                "plant_mean_mae"
            ]
            for g in genotypes
        ])

        b = np.asarray([
            by_condition["B_exposed_extra"][g][
                "plant_mean_mae"
            ]
            for g in genotypes
        ])

        c = np.asarray([
            by_condition["C_exposed_matched"][g][
                "plant_mean_mae"
            ]
            for g in genotypes
        ])

        summaries.append({
            "horizon_hours": horizon,
            "genotypes": len(genotypes),
            "unseen_macro_mae": float(
                np.mean(a)
            ),
            "exposed_extra_macro_mae": float(
                np.mean(b)
            ),
            "exposed_matched_macro_mae": float(
                np.mean(c)
            ),
            "extra_exposure_improvement_pct": float(
                100 * np.mean(a - b) / np.mean(a)
            ),
            "matched_exposure_improvement_pct": float(
                100 * np.mean(a - c) / np.mean(a)
            ),
            "extra_exposure_wins": int(
                np.sum(b < a)
            ),
            "matched_exposure_wins": int(
                np.sum(c < a)
            ),
        })

    save_csv(SUMMARY_FILE, summaries)

    # ----------------------------------------------------
    # Figures
    # ----------------------------------------------------

    try:
        import matplotlib.pyplot as plt

        for horizon in HORIZONS:
            rows = [
                r for r in genotype_rows
                if r["horizon_hours"] == horizon
            ]

            names = sorted({
                r["test_genotype"]
                for r in rows
            })

            lookup = {
                (r["test_genotype"], r["condition"]):
                    r["plant_mean_mae"]
                for r in rows
            }

            x = np.arange(len(names))
            width = 0.26

            fig, ax = plt.subplots(
                figsize=(14, 6)
            )

            for position, condition in enumerate(
                CONDITIONS
            ):
                values = [
                    lookup[(g, condition)]
                    for g in names
                ]

                ax.bar(
                    x + (position - 1) * width,
                    values,
                    width,
                    label=condition,
                )

            ax.set_xticks(x)
            ax.set_xticklabels(
                names,
                rotation=65,
                ha="right",
            )
            ax.set_ylabel(
                "Test plant-averaged MAE (pixels)"
            )
            ax.set_title(
                f"E1.6 — {horizon}h: "
                "genotype training exposure"
            )
            ax.legend()
            fig.tight_layout()

            fig.savefig(
                FIGURES
                / f"E1_6_genotype_exposure_{horizon}h.png",
                dpi=200,
            )
            plt.close(fig)

    except ImportError:
        print("Matplotlib unavailable; figures skipped.")

    # ----------------------------------------------------
    # Report
    # ----------------------------------------------------

    report = [
        "# E1.6 — Genotype Exposure Evaluation",
        "",
        "## Protocol",
        "",
        "- Same held-out test plants for A, B, and C.",
        "- A: train only on the other 20 genotypes.",
        "- B: A plus other plants of target genotype.",
        "- C: B with an equal count of development "
        "plants removed.",
        "- No evaluated plant contributes training windows.",
        "- Fixed E1.5 flexible boosting configuration.",
        "- Training sample weights balance genotype, "
        "plant, and within-plant windows.",
        "- Metrics average windows within plants, "
        "then plants within genotypes.",
        "",
        "## Results",
        "",
        "| Horizon | Unseen MAE | Exposed-extra MAE | "
        "Exposed-matched MAE |",
        "|---|---:|---:|---:|",
    ]

    for r in summaries:
        report.append(
            f"| {r['horizon_hours']}h "
            f"| {r['unseen_macro_mae']:.2f} "
            f"| {r['exposed_extra_macro_mae']:.2f} "
            f"| {r['exposed_matched_macro_mae']:.2f} |"
        )

    report.extend([
        "",
        "## Limitations",
        "",
        "- E1.6 uses a new test-plant subset; its A score "
        "must not be compared directly with E1.5 headline MAE.",
        "- B adds training plants and forecasting windows.",
        "- C matches plant counts but not necessarily "
        "window counts or training genotype composition.",
        "- Test plants supply their own six observed "
        "historical areas at prediction time.",
        "- Differences do not establish a causal "
        "genotype effect.",
        "- Genotype and acquisition schedules can "
        "remain confounded.",
        "- Overlapping forecast windows are correlated.",
        "- Leaf identity and target visibility require "
        "additional sensitivity analysis.",
    ])

    REPORT_FILE.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 68)
    print("E1.6 — FINAL RESULTS")
    print("=" * 68)

    for r in summaries:
        print(f"\n{r['horizon_hours']}h:")
        print(
            "  A Unseen: "
            f"{r['unseen_macro_mae']:.2f} px"
        )
        print(
            "  B Exposed-extra: "
            f"{r['exposed_extra_macro_mae']:.2f} px"
        )
        print(
            "  C Exposed-matched: "
            f"{r['exposed_matched_macro_mae']:.2f} px"
        )
        print(
            "  Exposure-extra improvement: "
            f"{r['extra_exposure_improvement_pct']:+.2f}%"
        )
        print(
            "  Exposure-matched improvement: "
            f"{r['matched_exposure_improvement_pct']:+.2f}%"
        )
        print(
            "  B beats A: "
            f"{r['extra_exposure_wins']}/21"
        )
        print(
            "  C beats A: "
            f"{r['matched_exposure_wins']}/21"
        )

    print("\nSaved:")
    print(SPLITS_FILE)
    print(METRICS_FILE)
    print(PLANT_FILE)
    print(SUMMARY_FILE)
    print(REPORT_FILE)
    print("\nE1.6 COMPLETE")


if __name__ == "__main__":
    main()