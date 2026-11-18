
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================================================
# E2.4 — CALIBRATION-SPLIT SENSITIVITY
#
# 21 outer genotypes x 4 inner assignments x 2 horizons.
#
# U0: pooled residual quantile
# U1: genotype/plant-balanced residual quantile
#
# Evaluates 90% coverage, interval score, and
# coverage by leaf-size subgroup.
#
# Fixed E1.5 B3 model; no hyperparameter tuning.
# No held-out genotype used for fitting/calibration.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "data/processed/forecasting/"
    "forecast_windows.csv"
)
FOLDS = ROOT / "configs/E1_1_genotype_folds.json"

BASE = ROOT / "experiments/E2"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

for directory in (TABLES, FIGURES, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

RESULTS = TABLES / "E2_4_split_metrics.csv"
SIZE_RESULTS = TABLES / "E2_4_size_metrics.csv"
SUMMARY = TABLES / "E2_4_summary.csv"
REPORT = REPORTS / "E2_4_calibration_sensitivity.md"

HORIZONS = (24, 48)
METHODS = ("U0_pooled", "U1_balanced")
EXPECTED = {24: 121431, 48: 107214}

NOMINAL = 0.90
ALPHA = 0.10

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


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def features(times, areas, horizon):
    # Exactly the E1.5/E2.1 feature definition.
    gaps = np.diff(times)
    changes = np.diff(areas)

    if len(times) != 6 or len(areas) != 6:
        raise ValueError("Expected six measurements")

    if np.any(gaps <= 0):
        raise ValueError("Invalid historical timestamps")

    rates = changes / gaps
    mt = np.mean(times)
    my = np.mean(areas)

    centered = times - mt
    denominator = float(np.dot(centered, centered))

    slope = (
        float(np.dot(centered, areas - my))
        / denominator
        if denominator > 1e-12
        else 0.0
    )

    intercept = my - slope * mt
    linear = max(
        0.0, float(intercept + slope * horizon)
    )

    values = [
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

    return np.asarray(values, dtype=np.float32)


def leaf_size(area):
    if area < 500:
        return "small_lt500"
    if area < 2000:
        return "medium_500to1999"
    return "large_ge2000"


def load_data():
    data = {
        h: {
            "X": [],
            "y": [],
            "g": [],
            "p": [],
            "size": [],
        }
        for h in HORIZONS
    }

    with MANIFEST.open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            h = int(row["horizon_hours"])

            if h not in HORIZONS:
                raise ValueError(f"Invalid horizon {h}")

            times = np.asarray(
                json.loads(row["history_relative_hours"]),
                dtype=np.float64,
            )
            areas = np.asarray(
                json.loads(row["history_areas_pixels"]),
                dtype=np.float64,
            )

            if abs(times[-1]) > 1e-5:
                raise ValueError("Invalid history cutoff")

            g = row["genotype"]
            p = row["plant_id"]

            if not p.startswith(g + "/"):
                raise ValueError("Genotype/plant mismatch")

            d = data[h]
            d["X"].append(features(times, areas, h))
            d["y"].append(float(row["target_area_pixels"]))
            d["g"].append(g)
            d["p"].append(p)
            d["size"].append(leaf_size(areas[-1]))

    for h, d in data.items():
        d["X"] = np.stack(d["X"])
        d["y"] = np.asarray(d["y"], dtype=np.float64)
        d["g"] = np.asarray(d["g"])
        d["p"] = np.asarray(d["p"])
        d["size"] = np.asarray(d["size"])

        if len(d["y"]) != EXPECTED[h]:
            raise ValueError(f"{h}h count mismatch")
        if not np.isfinite(d["X"]).all():
            raise ValueError("Nonfinite model features")
        if not np.isfinite(d["y"]).all():
            raise ValueError("Nonfinite target values")

    return data


def balanced_weights(genotypes, plants):
    genotype_plants = defaultdict(set)
    counts = Counter(plants)

    for g, p in zip(genotypes, plants):
        genotype_plants[str(g)].add(str(p))

    n_g = len(genotype_plants)

    weights = np.asarray([
        1.0 / (
            n_g
            * len(genotype_plants[str(g)])
            * counts[str(p)]
        )
        for g, p in zip(genotypes, plants)
    ])

    return weights * len(weights) / weights.sum()


def pooled_quantile(residuals):
    n = len(residuals)
    k = math.ceil((n + 1) * NOMINAL)

    if n == 0:
        raise ValueError("Empty calibration set")

    if k > n:
        return float("inf")

    return float(
        np.partition(residuals, k - 1)[k - 1]
    )


def balanced_quantile(residuals, g, p):
    weights = balanced_weights(g, p)
    weights = weights / weights.sum()

    order = np.argsort(residuals)
    cum = np.cumsum(weights[order])

    idx = np.searchsorted(
        cum, NOMINAL, side="left"
    )

    return float(residuals[order[min(
        int(idx), len(order) - 1
    )]])


def evaluate_by_plants(
    targets, predictions, lower, upper, plants
):
    covered = (targets >= lower) & (targets <= upper)

    score = (
        upper - lower
        + 20 * np.maximum(lower - targets, 0)
        + 20 * np.maximum(targets - upper, 0)
    )

    error = np.abs(targets - predictions)
    result = []

    for plant in np.unique(plants):
        mask = plants == plant

        result.append({
            "coverage": float(np.mean(covered[mask])),
            "width": float(np.mean(
                (upper - lower)[mask]
            )),
            "score": float(np.mean(score[mask])),
            "mae": float(np.mean(error[mask])),
            "windows": int(np.sum(mask)),
        })

    return {
        "plants": len(result),
        "windows": len(targets),
        "coverage": float(np.mean([
            r["coverage"] for r in result
        ])),
        "width": float(np.mean([
            r["width"] for r in result
        ])),
        "score": float(np.mean([
            r["score"] for r in result
        ])),
        "mae": float(np.mean([
            r["mae"] for r in result
        ])),
    }


def run_one(d, outer, inner, horizon):
    fit_g = set(inner["train_genotypes"])
    cal_g = set(inner["validation_genotypes"])
    test_g = outer["test_genotype"]

    if len(fit_g) != 15 or len(cal_g) != 5:
        raise ValueError("Invalid genotype split sizes")

    if fit_g & cal_g or test_g in fit_g | cal_g:
        raise ValueError("Genotype leakage detected")

    fit = np.isin(d["g"], list(fit_g))
    cal = np.isin(d["g"], list(cal_g))
    test = d["g"] == test_g

    if not (fit.any() and cal.any() and test.any()):
        raise ValueError("Empty fit/calibration/test role")

    if (
        set(d["g"][fit]) != fit_g
        or set(d["g"][cal]) != cal_g
        or set(d["g"][test]) != {test_g}
    ):
        raise ValueError("Unexpected genotype membership")

    model = HistGradientBoostingRegressor(
        **MODEL_PARAMS
    )

    model.fit(
        d["X"][fit],
        d["y"][fit],
        sample_weight=balanced_weights(
            d["g"][fit], d["p"][fit]
        ),
    )

    cal_predictions = np.maximum(
        model.predict(d["X"][cal]), 0.0
    )
    residuals = np.abs(
        d["y"][cal] - cal_predictions
    )

    thresholds = {
        "U0_pooled": pooled_quantile(residuals),
        "U1_balanced": balanced_quantile(
            residuals,
            d["g"][cal],
            d["p"][cal],
        ),
    }

    predictions = np.maximum(
        model.predict(d["X"][test]), 0.0
    )
    y = d["y"][test]
    p = d["p"][test]
    size = d["size"][test]

    rows = []
    size_rows = []

    for method, q in thresholds.items():
        lower = np.maximum(0.0, predictions - q)
        upper = predictions + q

        metric = evaluate_by_plants(
            y, predictions, lower, upper, p
        )

        rows.append({
            "horizon_hours": horizon,
            "outer_fold": outer["outer_fold"],
            "inner_fold": inner["inner_fold"],
            "test_genotype": test_g,
            "method": method,
            "calibration_plants": len(
                set(d["p"][cal])
            ),
            "calibration_windows": len(residuals),
            "threshold_pixels": q,
            **metric,
        })

        for category in (
            "small_lt500",
            "medium_500to1999",
            "large_ge2000",
        ):
            mask = size == category

            if not mask.any():
                continue

            group_metric = evaluate_by_plants(
                y[mask],
                predictions[mask],
                lower[mask],
                upper[mask],
                p[mask],
            )

            size_rows.append({
                "horizon_hours": horizon,
                "outer_fold": outer["outer_fold"],
                "inner_fold": inner["inner_fold"],
                "test_genotype": test_g,
                "method": method,
                "leaf_size": category,
                **group_metric,
            })

    return rows, size_rows


def summarize(rows):
    summary = []

    for horizon in HORIZONS:
        for method in METHODS:
            selected = [
                r for r in rows
                if r["horizon_hours"] == horizon
                and r["method"] == method
            ]

            if len(selected) != 84:
                raise ValueError(
                    f"Expected 84 results for "
                    f"{horizon}h {method}"
                )

            # First average four assignments for each
            # held-out genotype; then average genotypes.
            by_genotype = defaultdict(list)

            for r in selected:
                by_genotype[r["test_genotype"]].append(r)

            if len(by_genotype) != 21:
                raise ValueError(
                    "Incomplete held-out genotype coverage"
                )

            per_genotype = []

            for genotype, items in by_genotype.items():
                if sorted(
                    x["inner_fold"] for x in items
                ) != [1, 2, 3, 4]:
                    raise ValueError(
                        f"Missing calibration rotation: "
                        f"{genotype}"
                    )

                per_genotype.append({
                    "coverage": float(np.mean([
                        x["coverage"] for x in items
                    ])),
                    "width": float(np.mean([
                        x["width"] for x in items
                    ])),
                    "score": float(np.mean([
                        x["score"] for x in items
                    ])),
                    "coverage_sd": float(np.std(
                        [x["coverage"] for x in items],
                        ddof=1,
                    )),
                })

            summary.append({
                "horizon_hours": horizon,
                "method": method,
                "outer_genotypes": 21,
                "calibration_assignments": 4,
                "macro_coverage": float(np.mean([
                    x["coverage"] for x in per_genotype
                ])),
                "macro_width_pixels": float(np.mean([
                    x["width"] for x in per_genotype
                ])),
                "macro_interval_score": float(np.mean([
                    x["score"] for x in per_genotype
                ])),
                "mean_within_genotype_coverage_sd": (
                    float(np.mean([
                        x["coverage_sd"]
                        for x in per_genotype
                    ]))
                ),
                "genotypes_mean_coverage_below_90": sum(
                    x["coverage"] < NOMINAL
                    for x in per_genotype
                ),
                "genotypes_mean_coverage_below_80": sum(
                    x["coverage"] < 0.8
                    for x in per_genotype
                ),
            })

    return summary


def make_figures(rows, size_rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable: figures skipped.")
        return

    for horizon in HORIZONS:
        fig, ax = plt.subplots(figsize=(9, 5))

        for method in METHODS:
            means = []

            for inner in (1, 2, 3, 4):
                selected = [
                    r["coverage"] for r in rows
                    if r["horizon_hours"] == horizon
                    and r["method"] == method
                    and r["inner_fold"] == inner
                ]

                if len(selected) != 21:
                    raise ValueError(
                        "Incomplete figure data"
                    )

                means.append(float(np.mean(selected)))

            ax.plot(
                [1, 2, 3, 4],
                means,
                marker="o",
                label=method,
            )

        ax.axhline(
            NOMINAL, linestyle="--",
            label="Nominal 90%",
        )
        ax.set_ylim(0, 1.05)
        ax.set_xticks([1, 2, 3, 4])
        ax.set_xlabel("Calibration inner fold")
        ax.set_ylabel("Macro plant-averaged coverage")
        ax.set_title(
            f"E2.4 — {horizon}h calibration sensitivity"
        )
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            FIGURES
            / f"E2_4_calibration_sensitivity_{horizon}h.png",
            dpi=200,
        )
        plt.close(fig)

        # Summarize size-specific coverage across
        # the four calibration assignments.
        fig, ax = plt.subplots(figsize=(9, 5))

        labels = [
            "small_lt500",
            "medium_500to1999",
            "large_ge2000",
        ]

        x = np.arange(len(labels))
        width = 0.38

        for index, method in enumerate(METHODS):
            coverage = []

            for category in labels:
                selected = [
                    r["coverage"] for r in size_rows
                    if r["horizon_hours"] == horizon
                    and r["method"] == method
                    and r["leaf_size"] == category
                    and r["plants"] >= 2
                    and r["windows"] >= 30
                ]

                # Descriptive: genotype-fold averages.
                coverage.append(
                    float(np.mean(selected))
                    if selected else np.nan
                )

            ax.bar(
                x + (index - 0.5) * width,
                coverage, width,
                label=method,
            )

        ax.axhline(
            NOMINAL, linestyle="--",
            label="Nominal 90%",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=15)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("Mean eligible genotype-fold coverage")
        ax.set_title(
            f"E2.4 — {horizon}h leaf-size coverage"
        )
        ax.legend()
        fig.tight_layout()

        fig.savefig(
            FIGURES
            / f"E2_4_leaf_size_sensitivity_{horizon}h.png",
            dpi=200,
        )
        plt.close(fig)


def main():
    print("=" * 70)
    print("E2.4 — CALIBRATION-SPLIT SENSITIVITY")
    print("=" * 70)

    with FOLDS.open(encoding="utf-8") as f:
        folds = json.load(f)["folds"]

    if len(folds) != 21:
        raise ValueError("Expected 21 outer folds")

    data = load_data()

    rows = []
    size_rows = []

    for horizon in HORIZONS:
        d = data[horizon]
        print(
            f"\n{horizon}h — "
            f"{len(d['y']):,} forecast windows",
            flush=True,
        )

        for outer in folds:
            inner_folds = outer["inner_folds"]

            if len(inner_folds) != 4:
                raise ValueError(
                    "Expected four calibration splits"
                )

            for inner in inner_folds:
                result, size_result = run_one(
                    d, outer, inner, horizon
                )

                rows.extend(result)
                size_rows.extend(size_result)

                u0 = next(
                    r for r in result
                    if r["method"] == "U0_pooled"
                )
                u1 = next(
                    r for r in result
                    if r["method"] == "U1_balanced"
                )

                print(
                    f"  Outer {outer['outer_fold']:02d}/21 "
                    f"| inner {inner['inner_fold']}/4 "
                    f"| {outer['test_genotype']:11s} "
                    f"| U0={u0['coverage']:.3f} "
                    f"| U1={u1['coverage']:.3f}",
                    flush=True,
                )

        # Checkpoint: save all completed results
        # after each horizon.
        write_csv(RESULTS, rows)
        write_csv(SIZE_RESULTS, size_rows)

    summary = summarize(rows)

    write_csv(RESULTS, rows)
    write_csv(SIZE_RESULTS, size_rows)
    write_csv(SUMMARY, summary)

    make_figures(rows, size_rows)

    lines = [
        "# E2.4 — Calibration-Split Sensitivity",
        "",
        "## Protocol",
        "",
        "- 21 held-out genotypes.",
        "- Four predefined calibration assignments.",
        "- 15 fitting, five calibration genotypes.",
        "- Separate evaluation at 24h and 48h.",
        "- Fixed B3 gradient-boosting configuration.",
        "- No test-genotype fitting or calibration.",
        "- U0 pooled conformal residual quantile.",
        "- U1 genotype/plant-balanced empirical quantile.",
        "- Primary aggregation: windows, plants, genotypes.",
        "",
        "## Overall results",
        "",
        "| Horizon | Method | Coverage | Width (px) | "
        "Interval score | Genotypes below 90% |",
        "|---|---|---:|---:|---:|---:|",
    ]

    for r in summary:
        lines.append(
            f"| {r['horizon_hours']}h "
            f"| {r['method']} "
            f"| {r['macro_coverage']:.3f} "
            f"| {r['macro_width_pixels']:.2f} "
            f"| {r['macro_interval_score']:.2f} "
            f"| {r['genotypes_mean_coverage_below_90']}/21 |"
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- The four splits share the same outer test plants.",
        "- Inner-fold results are correlated replications.",
        "- Models are retrained when calibration roles rotate.",
        "- Calibration assignment influences both fitting "
        "and calibration data composition.",
        "- Leaf-size coverage summaries use supported "
        "genotype-fold groups, not independent samples.",
        "- No finite-sample genotype-shift coverage "
        "guarantee is asserted.",
        "- Annotation and acquisition confounding remain.",
        "",
        "## Decision",
        "",
        "Use this sensitivity analysis to decide whether "
        "the calibration gap and leaf-size heterogeneity "
        "are sufficiently stable to motivate a proposed "
        "research contribution.",
    ])

    REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("E2.4 — FINAL RESULTS")
    print("=" * 70)

    for r in summary:
        print(
            f"\n{r['horizon_hours']}h "
            f"| {r['method']}"
        )
        print(
            "  Macro coverage: "
            f"{r['macro_coverage']:.3f}"
        )
        print(
            "  Mean interval width: "
            f"{r['macro_width_pixels']:.2f} px"
        )
        print(
            "  Mean interval score: "
            f"{r['macro_interval_score']:.2f}"
        )
        print(
            "  Genotypes below 90%: "
            f"{r['genotypes_mean_coverage_below_90']}/21"
        )
        print(
            "  Mean calibration-split coverage SD: "
            f"{r['mean_within_genotype_coverage_sd']:.4f}"
        )

    print("\nSaved:")
    print(RESULTS)
    print(SIZE_RESULTS)
    print(SUMMARY)
    print(REPORT)
    print("\nE2.4 COMPLETE")


if __name__ == "__main__":
    main()
