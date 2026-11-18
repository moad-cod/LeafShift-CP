
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================================================
# E3.1 — NORMALIZED CONFORMAL PREDICTION
#
# U0: pooled absolute residuals
# U1: genotype/plant-balanced absolute residuals
# U2: pooled normalized residuals
# U3: genotype/plant-balanced normalized residuals
#
# Fixed scale: sqrt(max(last observed leaf area, 500))
#
# 21 outer folds x 4 inner assignments x 2 horizons.
# 168 forecasting models; 4 intervals per prediction.
#
# NO target genotype used for fitting or calibration.
# NO adaptive-scale tuning on outer test results.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = ROOT / "data/processed/forecasting/forecast_windows.csv"
FOLDS = ROOT / "configs/E1_1_genotype_folds.json"

BASE = ROOT / "experiments/E3"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

for directory in (TABLES, FIGURES, REPORTS):
    directory.mkdir(parents=True, exist_ok=True)

FOLD_OUT = TABLES / "E3_1_fold_metrics.csv"
SIZE_OUT = TABLES / "E3_1_size_metrics.csv"
SUMMARY_OUT = TABLES / "E3_1_summary.csv"
REPORT_OUT = REPORTS / "E3_1_normalized_conformal.md"

HORIZONS = (24, 48)
METHODS = (
    "U0_pooled",
    "U1_balanced",
    "U2_normalized_pooled",
    "U3_normalized_balanced",
)
NOMINAL = 0.90

PARAMS = {
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
        raise ValueError(f"No output rows: {path}")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def make_features(times, areas, horizon):
    if len(times) != 6 or len(areas) != 6:
        raise ValueError("Expected six observations")

    gaps = np.diff(times)
    changes = np.diff(areas)

    if np.any(gaps <= 0):
        raise ValueError("Invalid time gaps")

    rates = changes / gaps
    mt = float(np.mean(times))
    my = float(np.mean(areas))
    centered = times - mt
    denom = float(np.dot(centered, centered))

    slope = (
        float(np.dot(centered, areas - my)) / denom
        if denom > 1e-12 else 0.0
    )
    linear = max(0.0, my + slope * (horizon - mt))

    values = [
        *areas.tolist(),
        *times.tolist(),
        *changes.tolist(),
        *rates.tolist(),
        linear,
        float(areas[-1]),
        my,
        float(times[-1] - times[0]),
        float(np.max(gaps)),
        float(np.mean(rates)),
        float(areas[-1] / max(areas[0], 1.0)),
    ]

    return np.asarray(values, dtype=np.float32)


def size_category(last_area):
    if last_area < 500:
        return "small_lt500"
    if last_area < 2000:
        return "medium_500to1999"
    return "large_ge2000"


def load_data():
    result = {
        h: {
            "X": [], "y": [], "g": [], "p": [],
            "scale": [], "size": [],
        }
        for h in HORIZONS
    }

    with MANIFEST.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
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
                raise ValueError("History must end at cutoff")

            if not np.isfinite(areas).all() or np.any(areas <= 0):
                raise ValueError("Invalid historical areas")

            genotype = row["genotype"]
            plant = row["plant_id"]

            if not plant.startswith(genotype + "/"):
                raise ValueError("Plant/genotype mismatch")

            last = float(areas[-1])
            d = result[h]

            d["X"].append(make_features(times, areas, h))
            d["y"].append(float(row["target_area_pixels"]))
            d["g"].append(genotype)
            d["p"].append(plant)
            d["scale"].append(math.sqrt(max(last, 500.0)))
            d["size"].append(size_category(last))

    expected = {24: 121431, 48: 107214}

    for h, d in result.items():
        d["X"] = np.stack(d["X"])
        d["y"] = np.asarray(d["y"], dtype=np.float64)
        d["g"] = np.asarray(d["g"])
        d["p"] = np.asarray(d["p"])
        d["scale"] = np.asarray(d["scale"], dtype=np.float64)
        d["size"] = np.asarray(d["size"])

        if len(d["y"]) != expected[h]:
            raise ValueError(f"Unexpected {h}h count")
        if not np.isfinite(d["X"]).all():
            raise ValueError("Nonfinite features")
        if not np.isfinite(d["y"]).all():
            raise ValueError("Nonfinite targets")
        if np.any(d["scale"] <= 0):
            raise ValueError("Nonpositive scales")

    return result


def balanced_weights(genotypes, plants):
    genotype_plants = defaultdict(set)
    plant_counts = Counter(plants)

    for g, p in zip(genotypes, plants):
        genotype_plants[str(g)].add(str(p))

    weights = np.asarray([
        1.0 / (
            len(genotype_plants)
            * len(genotype_plants[str(g)])
            * plant_counts[str(p)]
        )
        for g, p in zip(genotypes, plants)
    ], dtype=np.float64)

    return weights / weights.sum()


def pooled_quantile(residuals):
    n = len(residuals)
    if n == 0:
        raise ValueError("No calibration residuals")

    k = math.ceil((n + 1) * NOMINAL)
    if k > n:
        return float("inf")

    return float(np.partition(residuals, k - 1)[k - 1])


def balanced_quantile(residuals, weights):
    order = np.argsort(residuals)
    cum = np.cumsum(weights[order])
    index = int(np.searchsorted(cum, NOMINAL, side="left"))
    return float(residuals[order[min(index, len(order) - 1)]])


def plant_aggregated_metrics(y, pred, lower, upper, plants):
    coverage = (y >= lower) & (y <= upper)
    width = upper - lower
    score = (
        width
        + 20.0 * np.maximum(lower - y, 0.0)
        + 20.0 * np.maximum(y - upper, 0.0)
    )
    abs_error = np.abs(y - pred)

    records = []

    for plant in np.unique(plants):
        m = plants == plant

        records.append((
            float(np.mean(coverage[m])),
            float(np.mean(width[m])),
            float(np.mean(score[m])),
            float(np.mean(abs_error[m])),
        ))

    values = np.asarray(records)

    return {
        "plants": len(records),
        "windows": len(y),
        "coverage": float(np.mean(values[:, 0])),
        "width": float(np.mean(values[:, 1])),
        "score": float(np.mean(values[:, 2])),
        "mae": float(np.mean(values[:, 3])),
    }


def evaluate_split(d, outer, inner, horizon):
    fit_g = set(inner["train_genotypes"])
    cal_g = set(inner["validation_genotypes"])
    test_g = outer["test_genotype"]

    if len(fit_g) != 15 or len(cal_g) != 5:
        raise ValueError("Invalid split sizes")

    if fit_g & cal_g or test_g in fit_g | cal_g:
        raise ValueError("Genotype leakage")

    fit = np.isin(d["g"], list(fit_g))
    cal = np.isin(d["g"], list(cal_g))
    test = d["g"] == test_g

    if (
        set(d["g"][fit]) != fit_g
        or set(d["g"][cal]) != cal_g
        or set(d["g"][test]) != {test_g}
    ):
        raise ValueError("Incomplete genotype membership")

    model = HistGradientBoostingRegressor(**PARAMS)

    fit_weights = balanced_weights(d["g"][fit], d["p"][fit])
    model.fit(
        d["X"][fit],
        d["y"][fit],
        sample_weight=fit_weights * fit_weights.size,
    )

    cal_pred = np.maximum(model.predict(d["X"][cal]), 0.0)
    cal_resid = np.abs(d["y"][cal] - cal_pred)
    cal_scaled_resid = cal_resid / d["scale"][cal]

    cal_weights = balanced_weights(d["g"][cal], d["p"][cal])

    thresholds = {
        "U0_pooled": pooled_quantile(cal_resid),
        "U1_balanced": balanced_quantile(cal_resid, cal_weights),
        "U2_normalized_pooled": pooled_quantile(cal_scaled_resid),
        "U3_normalized_balanced": balanced_quantile(
            cal_scaled_resid, cal_weights
        ),
    }

    pred = np.maximum(model.predict(d["X"][test]), 0.0)
    target = d["y"][test]
    plant = d["p"][test]
    scale = d["scale"][test]
    sizes = d["size"][test]

    fold_rows = []
    size_rows = []

    for method, threshold in thresholds.items():
        halfwidth = (
            threshold * scale
            if "normalized" in method
            else np.full(len(target), threshold)
        )

        lower = np.maximum(0.0, pred - halfwidth)
        upper = pred + halfwidth

        general = plant_aggregated_metrics(
            target, pred, lower, upper, plant
        )

        base = {
            "horizon_hours": horizon,
            "outer_fold": outer["outer_fold"],
            "inner_fold": inner["inner_fold"],
            "test_genotype": test_g,
            "method": method,
        }

        fold_rows.append({
            **base,
            "threshold": float(threshold),
            "calibration_plants": len(np.unique(d["p"][cal])),
            "calibration_windows": int(np.sum(cal)),
            **general,
        })

        for size in (
            "small_lt500",
            "medium_500to1999",
            "large_ge2000",
        ):
            m = sizes == size
            if not np.any(m):
                continue

            size_rows.append({
                **base,
                "leaf_size": size,
                **plant_aggregated_metrics(
                    target[m], pred[m], lower[m], upper[m], plant[m]
                ),
            })

    return fold_rows, size_rows


def summarize(fold_rows, size_rows):
    summary = []

    for h in HORIZONS:
        for method in METHODS:
            selected = [
                r for r in fold_rows
                if r["horizon_hours"] == h and r["method"] == method
            ]

            grouped = defaultdict(list)
            for r in selected:
                grouped[r["test_genotype"]].append(r)

            if len(grouped) != 21:
                raise ValueError("Missing test genotypes")

            per_genotype = []

            for g, items in grouped.items():
                if sorted(r["inner_fold"] for r in items) != [1, 2, 3, 4]:
                    raise ValueError(f"Missing calibration splits: {g}")

                per_genotype.append({
                    metric: float(np.mean([r[metric] for r in items]))
                    for metric in ("coverage", "width", "score", "mae")
                })

            summary.append({
                "horizon_hours": h,
                "method": method,
                "macro_coverage": float(np.mean([
                    r["coverage"] for r in per_genotype
                ])),
                "macro_width": float(np.mean([
                    r["width"] for r in per_genotype
                ])),
                "macro_interval_score": float(np.mean([
                    r["score"] for r in per_genotype
                ])),
                "macro_mae": float(np.mean([
                    r["mae"] for r in per_genotype
                ])),
                "genotypes_below_90": sum(
                    r["coverage"] < NOMINAL
                    for r in per_genotype
                ),
            })

    size_summary = []

    # Eligibility fixed in E2.3: >=2 plants and >=30
    # windows, for each genotype/split/subgroup.
    for h in HORIZONS:
        for method in METHODS:
            for size in (
                "small_lt500",
                "medium_500to1999",
                "large_ge2000",
            ):
                selected = [
                    r for r in size_rows
                    if r["horizon_hours"] == h
                    and r["method"] == method
                    and r["leaf_size"] == size
                    and r["plants"] >= 2
                    and r["windows"] >= 30
                ]

                grouped = defaultdict(list)
                for r in selected:
                    grouped[r["test_genotype"]].append(r)

                # Only genotypes represented in all
                # four assignments enter this summary.
                complete = [
                    items for items in grouped.values()
                    if len(items) == 4
                ]

                if not complete:
                    continue

                size_summary.append({
                    "horizon_hours": h,
                    "method": method,
                    "leaf_size": size,
                    "complete_genotypes": len(complete),
                    "macro_coverage": float(np.mean([
                        np.mean([r["coverage"] for r in items])
                        for items in complete
                    ])),
                    "macro_width": float(np.mean([
                        np.mean([r["width"] for r in items])
                        for items in complete
                    ])),
                    "macro_interval_score": float(np.mean([
                        np.mean([r["score"] for r in items])
                        for items in complete
                    ])),
                })

    return summary, size_summary


def make_figures(summary, size_summary):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; figures skipped")
        return

    for horizon in HORIZONS:
        selected = [
            r for r in summary if r["horizon_hours"] == horizon
        ]

        fig, ax = plt.subplots(figsize=(9, 5))
        labels = [r["method"] for r in selected]
        values = [100 * r["macro_coverage"] for r in selected]

        ax.bar(labels, values)
        ax.axhline(90, linestyle="--", label="Nominal 90%")
        ax.set_ylabel("Macro-genotype coverage (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"E3.1 — {horizon}h overall coverage")
        ax.tick_params(axis="x", labelrotation=20)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"E3_1_overall_coverage_{horizon}h.png",
            dpi=200
        )
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(10, 5))
        sizes = ["small_lt500", "medium_500to1999", "large_ge2000"]
        x = np.arange(len(sizes))
        width = 0.19

        lookup = {
            (r["method"], r["leaf_size"]): r
            for r in size_summary
            if r["horizon_hours"] == horizon
        }

        for j, method in enumerate(METHODS):
            values = [
                100 * lookup[(method, s)]["macro_coverage"]
                if (method, s) in lookup else np.nan
                for s in sizes
            ]
            ax.bar(
                x + (j - 1.5) * width,
                values, width,
                label=method,
            )

        ax.axhline(90, linestyle="--")
        ax.set_xticks(x)
        ax.set_xticklabels(sizes, rotation=10)
        ax.set_ylabel("Coverage (%)")
        ax.set_ylim(0, 105)
        ax.set_title(f"E3.1 — {horizon}h leaf-size reliability")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"E3_1_leaf_size_coverage_{horizon}h.png",
            dpi=200
        )
        plt.close(fig)


def main():
    print("=" * 70)
    print("E3.1 — NORMALIZED CONFORMAL PREDICTION")
    print("=" * 70)

    with FOLDS.open(encoding="utf-8") as file:
        folds = json.load(file)["folds"]

    if len(folds) != 21:
        raise ValueError("Expected 21 outer folds")

    data = load_data()
    fold_rows = []
    size_rows = []

    for h in HORIZONS:
        print(f"\n{h}h — {len(data[h]['y']):,} windows", flush=True)

        for outer in folds:
            for inner in outer["inner_folds"]:
                results, size_results = evaluate_split(
                    data[h], outer, inner, h
                )
                fold_rows.extend(results)
                size_rows.extend(size_results)

                vals = {r["method"]: r for r in results}
                print(
                    f"  Outer {outer['outer_fold']:02d}/21 "
                    f"| inner {inner['inner_fold']}/4 "
                    f"| {outer['test_genotype']:11s} "
                    f"| U1={vals['U1_balanced']['coverage']:.3f} "
                    f"| U2={vals['U2_normalized_pooled']['coverage']:.3f} "
                    f"| U3={vals['U3_normalized_balanced']['coverage']:.3f}",
                    flush=True,
                )

        # Checkpoint after each horizon.
        write_csv(FOLD_OUT, fold_rows)
        write_csv(SIZE_OUT, size_rows)

    summary, size_summary = summarize(fold_rows, size_rows)

    write_csv(FOLD_OUT, fold_rows)
    write_csv(SIZE_OUT, size_rows)
    write_csv(SUMMARY_OUT, summary)

    size_summary_path = TABLES / "E3_1_size_summary.csv"
    write_csv(size_summary_path, size_summary)

    make_figures(summary, size_summary)

    lines = [
        "# E3.1 — Normalized Conformal Prediction",
        "",
        "## Methods",
        "",
        "- U0: pooled absolute residual calibration.",
        "- U1: genotype/plant-balanced absolute residual calibration.",
        "- U2: pooled normalized residual calibration.",
        "- U3: genotype/plant-balanced normalized residual calibration.",
        "- Normalization scale: sqrt(max(last area, 500)).",
        "- All scale information available at forecast cutoff.",
        "- No scale exponent selected from test performance.",
        "",
        "## Results",
        "",
        "| Horizon | Method | Coverage | Width (px) | Score |",
        "|---|---|---:|---:|---:|",
    ]

    for r in summary:
        lines.append(
            f"| {r['horizon_hours']}h | {r['method']} "
            f"| {r['macro_coverage']:.3f} "
            f"| {r['macro_width']:.2f} "
            f"| {r['macro_interval_score']:.2f} |"
        )

    lines.extend([
        "",
        "## Limitations",
        "",
        "- Same test plants across four calibration assignments.",
        "- Fixed sqrt-area scale is a baseline, not a novel method.",
        "- Different calibration and test genotypes can violate "
        "exchangeability.",
        "- U1 and U3 weighted quantiles have no automatic "
        "finite-sample coverage guarantee.",
        "- Conditional and genotype-specific coverage are empirical.",
        "- Overlapping windows are correlated.",
        "- Annotation uncertainty remains unresolved.",
        "",
        "## Next",
        "",
        "Evaluate whether normalized calibration improves both "
        "large-leaf coverage and interval efficiency. "
        "Then compare against conformalized quantile regression.",
    ])

    REPORT_OUT.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("E3.1 — FINAL RESULTS")
    print("=" * 70)

    for r in summary:
        print(
            f"\n{r['horizon_hours']}h | {r['method']}"
            f"\n  Coverage: {r['macro_coverage']:.3f}"
            f"\n  Width: {r['macro_width']:.2f} px"
            f"\n  Interval score: {r['macro_interval_score']:.2f}"
            f"\n  Genotypes below 90%: {r['genotypes_below_90']}/21"
        )

    for r in size_summary:
        if r["method"] in ("U1_balanced", "U3_normalized_balanced"):
            print(
                f"\n{r['horizon_hours']}h "
                f"| {r['method']} | {r['leaf_size']}"
                f"\n  Coverage: {r['macro_coverage']:.3f}"
                f"\n  Interval score: {r['macro_interval_score']:.2f}"
                f"\n  Complete genotypes: {r['complete_genotypes']}/21"
            )

    print("\nSaved:")
    print(FOLD_OUT)
    print(SIZE_OUT)
    print(SUMMARY_OUT)
    print(size_summary_path)
    print(REPORT_OUT)
    print("\nE3.1 COMPLETE")


if __name__ == "__main__":
    main()
