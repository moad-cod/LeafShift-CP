
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json
import math

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

# ============================================================
# E3.2 — CONFORMALIZED QUANTILE REGRESSION
#
# U4: CQR with pooled calibration
# U5: CQR with genotype/plant-balanced calibration
#
# 21 outer x 4 inner x 2 horizons.
# Fixed model settings; no test-based tuning.
#
# Compare against E3.1 U1 and U3 on identical splits.
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

FOLD_OUT = TABLES / "E3_2_fold_metrics.csv"
SIZE_OUT = TABLES / "E3_2_size_metrics.csv"
SUMMARY_OUT = TABLES / "E3_2_summary.csv"
SIZE_SUMMARY_OUT = TABLES / "E3_2_size_summary.csv"
REPORT_OUT = REPORTS / "E3_2_cqr_baselines.md"

REFERENCE = TABLES / "E3_1_fold_metrics.csv"

HORIZONS = (24, 48)
METHODS = ("U4_CQR_pooled", "U5_CQR_balanced")
SIZES = ("small_lt500", "medium_500to1999", "large_ge2000")
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


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"Empty output: {path}")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_features(times, areas, horizon):
    if len(times) != 6 or len(areas) != 6:
        raise ValueError("Expected six historical samples")

    gaps = np.diff(times)
    changes = np.diff(areas)
    if np.any(gaps <= 0):
        raise ValueError("Non-increasing timestamps")

    rates = changes / gaps
    mt = float(np.mean(times))
    my = float(np.mean(areas))
    centered = times - mt
    denominator = float(np.dot(centered, centered))

    slope = (
        float(np.dot(centered, areas - my)) / denominator
        if denominator > 1e-12 else 0.0
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


def get_size(area):
    if area < 500:
        return SIZES[0]
    if area < 2000:
        return SIZES[1]
    return SIZES[2]


def load_data():
    output = {
        h: {"X": [], "y": [], "g": [], "p": [], "size": []}
        for h in HORIZONS
    }

    with MANIFEST.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            h = int(row["horizon_hours"])
            if h not in HORIZONS:
                raise ValueError(f"Unexpected horizon: {h}")

            times = np.asarray(
                json.loads(row["history_relative_hours"]), dtype=float
            )
            areas = np.asarray(
                json.loads(row["history_areas_pixels"]), dtype=float
            )

            if not np.isfinite(times).all():
                raise ValueError("Invalid history times")
            if not np.isfinite(areas).all() or np.any(areas <= 0):
                raise ValueError("Invalid historical leaf areas")
            if abs(times[-1]) > 1e-5:
                raise ValueError("History does not end at zero")

            genotype = row["genotype"]
            plant = row["plant_id"]
            if not plant.startswith(genotype + "/"):
                raise ValueError("Genotype/plant mismatch")

            d = output[h]
            d["X"].append(build_features(times, areas, h))
            d["y"].append(float(row["target_area_pixels"]))
            d["g"].append(genotype)
            d["p"].append(plant)
            d["size"].append(get_size(areas[-1]))

    expected = {24: 121431, 48: 107214}
    for h, d in output.items():
        d["X"] = np.stack(d["X"])
        d["y"] = np.asarray(d["y"], dtype=float)
        d["g"] = np.asarray(d["g"])
        d["p"] = np.asarray(d["p"])
        d["size"] = np.asarray(d["size"])

        if len(d["y"]) != expected[h]:
            raise ValueError(f"Window count mismatch: {h}")
        if not np.isfinite(d["X"]).all():
            raise ValueError("Nonfinite features")
        if not np.isfinite(d["y"]).all():
            raise ValueError("Nonfinite target")

    return output


def balanced_weights(genotypes, plants):
    gp = defaultdict(set)
    counts = Counter(plants)

    for g, p in zip(genotypes, plants):
        gp[str(g)].add(str(p))

    w = np.asarray([
        1.0 / (len(gp) * len(gp[str(g)]) * counts[str(p)])
        for g, p in zip(genotypes, plants)
    ], dtype=float)

    return w / w.sum()


def pooled_quantile(scores):
    n = len(scores)
    if n == 0:
        raise ValueError("Empty calibration scores")

    k = math.ceil((n + 1) * NOMINAL)
    if k > n:
        return float("inf")

    return float(np.partition(scores, k - 1)[k - 1])


def weighted_quantile(scores, weights):
    order = np.argsort(scores)
    cumulative = np.cumsum(weights[order])
    idx = int(np.searchsorted(cumulative, NOMINAL, side="left"))
    return float(scores[order[min(idx, len(order) - 1)]])


def plant_metrics(y, lower, upper, plants):
    covered = (y >= lower) & (y <= upper)
    width = upper - lower

    score = (
        width
        + 20.0 * np.maximum(lower - y, 0.0)
        + 20.0 * np.maximum(y - upper, 0.0)
    )

    values = []
    for plant in np.unique(plants):
        mask = plants == plant
        values.append([
            np.mean(covered[mask]),
            np.mean(width[mask]),
            np.mean(score[mask]),
        ])

    values = np.asarray(values, dtype=float)

    return {
        "plants": len(values),
        "windows": len(y),
        "coverage": float(values[:, 0].mean()),
        "width": float(values[:, 1].mean()),
        "score": float(values[:, 2].mean()),
    }


def fit_quantile_model(X, y, weights, quantile):
    model = HistGradientBoostingRegressor(
        loss="quantile",
        quantile=quantile,
        **PARAMS,
    )
    model.fit(X, y, sample_weight=weights * len(weights))
    return model


def evaluate(d, outer, inner, horizon):
    fitting = set(inner["train_genotypes"])
    calibration = set(inner["validation_genotypes"])
    test_genotype = outer["test_genotype"]

    if len(fitting) != 15 or len(calibration) != 5:
        raise ValueError("Incorrect split sizes")

    if (
        fitting & calibration
        or test_genotype in fitting
        or test_genotype in calibration
    ):
        raise ValueError("Genotype leakage")

    fit = np.isin(d["g"], list(fitting))
    cal = np.isin(d["g"], list(calibration))
    test = d["g"] == test_genotype

    if (
        set(d["g"][fit]) != fitting
        or set(d["g"][cal]) != calibration
        or set(d["g"][test]) != {test_genotype}
    ):
        raise ValueError("Missing genotype data")

    w_fit = balanced_weights(d["g"][fit], d["p"][fit])

    model_lo = fit_quantile_model(
        d["X"][fit], d["y"][fit], w_fit, 0.05
    )
    model_hi = fit_quantile_model(
        d["X"][fit], d["y"][fit], w_fit, 0.95
    )

    # Sort quantile estimates if they cross.
    cal_lo_raw = model_lo.predict(d["X"][cal])
    cal_hi_raw = model_hi.predict(d["X"][cal])

    cal_lo = np.minimum(cal_lo_raw, cal_hi_raw)
    cal_hi = np.maximum(cal_lo_raw, cal_hi_raw)

    scores = np.maximum(
        cal_lo - d["y"][cal],
        d["y"][cal] - cal_hi,
    )

    w_cal = balanced_weights(d["g"][cal], d["p"][cal])

    thresholds = {
        "U4_CQR_pooled": max(0.0, pooled_quantile(scores)),
        "U5_CQR_balanced": max(
            0.0, weighted_quantile(scores, w_cal)
        ),
    }

    test_lo_raw = model_lo.predict(d["X"][test])
    test_hi_raw = model_hi.predict(d["X"][test])

    test_lo = np.minimum(test_lo_raw, test_hi_raw)
    test_hi = np.maximum(test_lo_raw, test_hi_raw)

    y = d["y"][test]
    plants = d["p"][test]
    sizes = d["size"][test]

    common = {
        "outer_fold": int(outer["outer_fold"]),
        "inner_fold": int(inner["inner_fold"]),
        "horizon_hours": horizon,
        "test_genotype": test_genotype,
    }

    fold_rows = []
    size_rows = []

    for method, q in thresholds.items():
        lower = np.maximum(0.0, test_lo - q)
        upper = np.maximum(lower, test_hi + q)

        fold_rows.append({
            **common,
            "method": method,
            "threshold": q,
            "calibration_plants": len(np.unique(d["p"][cal])),
            "calibration_windows": int(cal.sum()),
            **plant_metrics(y, lower, upper, plants),
        })

        for category in SIZES:
            mask = sizes == category
            if not mask.any():
                continue

            size_rows.append({
                **common,
                "method": method,
                "leaf_size": category,
                **plant_metrics(
                    y[mask], lower[mask], upper[mask], plants[mask]
                ),
            })

    return fold_rows, size_rows


def aggregate(rows, field, group_fields):
    groups = defaultdict(list)
    for row in rows:
        key = tuple(row[k] for k in group_fields)
        groups[key].append(row)
    return {
        key: float(np.mean([r[field] for r in group]))
        for key, group in groups.items()
    }


def summarize(fold_rows, size_rows):
    overall = []
    conditional = []

    for h in HORIZONS:
        for method in METHODS:
            selected = [
                r for r in fold_rows
                if r["horizon_hours"] == h and r["method"] == method
            ]

            by_genotype = defaultdict(list)
            for r in selected:
                by_genotype[r["test_genotype"]].append(r)

            if len(by_genotype) != 21:
                raise ValueError("Expected 21 held-out genotypes")

            averaged = []
            for genotype, items in by_genotype.items():
                if sorted(x["inner_fold"] for x in items) != [1, 2, 3, 4]:
                    raise ValueError(f"Missing inner fold for {genotype}")

                averaged.append({
                    metric: np.mean([r[metric] for r in items])
                    for metric in ("coverage", "width", "score")
                })

            overall.append({
                "horizon_hours": h,
                "method": method,
                "macro_coverage": float(np.mean([
                    x["coverage"] for x in averaged
                ])),
                "macro_width": float(np.mean([
                    x["width"] for x in averaged
                ])),
                "macro_interval_score": float(np.mean([
                    x["score"] for x in averaged
                ])),
                "genotypes_below_90": sum(
                    x["coverage"] < NOMINAL for x in averaged
                ),
            })

            for size in SIZES:
                subset = [
                    r for r in size_rows
                    if r["horizon_hours"] == h
                    and r["method"] == method
                    and r["leaf_size"] == size
                    and r["plants"] >= 2
                    and r["windows"] >= 30
                ]

                by_g = defaultdict(list)
                for r in subset:
                    by_g[r["test_genotype"]].append(r)

                complete = [
                    entries for entries in by_g.values()
                    if sorted(r["inner_fold"] for r in entries)
                    == [1, 2, 3, 4]
                ]

                if not complete:
                    continue

                conditional.append({
                    "horizon_hours": h,
                    "method": method,
                    "leaf_size": size,
                    "complete_genotypes": len(complete),
                    "macro_coverage": float(np.mean([
                        np.mean([r["coverage"] for r in entries])
                        for entries in complete
                    ])),
                    "macro_width": float(np.mean([
                        np.mean([r["width"] for r in entries])
                        for entries in complete
                    ])),
                    "macro_interval_score": float(np.mean([
                        np.mean([r["score"] for r in entries])
                        for entries in complete
                    ])),
                })

    return overall, conditional


def figures(overall, conditional):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; skipping figures")
        return

    for h in HORIZONS:
        data = [
            r for r in conditional if r["horizon_hours"] == h
        ]
        lookup = {
            (r["method"], r["leaf_size"]): r
            for r in data
        }

        x = np.arange(len(SIZES))
        width = 0.36

        fig, ax = plt.subplots(figsize=(9, 5))
        for j, method in enumerate(METHODS):
            values = [
                100 * lookup[(method, size)]["macro_coverage"]
                if (method, size) in lookup else np.nan
                for size in SIZES
            ]
            ax.bar(
                x + (j - 0.5) * width, values, width,
                label=method,
            )

        ax.axhline(90, linestyle="--", label="Nominal 90%")
        ax.set_xticks(x)
        ax.set_xticklabels(SIZES, rotation=12)
        ax.set_ylim(0, 105)
        ax.set_ylabel("Plant/genotype-averaged coverage (%)")
        ax.set_title(f"E3.2 — {h}h CQR leaf-size reliability")
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"E3_2_cqr_size_coverage_{h}h.png",
            dpi=200,
        )
        plt.close(fig)


def compare_reference(fold_rows):
    if not REFERENCE.exists():
        print("E3.1 reference unavailable; comparison skipped.")
        return

    with REFERENCE.open(newline="", encoding="utf-8") as f:
        reference = list(csv.DictReader(f))

    ref = {
        (
            int(r["horizon_hours"]),
            int(r["outer_fold"]),
            int(r["inner_fold"]),
            r["method"],
        ): r
        for r in reference
    }

    for h in HORIZONS:
        print(f"\n{h}h — E3.1 reference:")
        for method in ("U1_balanced", "U3_normalized_balanced"):
            subset = [
                r for key, r in ref.items()
                if key[0] == h and key[3] == method
            ]
            if len(subset) != 84:
                raise ValueError("Incomplete E3.1 reference")

            by_g = defaultdict(list)
            for r in subset:
                by_g[r["test_genotype"]].append(r)

            coverage = np.mean([
                np.mean([float(x["coverage"]) for x in values])
                for values in by_g.values()
            ])
            score = np.mean([
                np.mean([float(x["score"]) for x in values])
                for values in by_g.values()
            ])

            print(
                f"  {method}: coverage={coverage:.3f}, "
                f"score={score:.2f}"
            )


def main():
    print("=" * 70)
    print("E3.2 — CONFORMALIZED QUANTILE REGRESSION")
    print("=" * 70)

    with FOLDS.open(encoding="utf-8") as f:
        folds = json.load(f)["folds"]

    if len(folds) != 21:
        raise ValueError("Expected 21 outer folds")

    data = load_data()
    fold_rows = []
    size_rows = []

    for h in HORIZONS:
        print(f"\n{h}h — {len(data[h]['y']):,} windows", flush=True)

        for outer in folds:
            for inner in outer["inner_folds"]:
                metrics, subgroup_metrics = evaluate(
                    data[h], outer, inner, h
                )
                fold_rows.extend(metrics)
                size_rows.extend(subgroup_metrics)

                by_method = {
                    r["method"]: r for r in metrics
                }
                print(
                    f"  Outer {outer['outer_fold']:02d}/21 "
                    f"| inner {inner['inner_fold']}/4 "
                    f"| {outer['test_genotype']:11s} "
                    f"| U4={by_method['U4_CQR_pooled']['coverage']:.3f} "
                    f"| U5={by_method['U5_CQR_balanced']['coverage']:.3f}",
                    flush=True,
                )

        # Checkpoint after completing each horizon.
        save_csv(FOLD_OUT, fold_rows)
        save_csv(SIZE_OUT, size_rows)

    overall, conditional = summarize(fold_rows, size_rows)

    save_csv(SUMMARY_OUT, overall)
    save_csv(SIZE_SUMMARY_OUT, conditional)
    figures(overall, conditional)

    lines = [
        "# E3.2 — Conformalized Quantile Regression",
        "",
        "## Methods",
        "",
        "- U4: 5%/95% quantile regression plus pooled CQR.",
        "- U5: 5%/95% quantile regression plus balanced CQR.",
        "- Two quantile models fitted for each split.",
        "- Fixed B3-style model complexity.",
        "- Nonnegative conformal expansion threshold.",
        "- No test-genotype fitting or calibration.",
        "- No outer-test-based hyperparameter selection.",
        "",
        "## Results",
        "",
        "| Horizon | Method | Coverage | Width | Score |",
        "|---|---|---:|---:|---:|",
    ]

    for r in overall:
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
        "- CQR is an existing method, not a new contribution.",
        "- Weighted empirical quantiles do not automatically "
        "provide genotype-shift coverage guarantees.",
        "- Quantile crossing is repaired by sorting endpoints.",
        "- Four assignments share the same test plants.",
        "- Outer-genotype results have informed the exploratory "
        "research direction.",
        "- Correlated windows and annotation uncertainty remain.",
        "",
        "## Next",
        "",
        "Compare CQR against U1/U3 on coverage, interval score, "
        "and leaf-size subgroup reliability. Evaluate Mondrian "
        "calibration before designing a novel method.",
    ])

    REPORT_OUT.write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )

    print("\n" + "=" * 70)
    print("E3.2 — FINAL RESULTS")
    print("=" * 70)

    for row in overall:
        print(
            f"\n{row['horizon_hours']}h | {row['method']}"
            f"\n  Coverage: {row['macro_coverage']:.3f}"
            f"\n  Mean width: {row['macro_width']:.2f} px"
            f"\n  Interval score: {row['macro_interval_score']:.2f}"
            f"\n  Genotypes below 90%: {row['genotypes_below_90']}/21"
        )

    for row in conditional:
        print(
            f"\n{row['horizon_hours']}h | {row['method']} "
            f"| {row['leaf_size']}"
            f"\n  Coverage: {row['macro_coverage']:.3f}"
            f"\n  Interval score: {row['macro_interval_score']:.2f}"
            f"\n  Complete genotypes: "
            f"{row['complete_genotypes']}/21"
        )

    compare_reference(fold_rows)

    print("\nSaved:")
    for path in (
        FOLD_OUT, SIZE_OUT, SUMMARY_OUT,
        SIZE_SUMMARY_OUT, REPORT_OUT
    ):
        print(path)

    print("\nE3.2 COMPLETE")


if __name__ == "__main__":
    main()