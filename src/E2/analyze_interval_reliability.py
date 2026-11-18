
from pathlib import Path
from collections import defaultdict
import csv
import math

import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# E2.2 — INTERVAL RELIABILITY ANALYSIS
#
# Inputs: saved E2.1 held-out predictions
# No model fitting or calibration.
#
# Analysis:
# - Genotype-level coverage
# - Plant-averaged interval score
# - Coverage/width tradeoff
# - Paired genotype bootstrap
# - Worst-covered genotypes
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BASE = ROOT / "experiments/E2"
TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

INPUT = TABLES / "E2_1_interval_predictions.csv"

PLANT_OUT = TABLES / "E2_2_plant_scores.csv"
GENOTYPE_OUT = TABLES / "E2_2_genotype_reliability.csv"
SUMMARY_OUT = TABLES / "E2_2_reliability_summary.csv"
REPORT_OUT = REPORTS / "E2_2_interval_reliability.md"

HORIZONS = (24, 48)
METHODS = ("U0_pooled", "U1_balanced")
TARGET_COVERAGE = 0.90
ALPHA = 1 - TARGET_COVERAGE

BOOTSTRAPS = 10000
SEED = 20261108

FIGURES.mkdir(parents=True, exist_ok=True)
REPORTS.mkdir(parents=True, exist_ok=True)


def save_csv(path, rows):
    if not rows:
        raise RuntimeError(f"No results for {path}")

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def interval_score(y, lower, upper):
    width = upper - lower

    below_penalty = (
        (2 / ALPHA) * max(lower - y, 0.0)
    )

    above_penalty = (
        (2 / ALPHA) * max(y - upper, 0.0)
    )

    return width + below_penalty + above_penalty


def bootstrap_mean_ci(values, seed):
    values = np.asarray(values, dtype=np.float64)

    rng = np.random.default_rng(seed)
    n = len(values)
    estimates = np.empty(BOOTSTRAPS)

    for i in range(BOOTSTRAPS):
        sample = rng.integers(0, n, n)
        estimates[i] = values[sample].mean()

    return np.quantile(
        estimates, [0.025, 0.975]
    )


def main():
    print("=" * 68)
    print("E2.2 — INTERVAL RELIABILITY ANALYSIS")
    print("=" * 68)

    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    # Each key identifies one test plant, method,
    # and forecasting horizon.
    sums = defaultdict(
        lambda: {
            "n": 0,
            "covered": 0,
            "width": 0.0,
            "score": 0.0,
            "absolute_error": 0.0,
            "miss_distance": 0.0,
        }
    )

    total_rows = 0

    with INPUT.open(
        newline="", encoding="utf-8"
    ) as f:
        for row in csv.DictReader(f):
            h = int(row["horizon_hours"])
            method = row["method"]
            genotype = row["genotype"]
            plant = row["plant_id"]

            if h not in HORIZONS:
                raise ValueError("Unexpected horizon")

            if method not in METHODS:
                raise ValueError("Unexpected method")

            if not plant.startswith(genotype + "/"):
                raise ValueError("Plant/genotype mismatch")

            y = float(row["target_area_pixels"])
            pred = float(row["predicted_area_pixels"])
            lower = float(row["lower_pixels"])
            upper = float(row["upper_pixels"])

            if not all(map(
                math.isfinite,
                (y, pred, lower, upper)
            )):
                raise ValueError("Nonfinite prediction")

            if lower < 0 or upper < lower:
                raise ValueError("Invalid interval")

            covered = int(lower <= y <= upper)
            if covered != int(row["covered"]):
                raise ValueError("Coverage flag mismatch")

            miss = max(
                lower - y,
                y - upper,
                0.0,
            )

            key = (h, genotype, plant, method)
            acc = sums[key]

            acc["n"] += 1
            acc["covered"] += covered
            acc["width"] += upper - lower
            acc["score"] += interval_score(
                y, lower, upper
            )
            acc["absolute_error"] += abs(pred - y)
            acc["miss_distance"] += miss

            total_rows += 1

    plant_rows = []

    for key, acc in sorted(sums.items()):
        h, g, p, method = key
        n = acc["n"]

        plant_rows.append({
            "horizon_hours": h,
            "genotype": g,
            "plant_id": p,
            "method": method,
            "windows": n,
            "coverage": acc["covered"] / n,
            "mean_width_pixels": acc["width"] / n,
            "mean_interval_score": acc["score"] / n,
            "mae_pixels": acc["absolute_error"] / n,
            "mean_miss_distance": acc["miss_distance"] / n,
        })

    save_csv(PLANT_OUT, plant_rows)

    # Equal weight for every plant within genotype.
    grouped = defaultdict(list)
    for row in plant_rows:
        grouped[
            (
                row["horizon_hours"],
                row["genotype"],
                row["method"],
            )
        ].append(row)

    genotype_rows = []

    for (h, g, method), rows in sorted(grouped.items()):
        def avg(field):
            return float(np.mean([
                r[field] for r in rows
            ]))

        genotype_rows.append({
            "horizon_hours": h,
            "genotype": g,
            "method": method,
            "plants": len(rows),
            "windows": sum(
                r["windows"] for r in rows
            ),
            "coverage": avg("coverage"),
            "coverage_gap_pp": (
                100 * (avg("coverage") - TARGET_COVERAGE)
            ),
            "mean_width_pixels": avg(
                "mean_width_pixels"
            ),
            "mean_interval_score": avg(
                "mean_interval_score"
            ),
            "mae_pixels": avg("mae_pixels"),
            "mean_miss_distance": avg(
                "mean_miss_distance"
            ),
        })

    save_csv(GENOTYPE_OUT, genotype_rows)

    summaries = []

    for h in HORIZONS:
        by_method = {}

        for method in METHODS:
            rows = [
                r for r in genotype_rows
                if (
                    r["horizon_hours"] == h
                    and r["method"] == method
                )
            ]

            by_method[method] = {
                r["genotype"]: r
                for r in rows
            }

        genotypes = sorted(by_method[METHODS[0]])

        if len(genotypes) != 21:
            raise RuntimeError(
                f"Expected 21 genotypes for {h}h"
            )

        if set(genotypes) != set(by_method[METHODS[1]]):
            raise RuntimeError("Genotype pairing mismatch")

        u0 = by_method["U0_pooled"]
        u1 = by_method["U1_balanced"]

        for g in genotypes:
            if (
                u0[g]["plants"] != u1[g]["plants"]
                or u0[g]["windows"] != u1[g]["windows"]
            ):
                raise RuntimeError(
                    f"Unpaired test data for {g}"
                )

        def metric(method_data, field):
            return np.asarray([
                method_data[g][field]
                for g in genotypes
            ], dtype=np.float64)

        cover0 = metric(u0, "coverage")
        cover1 = metric(u1, "coverage")
        width0 = metric(u0, "mean_width_pixels")
        width1 = metric(u1, "mean_width_pixels")
        score0 = metric(u0, "mean_interval_score")
        score1 = metric(u1, "mean_interval_score")

        coverage_gain = cover1 - cover0
        score_improvement = score0 - score1

        cover_ci = bootstrap_mean_ci(
            coverage_gain, SEED + h
        )
        score_ci = bootstrap_mean_ci(
            score_improvement, SEED + 100 + h
        )

        summaries.append({
            "horizon_hours": h,
            "genotypes": len(genotypes),
            "U0_coverage": float(cover0.mean()),
            "U1_coverage": float(cover1.mean()),
            "coverage_gain_pp": float(
                100 * coverage_gain.mean()
            ),
            "coverage_gain_CI_low_pp": float(
                100 * cover_ci[0]
            ),
            "coverage_gain_CI_high_pp": float(
                100 * cover_ci[1]
            ),
            "U0_width": float(width0.mean()),
            "U1_width": float(width1.mean()),
            "width_increase_pct": float(
                100 * (
                    width1.mean() / width0.mean() - 1
                )
            ),
            "U0_interval_score": float(score0.mean()),
            "U1_interval_score": float(score1.mean()),
            "score_improvement_U0_minus_U1": float(
                score_improvement.mean()
            ),
            "score_CI_low": float(score_ci[0]),
            "score_CI_high": float(score_ci[1]),
            "U1_score_wins": int(
                np.sum(score1 < score0)
            ),
            "U0_score_wins": int(
                np.sum(score0 < score1)
            ),
            "U1_below_90": int(
                np.sum(cover1 < TARGET_COVERAGE)
            ),
            "U1_below_80": int(
                np.sum(cover1 < 0.8)
            ),
        })

        # Representative figure: genotype-specific
        # interval scores; lower is better.
        x = np.arange(len(genotypes))
        fig, ax = plt.subplots(figsize=(13, 6))

        ax.plot(
            x, score0, marker="o",
            label="U0 pooled"
        )
        ax.plot(
            x, score1, marker="s",
            label="U1 balanced"
        )

        ax.set_xticks(x)
        ax.set_xticklabels(
            genotypes, rotation=65, ha="right"
        )
        ax.set_ylabel(
            "Plant-averaged 90% interval score (px)"
        )
        ax.set_title(
            f"E2.2 — {h}h interval score by genotype"
        )
        ax.legend()
        ax.grid(axis="y", alpha=0.25)
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"E2_2_interval_score_{h}h.png",
            dpi=200
        )
        plt.close(fig)

        # Coverage versus width across genotypes.
        fig, ax = plt.subplots(figsize=(9, 6))

        ax.scatter(
            width0, cover0,
            label="U0 pooled"
        )
        ax.scatter(
            width1, cover1,
            label="U1 balanced"
        )

        ax.axhline(
            TARGET_COVERAGE,
            linestyle="--",
            label="Nominal 90%"
        )
        ax.set_xlabel(
            "Mean prediction interval width (pixels)"
        )
        ax.set_ylabel("Plant-averaged coverage")
        ax.set_title(
            f"E2.2 — {h}h coverage–width tradeoff"
        )
        ax.set_ylim(0, 1.05)
        ax.legend()
        fig.tight_layout()
        fig.savefig(
            FIGURES / f"E2_2_coverage_width_{h}h.png",
            dpi=200
        )
        plt.close(fig)

    save_csv(SUMMARY_OUT, summaries)

    lines = [
        "# E2.2 — Interval Reliability Analysis",
        "",
        "## Evaluation",
        "",
        "- Input: saved E2.1 held-out interval predictions.",
        "- Primary aggregation: window -> plant -> genotype.",
        "- Genotype-level paired bootstrap with 10,000 resamples.",
        "- Nominal coverage: 90%.",
        "- Interval score: width + 20 times total miss distance.",
        "- Lower interval score is better.",
        "",
        "## Results",
        "",
        "| Horizon | Coverage gain (pp) | Width increase | "
        "U0 score | U1 score | U1 score wins |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for r in summaries:
        lines.append(
            f"| {r['horizon_hours']}h "
            f"| {r['coverage_gain_pp']:+.2f} "
            f"| {r['width_increase_pct']:+.2f}% "
            f"| {r['U0_interval_score']:.2f} "
            f"| {r['U1_interval_score']:.2f} "
            f"| {r['U1_score_wins']}/21 |"
        )

    lines.extend([
        "",
        "## Interpretation restrictions",
        "",
        "- Results use one predefined calibration split.",
        "- Bootstrap intervals describe variation across "
        "the evaluated genotype effects.",
        "- The analysis does not establish coverage guarantees.",
        "- U1's higher coverage must be considered with "
        "its wider intervals and interval score.",
        "- Overlapping windows are not independent units.",
        "- Model-selection and annotation uncertainty "
        "are not included in the bootstrap.",
        "",
        "## Next",
        "",
        "E2.3: Study coverage across leaf size, growth "
        "variability, observation gaps, and QC flags.",
        "E2.4: Evaluate calibration-split sensitivity "
        "without selecting the best split from test results.",
    ])

    REPORT_OUT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print(f"Analyzed {total_rows:,} prediction-method rows")

    for r in summaries:
        print(f"\n{r['horizon_hours']}h:")
        print(
            "  U1 coverage gain: "
            f"{r['coverage_gain_pp']:+.2f} percentage points"
        )
        print(
            "  95% genotype-bootstrap interval: "
            f"[{r['coverage_gain_CI_low_pp']:.2f}, "
            f"{r['coverage_gain_CI_high_pp']:.2f}] pp"
        )
        print(
            "  U1 interval width increase: "
            f"{r['width_increase_pct']:+.2f}%"
        )
        print(
            "  U0 interval score: "
            f"{r['U0_interval_score']:.2f}"
        )
        print(
            "  U1 interval score: "
            f"{r['U1_interval_score']:.2f}"
        )
        print(
            "  U1 score wins: "
            f"{r['U1_score_wins']}/21"
        )
        print(
            "  U1 genotypes below 90%: "
            f"{r['U1_below_90']}/21"
        )

    print("\nSaved:")
    print(PLANT_OUT)
    print(GENOTYPE_OUT)
    print(SUMMARY_OUT)
    print(REPORT_OUT)
    print("\nE2.2 COMPLETE")


if __name__ == "__main__":
    main()
