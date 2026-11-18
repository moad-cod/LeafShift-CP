
from pathlib import Path
from collections import defaultdict
import csv
import json

import numpy as np

# ============================================================
# E1.7 — CONSOLIDATED BASELINES AND PAIRED STATISTICS
#
# Inputs: E1.5 and E1.6 genotype metrics
# Unit of resampling: genotype
#
# No models trained. No source results overwritten.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/E1"

TABLES = BASE / "tables"
FIGURES = BASE / "figures"
REPORTS = BASE / "reports"

E15 = TABLES / "E1_5_genotype_metrics.csv"
E16 = TABLES / "E1_6_genotype_metrics.csv"

OUTPUT = TABLES / "E1_7_paired_comparisons.csv"
PER_GENOTYPE = TABLES / "E1_7_genotype_differences.csv"
REPORT = REPORTS / "E1_7_consolidated_analysis.md"

SEED = 20261108
BOOTSTRAP_SAMPLES = 10000
HORIZONS = (24, 48)


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_rows(path, rows):
    if not rows:
        raise ValueError("Cannot save empty results")
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def paired_bootstrap(differences, seed):
    """Resample complete genotype-level paired effects."""
    values = np.asarray(differences, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)

    estimates = np.empty(BOOTSTRAP_SAMPLES)

    for i in range(BOOTSTRAP_SAMPLES):
        indices = rng.integers(0, n, size=n)
        estimates[i] = values[indices].mean()

    return np.quantile(
        estimates, [0.025, 0.975]
    )


def build_comparison(name, horizon, values):
    """
    values maps genotype -> (reference MAE, challenger MAE).
    Positive difference means challenger is better.
    """
    genotypes = sorted(values)

    if len(genotypes) != 21:
        raise ValueError(
            f"{name} {horizon}h: expected 21 genotypes, "
            f"found {len(genotypes)}"
        )

    reference = np.asarray([
        values[g][0] for g in genotypes
    ], dtype=float)

    challenger = np.asarray([
        values[g][1] for g in genotypes
    ], dtype=float)

    if not (
        np.isfinite(reference).all()
        and np.isfinite(challenger).all()
    ):
        raise ValueError("Nonfinite metrics")

    differences = reference - challenger

    low, high = paired_bootstrap(
        differences,
        SEED + horizon + sum(ord(c) for c in name),
    )

    summary = {
        "comparison": name,
        "horizon_hours": horizon,
        "genotypes": len(genotypes),
        "reference_macro_mae": float(reference.mean()),
        "challenger_macro_mae": float(challenger.mean()),
        "mean_improvement_pixels": float(
            differences.mean()
        ),
        "relative_improvement_pct": float(
            100 * differences.mean() / reference.mean()
        ),
        "bootstrap_95_lower_pixels": float(low),
        "bootstrap_95_upper_pixels": float(high),
        "challenger_wins": int(
            np.sum(differences > 0)
        ),
        "reference_wins": int(
            np.sum(differences < 0)
        ),
        "ties": int(np.sum(differences == 0)),
    }

    details = []

    for i, genotype in enumerate(genotypes):
        details.append({
            "comparison": name,
            "horizon_hours": horizon,
            "genotype": genotype,
            "reference_mae": reference[i],
            "challenger_mae": challenger[i],
            "improvement_pixels": differences[i],
        })

    return summary, details


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    e15 = read_rows(E15)
    e16 = read_rows(E16)

    summaries = []
    details = []

    for horizon in HORIZONS:
        # E1.5: compare Ridge and boosting on identical
        # unseen-genotype forecasting windows.
        ridge_boost = {}

        for row in e15:
            if int(row["horizon_hours"]) != horizon:
                continue

            genotype = row["test_genotype"]

            if genotype in ridge_boost:
                raise ValueError("Duplicate E1.5 genotype")

            ridge_boost[genotype] = (
                float(row["B2_mae"]),
                float(row["B3_mae"]),
            )

        summary, rows = build_comparison(
            "E1.5_Ridge_vs_Boosting",
            horizon,
            ridge_boost,
        )

        summaries.append(summary)
        details.extend(rows)

        # E1.6: compare matched test plants.
        conditions = defaultdict(dict)

        for row in e16:
            if int(row["horizon_hours"]) != horizon:
                continue

            genotype = row["test_genotype"]
            condition = row["condition"]

            if condition in conditions[genotype]:
                raise ValueError(
                    "Duplicate E1.6 genotype/condition"
                )

            conditions[genotype][condition] = {
                "mae": float(row["plant_mean_mae"]),
                "plants": int(row["test_plants"]),
                "windows": int(row["test_windows"]),
            }

        for challenger, label in (
            ("B_exposed_extra", "E1.6_Unseen_vs_Extra"),
            ("C_exposed_matched", "E1.6_Unseen_vs_Matched"),
        ):
            paired = {}

            for genotype, group in conditions.items():
                a = group["A_unseen"]
                b = group[challenger]

                if (
                    a["plants"] != b["plants"]
                    or a["windows"] != b["windows"]
                ):
                    raise ValueError(
                        f"Test-set mismatch: {genotype}"
                    )

                paired[genotype] = (
                    a["mae"], b["mae"]
                )

            summary, rows = build_comparison(
                label, horizon, paired
            )
            summaries.append(summary)
            details.extend(rows)

    save_rows(OUTPUT, summaries)
    save_rows(PER_GENOTYPE, details)

    try:
        import matplotlib.pyplot as plt

        for horizon in HORIZONS:
            rows = [
                r for r in details
                if (
                    r["horizon_hours"] == horizon
                    and r["comparison"]
                    == "E1.6_Unseen_vs_Matched"
                )
            ]

            rows.sort(
                key=lambda r: r["improvement_pixels"]
            )

            fig, ax = plt.subplots(figsize=(10, 8))
            ax.barh(
                [r["genotype"] for r in rows],
                [r["improvement_pixels"] for r in rows],
            )
            ax.axvline(0, linestyle="--")
            ax.set_xlabel(
                "Unseen MAE minus matched-exposure MAE (px)"
            )
            ax.set_title(
                f"E1.7 — {horizon}h genotype exposure effect"
            )
            fig.tight_layout()
            fig.savefig(
                FIGURES
                / f"E1_7_matched_exposure_{horizon}h.png",
                dpi=200
            )
            plt.close(fig)

    except ImportError:
        print("Matplotlib unavailable; figures skipped.")

    lines = [
        "# E1.7 — Consolidated Statistical Analysis",
        "",
        "Positive improvement favors the challenger.",
        "",
        "| Comparison | Horizon | Improvement (px) | "
        "95% bootstrap interval (px) | Wins |",
        "|---|---:|---:|---|---:|",
    ]

    for row in summaries:
        lines.append(
            f"| {row['comparison']} "
            f"| {row['horizon_hours']}h "
            f"| {row['mean_improvement_pixels']:.2f} "
            f"| [{row['bootstrap_95_lower_pixels']:.2f}, "
            f"{row['bootstrap_95_upper_pixels']:.2f}] "
            f"| {row['challenger_wins']}/21 |"
        )

    lines.extend([
        "",
        "## Interpretation",
        "",
        "- E1.5 assesses predictive-model differences.",
        "- E1.6 assesses target-genotype training exposure.",
        "- E1.5 and E1.6 use different test-set definitions.",
        "- Bootstrap resamples genotype-level paired effects.",
        "- These are descriptive confidence intervals.",
        "- A single deterministic E1.6 plant split was used.",
        "- Overlapping windows are not treated as independent.",
        "- Exposure effects are not causal genotype effects.",
        "- Acquisition schedules and annotation QC remain "
        "potential confounders.",
        "- The analyses do not quantify uncertainty coverage.",
        "",
        "## Next",
        "",
        "Review the magnitude and uncertainty of exposure "
        "effects, then design E2 uncertainty calibration "
        "using development-only calibration data.",
    ])

    REPORT.write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8"
    )

    print("=" * 68)
    print("E1.7 — CONSOLIDATED ANALYSIS")
    print("=" * 68)

    for row in summaries:
        print(
            f"\n{row['comparison']} "
            f"| {row['horizon_hours']}h"
        )
        print(
            f"  Mean improvement: "
            f"{row['mean_improvement_pixels']:+.2f} px"
        )
        print(
            f"  Relative improvement: "
            f"{row['relative_improvement_pct']:+.2f}%"
        )
        print(
            "  95% genotype-bootstrap interval: "
            f"[{row['bootstrap_95_lower_pixels']:.2f}, "
            f"{row['bootstrap_95_upper_pixels']:.2f}] px"
        )
        print(
            f"  Challenger wins: "
            f"{row['challenger_wins']}/21"
        )

    print("\nSaved:")
    print(OUTPUT)
    print(PER_GENOTYPE)
    print(REPORT)
    print("\nE1.7 COMPLETE")


if __name__ == "__main__":
    main()
