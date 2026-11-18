
from pathlib import Path
from collections import defaultdict
from statistics import mean, median
import csv
import random

# ============================================================
# E1.3 — GENOTYPE-LEVEL BASELINE ERROR ANALYSIS
#
# Inputs: E1.2 genotype-level metrics
#
# No model fitting.
# No changes to the original dataset.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

EXP = ROOT / "experiments/E1"
TABLES = EXP / "tables"
FIGURES = EXP / "figures"
REPORTS = EXP / "reports"

INPUT = TABLES / "E1_2_genotype_metrics.csv"

OUTPUT = TABLES / "E1_3_genotype_comparison.csv"
SUMMARY = TABLES / "E1_3_paired_summary.csv"
REPORT = REPORTS / "E1_3_baseline_analysis.md"

HORIZONS = (24, 48)
N_BOOTSTRAP = 10000
SEED = 20261108


def read_csv(path):
    with path.open(
        newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def save_csv(path, rows, fields):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=fields
        )
        writer.writeheader()
        writer.writerows(rows)


def bootstrap_interval(differences):
    """
    Paired bootstrap of the mean genotype-level
    MAE difference (B0 - B1).

    Resamples whole genotype-level differences,
    NOT overlapping forecast windows.
    """
    rng = random.Random(SEED)
    n = len(differences)
    estimates = []

    for _ in range(N_BOOTSTRAP):
        sample = [
            differences[rng.randrange(n)]
            for _ in range(n)
        ]
        estimates.append(mean(sample))

    estimates.sort()

    return (
        estimates[int(0.025 * N_BOOTSTRAP)],
        estimates[int(0.975 * N_BOOTSTRAP)]
    )


def main():
    FIGURES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    data = defaultdict(dict)

    for row in read_csv(INPUT):
        if row["subset"] != "ALL":
            continue

        horizon = int(row["horizon_hours"])
        genotype = row["genotype"]
        model = row["model"]

        data[(horizon, genotype)][model] = {
            "mae": float(
                row["plant_mean_mae_pixels"]
            ),
            "plants": int(row["plants"]),
            "windows": int(row["windows"]),
        }

    comparisons = []

    for (horizon, genotype), models in sorted(data.items()):
        b0 = models["B0_persistence"]
        b1 = models["B1_linear"]

        if b0["plants"] != b1["plants"]:
            raise ValueError(
                f"Plant mismatch: {genotype}"
            )

        reduction = b0["mae"] - b1["mae"]

        comparisons.append({
            "horizon_hours": horizon,
            "genotype": genotype,
            "plants": b0["plants"],
            "windows": b0["windows"],
            "B0_mae": round(b0["mae"], 5),
            "B1_mae": round(b1["mae"], 5),
            "mae_reduction_pixels": round(
                reduction, 5
            ),
            "relative_reduction_pct": round(
                100 * reduction / b0["mae"], 3
            ) if b0["mae"] else 0.0,
            "winner": (
                "B1" if reduction > 0
                else "B0" if reduction < 0
                else "TIE"
            )
        })

    save_csv(
        OUTPUT,
        comparisons,
        [
            "horizon_hours", "genotype", "plants",
            "windows", "B0_mae", "B1_mae",
            "mae_reduction_pixels",
            "relative_reduction_pct", "winner"
        ]
    )

    summaries = []

    for horizon in HORIZONS:
        rows = [
            r for r in comparisons
            if r["horizon_hours"] == horizon
        ]

        if len(rows) != 21:
            raise RuntimeError(
                f"{horizon}h has {len(rows)} genotypes; expected 21"
            )

        differences = [
            r["mae_reduction_pixels"]
            for r in rows
        ]

        lower, upper = bootstrap_interval(
            differences
        )

        summaries.append({
            "horizon_hours": horizon,
            "genotypes": len(rows),
            "B1_wins": sum(
                r["winner"] == "B1" for r in rows
            ),
            "B0_wins": sum(
                r["winner"] == "B0" for r in rows
            ),
            "ties": sum(
                r["winner"] == "TIE" for r in rows
            ),
            "mean_B0_mae": round(
                mean(r["B0_mae"] for r in rows), 4
            ),
            "mean_B1_mae": round(
                mean(r["B1_mae"] for r in rows), 4
            ),
            "mean_mae_reduction": round(
                mean(differences), 4
            ),
            "median_mae_reduction": round(
                median(differences), 4
            ),
            "bootstrap_95_lower": round(lower, 4),
            "bootstrap_95_upper": round(upper, 4),
        })

    save_csv(
        SUMMARY,
        summaries,
        list(summaries[0].keys())
    )

    try:
        import matplotlib.pyplot as plt

        for horizon in HORIZONS:
            rows = [
                r for r in comparisons
                if r["horizon_hours"] == horizon
            ]

            rows.sort(
                key=lambda r: r["mae_reduction_pixels"]
            )

            names = [r["genotype"] for r in rows]
            changes = [
                r["mae_reduction_pixels"]
                for r in rows
            ]

            fig, ax = plt.subplots(
                figsize=(10, 8)
            )

            ax.barh(names, changes)
            ax.axvline(
                0, linestyle="--", linewidth=1
            )
            ax.set_xlabel(
                "MAE improvement: B0 - B1 (pixels)"
            )
            ax.set_ylabel("Held-out genotype")
            ax.set_title(
                f"E1.3 — {horizon}h forecast: "
                "linear vs persistence"
            )

            fig.tight_layout()
            fig.savefig(
                FIGURES
                / f"E1_3_genotype_improvement_{horizon}h.png",
                dpi=200
            )
            plt.close(fig)

    except ImportError:
        print(
            "Matplotlib unavailable; figures skipped."
        )

    report = [
        "# E1.3 — Genotype-Level Baseline Comparison",
        "",
        "Positive difference means B1 has lower MAE.",
        "",
        "| Horizon | B1 wins | B0 wins | "
        "Mean reduction (px) | Bootstrap 95% interval |",
        "|---|---:|---:|---:|---|",
    ]

    for row in summaries:
        report.append(
            f"| {row['horizon_hours']}h "
            f"| {row['B1_wins']} "
            f"| {row['B0_wins']} "
            f"| {row['mean_mae_reduction']:.2f} "
            f"| [{row['bootstrap_95_lower']:.2f}, "
            f"{row['bootstrap_95_upper']:.2f}] |"
        )

    report.extend([
        "",
        "## Interpretation limits",
        "",
        "- Genotypes, not windows, are resampled.",
        "- These methods have no population-fitted parameters.",
        "- This does not compare seen versus unseen genotypes.",
        "- The bootstrap is descriptive for these genotypes.",
        "- Identity and visible-area quality concerns remain.",
    ])

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("=" * 68)
    print("E1.3 — GENOTYPE-LEVEL BASELINE ANALYSIS")
    print("=" * 68)

    for row in summaries:
        print(
            f"\n{row['horizon_hours']}h:"
        )
        print(
            f"  B1 wins: {row['B1_wins']}/21"
        )
        print(
            f"  B0 wins: {row['B0_wins']}/21"
        )
        print(
            f"  Mean MAE improvement: "
            f"{row['mean_mae_reduction']:.2f} px"
        )
        print(
            "  95% genotype-bootstrap interval: "
            f"[{row['bootstrap_95_lower']:.2f}, "
            f"{row['bootstrap_95_upper']:.2f}] px"
        )

        worst = sorted(
            (
                r for r in comparisons
                if r["horizon_hours"]
                == row["horizon_hours"]
            ),
            key=lambda r: r["mae_reduction_pixels"]
        )[:3]

        print("  Weakest B1 comparisons:")
        for item in worst:
            print(
                f"    {item['genotype']:12s} "
                f"improvement="
                f"{item['mae_reduction_pixels']:.2f} px"
            )

    print("\nSaved:")
    print(OUTPUT)
    print(SUMMARY)
    print(REPORT)
    print("\nE1.3 complete.")


if __name__ == "__main__":
    main()
