
#!/usr/bin/env python3
"""
P3.8 — Does historical growth add value beyond leaf size?

Direct paired comparison:
    SGA_CQR minus SA_CQR

Uses archived P3.6 genotype-level metrics.
No model fitting, recalibration, or test-based tuning.

Primary metric: interval score (lower is better).
Secondary: coverage, width, directional miscoverage.

Resampling unit: genotype.
Four inner assignments have already been averaged.
This is retrospective exploratory inference.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "experiments/P3/P3_6/P3_6_genotype_metrics.csv"
OUT = ROOT / "experiments/P3/P3_8"
OUT.mkdir(parents=True, exist_ok=True)

B = 20000
SEED = 20261116
rng = np.random.default_rng(SEED)

METRICS = [
    "score",
    "width",
    "coverage",
    "upper_miss",
    "lower_miss",
]

df = pd.read_csv(INPUT)

required = {
    "horizon_hours",
    "test_genotype",
    "method",
    *METRICS,
}

if missing := required - set(df.columns):
    raise ValueError(f"Missing columns: {sorted(missing)}")

if df.duplicated(
    ["horizon_hours", "test_genotype", "method"]
).any():
    raise ValueError("Duplicate genotype/method records")

if df[list(METRICS)].isna().any().any():
    raise ValueError("Missing metric values")

if not set(df.method.unique()) == {
    "U5", "SA_CQR", "SGA_CQR"
}:
    raise ValueError("Unexpected or missing methods")

summary_rows = []
paired_rows = []
plot_data = {}

for h in (24, 48):
    part = df[df.horizon_hours == h]

    wide = part.pivot(
        index="test_genotype",
        columns="method",
        values=METRICS,
    ).sort_index()

    if len(wide) != 21 or wide.isna().any().any():
        raise ValueError(f"Incomplete comparison at {h}h")

    # Verify the original equal-genotype U5 coverage.
    u5_coverage = wide[("coverage", "U5")].mean()
    expected = 0.899712 if h == 24 else 0.9004006

    if abs(u5_coverage - expected) > 0.00015:
        raise ValueError(f"U5 reproduction failed at {h}h")

    differences = {}

    for metric in METRICS:
        # Negative score/width = SGA better.
        # Negative upper_miss = fewer upper-tail failures.
        delta = (
            wide[(metric, "SGA_CQR")]
            - wide[(metric, "SA_CQR")]
        ).to_numpy(dtype=float)

        differences[metric] = delta

        # Paired genotype bootstrap:
        # resample entire genotype-level differences.
        n = len(delta)
        indices = rng.integers(0, n, size=(B, n))
        bootstrap = delta[indices].mean(axis=1)

        ci_low, ci_high = np.quantile(
            bootstrap, [0.025, 0.975]
        )

        loo = (
            delta.sum() - delta
        ) / (n - 1)

        multiplier = (
            100.0 if metric in (
                "coverage", "upper_miss", "lower_miss"
            ) else 1.0
        )

        summary_rows.append({
            "horizon": h,
            "comparison": "SGA_CQR_minus_SA_CQR",
            "metric": metric,
            "n_genotypes": n,
            "mean_delta": delta.mean() * multiplier,
            "median_delta": np.median(delta) * multiplier,
            "bootstrap_low": ci_low * multiplier,
            "bootstrap_high": ci_high * multiplier,
            "negative_genotypes": int((delta < 0).sum()),
            "positive_genotypes": int((delta > 0).sum()),
            "ties": int((delta == 0).sum()),
            "loo_min": loo.min() * multiplier,
            "loo_max": loo.max() * multiplier,
            "unit": (
                "percentage points"
                if multiplier == 100 else "px²"
            ),
        })

    for i, genotype in enumerate(wide.index):
        paired_rows.append({
            "horizon": h,
            "genotype": genotype,
            **{
                f"delta_{metric}": differences[metric][i]
                for metric in METRICS
            },
        })

    plot_data[h] = pd.DataFrame({
        "genotype": wide.index,
        "delta_score": differences["score"],
    }).sort_values("delta_score")

summary = pd.DataFrame(summary_rows)
paired = pd.DataFrame(paired_rows)

summary.to_csv(
    OUT / "P3_8_direct_comparison_summary.csv",
    index=False,
)
paired.to_csv(
    OUT / "P3_8_genotype_differences.csv",
    index=False,
)

# ---------------------------------------------------------
# Publication figures: genotype-paired score differences
# ---------------------------------------------------------

for h, data in plot_data.items():
    fig, ax = plt.subplots(figsize=(9, 7.5))

    y = np.arange(len(data))

    ax.barh(
        y,
        data["delta_score"],
    )

    ax.set_yticks(y, data["genotype"])
    ax.invert_yaxis()

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_xlabel(
        "SGA-CQR minus SA-CQR interval score (px²)\n"
        "Negative values favor adding historical growth"
    )

    ax.set_title(
        f"P3.8 — Incremental value of growth ({h}h)"
    )

    ax.grid(axis="x", alpha=0.15)
    fig.tight_layout()

    for suffix in ("png", "pdf"):
        fig.savefig(
            OUT / f"P3_8_growth_value_{h}h.{suffix}",
            dpi=250,
            bbox_inches="tight",
        )

    plt.close(fig)

# ---------------------------------------------------------
# Research report
# ---------------------------------------------------------

report = [
    "# P3.8 — Incremental Value of Historical Growth",
    "",
    "Comparison: SGA-CQR minus SA-CQR.",
    "",
    f"Bootstrap resamples: {B}",
    f"Random seed: {SEED}",
    "Resampling unit: held-out genotype.",
    "",
]

for h in (24, 48):
    report += [
        f"## {h}-hour horizon",
        "",
    ]

    rows = summary[summary.horizon == h]

    for _, row in rows.iterrows():
        report.extend([
            f"### {row.metric}",
            f"- Mean difference: "
            f"{row.mean_delta:+.4f} {row.unit}",
            "- Bootstrap percentile interval: "
            f"[{row.bootstrap_low:+.4f}, "
            f"{row.bootstrap_high:+.4f}] {row.unit}",
            f"- Negative genotype differences: "
            f"{row.negative_genotypes}/21",
            f"- Positive genotype differences: "
            f"{row.positive_genotypes}/21",
            "- Leave-one-genotype-out mean range: "
            f"[{row.loo_min:+.4f}, "
            f"{row.loo_max:+.4f}] {row.unit}",
            "",
        ])

    score_row = rows[rows.metric == "score"].iloc[0]

    if score_row.bootstrap_high < 0:
        interpretation = (
            "The observed paired genotype-bootstrap interval "
            "for interval score lies entirely below zero."
        )
    elif score_row.bootstrap_low > 0:
        interpretation = (
            "The observed paired genotype-bootstrap interval "
            "for interval score lies entirely above zero."
        )
    else:
        interpretation = (
            "The observed paired genotype-bootstrap interval "
            "for interval score includes zero."
        )

    report += [f"**Descriptive result:** {interpretation}", ""]

report += [
    "## Limitations",
    "",
    "- These are retrospective, exploratory comparisons.",
    "- The 21 test genotypes were previously inspected.",
    "- Bootstrap intervals do not provide independent validation.",
    "- Multiple outcomes and methods were examined.",
    "- No finite-sample genotype-shift guarantee is claimed.",
    "- Differences cannot establish a biological mechanism.",
]

report_text = "\n".join(report) + "\n"

(OUT / "P3_8_results.md").write_text(
    report_text,
    encoding="utf-8",
)

print(report_text)
print("Outputs saved to:", OUT)
