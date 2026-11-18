
#!/usr/bin/env python3
"""
P3.4 — Robustness of historical-feature associations.

Inputs:
    experiments/P3/P3_3/P3_3_genotype_contrasts.csv

Outputs:
    experiments/P3/P3_4/

All analyses are retrospective and descriptive.
Bootstrap intervals represent variation when resampling
the observed genotype-level contrasts. They are not
prospective validation or distribution-shift guarantees.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT / "experiments/P3/P3_3/"
    "P3_3_genotype_contrasts.csv"
)

OUT = ROOT / "experiments/P3/P3_4"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261116
N_BOOT = 20000
rng = np.random.default_rng(SEED)

METRICS = [
    "upper_miss_high_minus_low",
    "lower_miss_high_minus_low",
    "covered_high_minus_low",
]

df = pd.read_csv(INPUT)

REQUIRED = {
    "feature",
    "horizon_hours",
    "test_genotype",
    "paired_plants",
    *METRICS,
}

missing = REQUIRED - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {missing}")

if df.duplicated(
    ["feature", "horizon_hours", "test_genotype"]
).any():
    raise ValueError("Duplicate genotype contrast records")

if df[list(METRICS)].isna().any().any():
    raise ValueError("Missing paired contrast values")

if not np.allclose(
    df["covered_high_minus_low"]
    + df["upper_miss_high_minus_low"]
    + df["lower_miss_high_minus_low"],
    0,
    atol=1e-9,
):
    raise ValueError("Directional decomposition failed")

# These groups are defined by plant replication,
# NOT by the number of eligible paired plants.
original = pd.read_csv(
    ROOT / "experiments/P3/P3_1/"
    "P3_1_genotype_summary.csv"
)

original = original.rename(
    columns={
        "horizon": "horizon_hours",
        "genotype": "test_genotype",
        "plants": "original_plants",
    }
)

df = df.merge(
    original[
        ["horizon_hours", "test_genotype", "original_plants"]
    ],
    on=["horizon_hours", "test_genotype"],
    how="left",
    validate="many_to_one",
)

if df["original_plants"].isna().any():
    raise ValueError("Unable to recover plant replication counts")

df["replication_group"] = np.where(
    df["original_plants"] < 8,
    "<8 plants",
    ">=8 plants",
)


def bootstrap_mean(values):
    """Genotype-level nonparametric bootstrap."""
    values = np.asarray(values, dtype=float)

    n = len(values)
    if n < 2:
        return np.nan, np.nan

    boot = np.empty(N_BOOT)

    for j in range(N_BOOT):
        indices = rng.integers(0, n, size=n)
        boot[j] = values[indices].mean()

    return (
        float(np.quantile(boot, 0.025)),
        float(np.quantile(boot, 0.975)),
    )


def summarize(values):
    values = np.asarray(values, dtype=float)

    n = len(values)
    estimate = float(values.mean())
    low, high = bootstrap_mean(values)

    # Exact descriptive leave-one-out calculations.
    if n >= 2:
        loo = (
            values.sum() - values
        ) / (n - 1)

        loo_min = float(loo.min())
        loo_max = float(loo.max())
    else:
        loo_min = np.nan
        loo_max = np.nan

    return {
        "n_genotypes": n,
        "mean_pp": estimate * 100,
        "median_pp": float(np.median(values)) * 100,
        "positive_genotypes": int(np.sum(values > 0)),
        "negative_genotypes": int(np.sum(values < 0)),
        "ci_lower_pp": low * 100,
        "ci_upper_pp": high * 100,
        "loo_min_pp": loo_min * 100,
        "loo_max_pp": loo_max * 100,
    }


rows = []

for (feature, horizon), frame in df.groupby(
    ["feature", "horizon_hours"],
    sort=True,
):
    for group_name, group in [
        ("all", frame),
        (
            "<8 plants",
            frame[frame["replication_group"] == "<8 plants"],
        ),
        (
            ">=8 plants",
            frame[frame["replication_group"] == ">=8 plants"],
        ),
    ]:
        if group.empty:
            continue

        for metric in METRICS:
            result = summarize(group[metric])

            rows.append({
                "feature": feature,
                "horizon": int(horizon),
                "replication_group": group_name,
                "metric": metric,
                "paired_plants_total": int(
                    group["paired_plants"].sum()
                ),
                **result,
            })

summary = pd.DataFrame(rows)

summary.to_csv(
    OUT / "P3_4_robustness_summary.csv",
    index=False,
)

# Build one focused plot for upper-tail misses,
# with equal-genotype bootstrap intervals.
plot_data = summary[
    (summary["replication_group"] == "all")
    & (
        summary["metric"]
        == "upper_miss_high_minus_low"
    )
].copy()

features = [
    "absolute_growth",
    "relative_growth",
    "cutoff_area",
    "history_gap",
]

fig, axes = plt.subplots(
    1, 2, figsize=(12, 6), sharey=True
)

for ax, horizon in zip(axes, [24, 48]):
    current = plot_data[
        plot_data["horizon"] == horizon
    ].set_index("feature")

    for i, feature in enumerate(features):
        if feature not in current.index:
            continue

        row = current.loc[feature]
        estimate = row["mean_pp"]

        left = max(
            0,
            estimate - row["ci_lower_pp"],
        )
        right = max(
            0,
            row["ci_upper_pp"] - estimate,
        )

        ax.errorbar(
            estimate,
            i,
            xerr=np.array([[left], [right]]),
            fmt="o",
            capsize=4,
        )

    ax.axvline(
        0,
        color="black",
        linestyle="--",
        linewidth=1,
    )

    ax.set_title(f"{horizon}-hour horizon")
    ax.set_yticks(
        range(len(features)),
        [
            "Absolute growth",
            "Relative growth",
            "Cutoff area",
            "History gap",
        ],
    )
    ax.set_xlabel(
        "High minus low upper-tail misses\n"
        "(percentage points)"
    )
    ax.grid(axis="x", alpha=0.2)
    ax.invert_yaxis()

fig.suptitle(
    "P3.4 — Historical features and directional miscoverage"
)

fig.tight_layout()

fig.savefig(
    OUT / "P3_4_upper_tail_robustness.png",
    dpi=250,
    bbox_inches="tight",
)
fig.savefig(
    OUT / "P3_4_upper_tail_robustness.pdf",
    bbox_inches="tight",
)
plt.close(fig)

# Concise report focused on the key hypothesis.
report = [
    "# P3.4 — Growth Association Robustness",
    "",
    f"Seed: {SEED}",
    f"Bootstrap replicates: {N_BOOT}",
    "",
    "Genotypes are the bootstrap resampling units.",
    "Four assignments are not independent replicates.",
    "",
]

upper = summary[
    summary["metric"] == "upper_miss_high_minus_low"
]

for feature in features:
    for horizon in (24, 48):
        current = upper[
            (upper.feature == feature)
            & (upper.horizon == horizon)
        ]

        if current.empty:
            continue

        report += [
            f"## {feature} — {horizon}h",
            "",
        ]

        for _, row in current.iterrows():
            report += [
                f"### {row['replication_group']}",
                f"- Genotypes: {row['n_genotypes']}",
                f"- Paired plants: {row['paired_plants_total']}",
                f"- Mean contrast: {row['mean_pp']:+.3f} pp",
                f"- Bootstrap percentile interval: "
                f"[{row['ci_lower_pp']:+.3f}, "
                f"{row['ci_upper_pp']:+.3f}] pp",
                f"- Positive genotype contrasts: "
                f"{row['positive_genotypes']}/"
                f"{row['n_genotypes']}",
                f"- Leave-one-genotype-out mean range: "
                f"[{row['loo_min_pp']:+.3f}, "
                f"{row['loo_max_pp']:+.3f}] pp",
                "",
            ]

report += [
    "## Interpretation limits",
    "",
    "- All analyses are exploratory and post hoc.",
    "- Percentile intervals are descriptive, not calibrated",
    "  confidence intervals for novel genotype populations.",
    "- Multiple feature/horizon comparisons were examined.",
    "- Associations are not adjusted for correlated features.",
    "- Results do not establish biological causation.",
    "- No change to U5 or its forecast intervals was made.",
]

text = "\n".join(report) + "\n"

(OUT / "P3_4_results.md").write_text(
    text,
    encoding="utf-8",
)

print(text)
print("Outputs:", OUT)
