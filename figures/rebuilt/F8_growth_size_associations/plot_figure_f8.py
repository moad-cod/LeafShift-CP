
#!/usr/bin/env python3
"""
Figure 8 — Historical growth and upper-tail miscoverage.

Inputs:
  experiments/P3/P3_3/P3_3_genotype_contrasts.csv
  experiments/P3/P3_5/P3_5_genotype_results.csv

Outputs:
  figures/rebuilt/F8_growth_size_associations/
    Figure_F8_Growth_Size_Associations.pdf
    Figure_F8_Growth_Size_Associations.png
    Figure_F8_Growth_Size_Associations.svg
    Figure_F8_source.csv
    Figure_F8_genotype_source.csv

Retrospective, descriptive association analysis.
Bootstrap resampling unit: held-out genotype.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]

INPUTS = {
    "Unadjusted": (
        ROOT / "experiments/P3/P3_3/"
        "P3_3_genotype_contrasts.csv"
    ),
    "Size-stratified": (
        ROOT / "experiments/P3/P3_5/"
        "P3_5_genotype_results.csv"
    ),
}

OUT = (
    ROOT / "figures/rebuilt/"
    "F8_growth_size_associations"
)
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261116
B = 20000
rng = np.random.default_rng(SEED)

FEATURES = {
    "absolute_growth": "Absolute historical growth",
    "relative_growth": "Relative historical growth",
}

HORIZONS = (24, 48)
CONDITIONS = ("Unadjusted", "Size-stratified")

METRIC = "upper_miss_high_minus_low"

# -------------------------------------------------------
# Publication style
# -------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

COLORS = {
    24: "#4C78A8",
    48: "#E68627",
}

# -------------------------------------------------------
# Load and validate
# -------------------------------------------------------

frames = []

for condition, path in INPUTS.items():

    if not path.exists():
        raise FileNotFoundError(path)

    d = pd.read_csv(path)

    required = {
        "feature",
        "horizon_hours",
        "test_genotype",
        METRIC,
    }

    missing = required - set(d.columns)

    if missing:
        raise ValueError(
            f"{path.name}: missing {sorted(missing)}"
        )

    d = d[
        d["feature"].isin(FEATURES)
        & d["horizon_hours"].isin(HORIZONS)
    ].copy()

    if d.duplicated([
        "feature",
        "horizon_hours",
        "test_genotype",
    ]).any():
        raise ValueError(
            f"Duplicate genotype records: {path}"
        )

    d[METRIC] = pd.to_numeric(
        d[METRIC], errors="raise"
    )

    if d[METRIC].isna().any():
        raise ValueError("Missing contrast values")

    d["condition"] = condition

    frames.append(d[
        [
            "feature",
            "horizon_hours",
            "test_genotype",
            "condition",
            METRIC,
        ]
    ])

data = pd.concat(frames, ignore_index=True)

# Ensure that every comparison uses the same
# 21 held-out genotypes across both conditions.

for feature in FEATURES:
    for horizon in HORIZONS:

        sets = []

        for condition in CONDITIONS:

            sub = data[
                (data.feature == feature)
                & (data.horizon_hours == horizon)
                & (data.condition == condition)
            ]

            sets.append(set(sub.test_genotype))

        if sets[0] != sets[1]:
            raise ValueError(
                f"Genotype mismatch: {feature}, {horizon}h"
            )

        if len(sets[0]) != 21:
            raise ValueError(
                f"Expected 21 genotypes: "
                f"{feature}, {horizon}h"
            )

data["contrast_pp"] = data[METRIC] * 100.0

data.to_csv(
    OUT / "Figure_F8_genotype_source.csv",
    index=False,
)

# -------------------------------------------------------
# Paired genotype bootstrap
# -------------------------------------------------------

results = []

for feature in FEATURES:
    for horizon in HORIZONS:

        sub = data[
            (data.feature == feature)
            & (data.horizon_hours == horizon)
        ]

        wide = sub.pivot(
            index="test_genotype",
            columns="condition",
            values="contrast_pp",
        ).sort_index()

        if wide.isna().any().any():
            raise ValueError("Incomplete paired matrix")

        n = len(wide)

        # Use identical resampled genotypes for
        # the two conditions of the comparison.
        indices = rng.integers(
            0, n, size=(B, n)
        )

        for condition in CONDITIONS:

            values = wide[
                condition
            ].to_numpy(dtype=float)

            estimates = (
                values[indices].mean(axis=1)
            )

            low, high = np.quantile(
                estimates, [.025, .975]
            )

            results.append({
                "feature": feature,
                "horizon": horizon,
                "condition": condition,
                "genotypes": n,
                "mean_pp": values.mean(),
                "ci_low_pp": low,
                "ci_high_pp": high,
                "positive_genotypes": int(
                    (values > 0).sum()
                ),
                "negative_genotypes": int(
                    (values < 0).sum()
                ),
            })

summary = pd.DataFrame(results)

summary.to_csv(
    OUT / "Figure_F8_source.csv",
    index=False,
)

# -------------------------------------------------------
# Figure: paired forest plots
# -------------------------------------------------------

fig, axes = plt.subplots(
    1,
    2,
    figsize=(10.8, 4.1),
    sharex=True,
    sharey=True,
)

positions = {
    "Unadjusted": 1,
    "Size-stratified": 0,
}

offsets = {
    24: +0.12,
    48: -0.12,
}

for ax, (feature, feature_label) in zip(
    axes, FEATURES.items()
):
    current = summary[
        summary.feature == feature
    ]

    for horizon in HORIZONS:

        color = COLORS[horizon]

        for condition in CONDITIONS:

            row = current[
                (current.horizon == horizon)
                & (current.condition == condition)
            ].iloc[0]

            x = float(row.mean_pp)

            y = (
                positions[condition]
                + offsets[horizon]
            )

            left = max(
                0.0,
                x - float(row.ci_low_pp)
            )
            right = max(
                0.0,
                float(row.ci_high_pp) - x
            )

            ax.errorbar(
                x,
                y,
                xerr=np.array([[left], [right]]),
                fmt="o",
                color=color,
                ecolor=color,
                markersize=6,
                markeredgecolor="white",
                markeredgewidth=0.65,
                elinewidth=1.7,
                capsize=3,
                capthick=1.25,
                zorder=4,
            )

    ax.axvline(
        0,
        color="#444444",
        linestyle=(0, (4, 3)),
        linewidth=1,
        zorder=1,
    )

    ax.set_yticks(
        [positions[c] for c in CONDITIONS]
    )
    ax.set_yticklabels(CONDITIONS)

    ax.set_ylim(-0.55, 1.55)

    ax.set_title(
        feature_label,
        pad=14,
        fontweight="medium",
    )

    ax.set_axisbelow(True)

    ax.grid(
        axis="x",
        color="#D7D7D7",
        alpha=0.65,
        linewidth=0.65,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(axis="y", length=0)

    ax.set_xlabel(
        "High minus low upper-tail misses\n"
        "(percentage points)"
    )

# Use common horizontal axis bounds.
min_x = float(summary.ci_low_pp.min())
max_x = float(summary.ci_high_pp.max())
span = max(max_x - min_x, 1.0)
pad = max(0.7, span * 0.10)

for ax in axes:
    ax.set_xlim(
        min(min_x - pad, -0.5),
        max_x + pad,
    )

# Panel letters outside axes.
axes[0].text(
    -0.06, 1.14, "(a)",
    transform=axes[0].transAxes,
    fontsize=12,
    fontweight="bold",
)
axes[1].text(
    -0.06, 1.14, "(b)",
    transform=axes[1].transAxes,
    fontsize=12,
    fontweight="bold",
)

handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=COLORS[h],
        markeredgecolor="white",
        markersize=7,
        label=f"{h}-hour forecast",
    )
    for h in HORIZONS
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.03),
    ncol=2,
    frameon=False,
    columnspacing=2.5,
)

fig.subplots_adjust(
    left=0.17,
    right=0.98,
    bottom=0.23,
    top=0.79,
    wspace=0.20,
)

# -------------------------------------------------------
# Export
# -------------------------------------------------------

stem = "Figure_F8_Growth_Size_Associations"

fig.savefig(
    OUT / f"{stem}.pdf",
    bbox_inches="tight",
)

fig.savefig(
    OUT / f"{stem}.svg",
    bbox_inches="tight",
)

fig.savefig(
    OUT / f"{stem}.png",
    dpi=450,
    bbox_inches="tight",
)

plt.close(fig)

print("\nFIGURE 8 — SUCCESS")
print("=" * 65)

print(
    summary[
        [
            "feature",
            "horizon",
            "condition",
            "mean_pp",
            "ci_low_pp",
            "ci_high_pp",
            "positive_genotypes",
        ]
    ].to_string(index=False)
)

print("\nSaved to:", OUT)
print(
    "\nNOTE: Intervals are retrospective, descriptive "
    "genotype-bootstrap percentile intervals."
)
print(
    "Size-stratified and unadjusted analyses use "
    "different aggregation estimands."
)
