
#!/usr/bin/env python3
"""
Supplementary Figure S3 — Plant Replication Sensitivity

Input:
  experiments/P3/P3_1/P3_1_subsampling_summary.csv

Output:
  figures/rebuilt/S3_replication_sensitivity/
    Figure_S3_Replication_Sensitivity.pdf
    Figure_S3_Replication_Sensitivity.png
    Figure_S3_Replication_Sensitivity.svg
    Figure_S3_source.csv

Two panels: 24h and 48h.
Four subsampling sizes: 4, 5, 6, 7 plants.
Ranges: 2.5th–97.5th simulation percentiles.
Black diamonds: original full-plant coverage.

No simulations are rerun.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT / "experiments/P3/P3_1/"
    "P3_1_subsampling_summary.csv"
)

OUT = (
    ROOT / "figures/rebuilt/"
    "S3_replication_sensitivity"
)

OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = (24, 48)
SAMPLE_SIZES = (4, 5, 6, 7)

COLORS = {
    4: "#4C78A8",
    5: "#E68627",
    6: "#65A69A",
    7: "#9673B9",
}

OFFSETS = {
    4: -0.27,
    5: -0.09,
    6: +0.09,
    7: +0.27,
}

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
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

# -----------------------------------------------------
# 1. Read existing P3.1 results
# -----------------------------------------------------

if not INPUT.is_file():
    raise FileNotFoundError(
        f"Cannot locate P3.1 results:\n{INPUT}"
    )

df = pd.read_csv(INPUT)

# Support slight differences in horizon column names.
if "horizon" not in df.columns:
    if "horizon_hours" in df.columns:
        df = df.rename(columns={
            "horizon_hours": "horizon"
        })
    else:
        raise ValueError(
            "Expected 'horizon' or 'horizon_hours'. "
            f"Available columns: {list(df.columns)}"
        )

required = {
    "horizon",
    "genotype",
    "sample_plants",
    "full_plants",
    "full_coverage",
    "p025",
    "p975",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        "Missing required P3.1 summary columns: "
        f"{sorted(missing)}\n"
        f"Available: {list(df.columns)}"
    )

for col in [
    "horizon",
    "sample_plants",
    "full_plants",
    "full_coverage",
    "p025",
    "p975",
]:
    df[col] = pd.to_numeric(
        df[col], errors="raise"
    )

df = df[
    df["horizon"].isin(HORIZONS)
    & df["sample_plants"].isin(SAMPLE_SIZES)
].copy()

if df.empty:
    raise ValueError("No eligible P3.1 records found")

if df.duplicated([
    "horizon", "genotype", "sample_plants"
]).any():
    raise ValueError(
        "Duplicate genotype/horizon/sample-size cells"
    )

# The original P3.1 design included eight source
# genotypes with >=20 independent plants.
for horizon in HORIZONS:
    sub = df[df.horizon == horizon]

    if sub["genotype"].nunique() != 8:
        raise ValueError(
            f"{horizon}h: expected 8 source genotypes"
        )

    for g, group in sub.groupby("genotype"):
        if set(group.sample_plants) != set(SAMPLE_SIZES):
            raise ValueError(
                f"{horizon}h, {g}: missing sample sizes"
            )

        if group.full_plants.nunique() != 1:
            raise ValueError(
                f"{horizon}h, {g}: inconsistent plant count"
            )

        if group.full_coverage.nunique() != 1:
            raise ValueError(
                f"{horizon}h, {g}: inconsistent full coverage"
            )

        if group.full_plants.iloc[0] < 20:
            raise ValueError(
                f"{horizon}h, {g}: source has <20 plants"
            )

for col in ["full_coverage", "p025", "p975"]:
    if (
        df[col].isna().any()
        or not df[col].between(0, 1).all()
    ):
        raise ValueError(
            f"{col} must contain valid fractions in [0,1]"
        )

if not (df.p025 <= df.p975).all():
    raise ValueError("Reversed percentile limits")

# Verify identical genotype sets for both horizons.
g24 = set(df[df.horizon == 24].genotype)
g48 = set(df[df.horizon == 48].genotype)

if g24 != g48:
    raise ValueError(
        "The two horizons have different source genotypes"
    )

# -----------------------------------------------------
# 2. Order genotypes by original 48h coverage
# -----------------------------------------------------

full48 = (
    df[
        (df.horizon == 48)
        & (df.sample_plants == 4)
    ]
    .sort_values(
        "full_coverage",
        ascending=True
    )
)

ORDER = full48.genotype.tolist()

plant_counts = (
    full48.set_index("genotype")
    ["full_plants"]
    .astype(int)
    .to_dict()
)

# Convert fractions to percentages for plotting.
for col in ["p025", "p975", "full_coverage"]:
    df[col + "_pct"] = 100.0 * df[col]

df["genotype_order"] = df.genotype.map(
    {g: i for i, g in enumerate(ORDER)}
)

df = df.sort_values([
    "horizon",
    "genotype_order",
    "sample_plants"
])

df.to_csv(
    OUT / "Figure_S3_source.csv",
    index=False
)

# -----------------------------------------------------
# 3. Two-panel publication figure
# -----------------------------------------------------

fig, axes = plt.subplots(
    1, 2,
    figsize=(11.4, 5.8),
    sharey=True
)

all_percentiles = np.concatenate([
    df.p025_pct.to_numpy(),
    df.p975_pct.to_numpy(),
    df.full_coverage_pct.to_numpy(),
    [90.0],
])

xmin = max(
    0.0,
    np.floor((all_percentiles.min() - 2.0) / 5) * 5
)

xmax = min(
    100.0,
    np.ceil((all_percentiles.max() + 2.0) / 5) * 5
)

for ax, horizon in zip(axes, HORIZONS):

    sub = df[df.horizon == horizon]

    for i, genotype in enumerate(ORDER):

        gd = sub[sub.genotype == genotype]

        for n in SAMPLE_SIZES:

            row = gd[
                gd.sample_plants == n
            ].iloc[0]

            lo = float(row.p025_pct)
            hi = float(row.p975_pct)
            y = i + OFFSETS[n]

            # Display the simulation percentile
            # interval without inventing a mean
            # or median from the endpoints.
            ax.hlines(
                y,
                lo,
                hi,
                color=COLORS[n],
                linewidth=2.0,
                zorder=3
            )

            ax.plot(
                [lo, hi],
                [y, y],
                marker="|",
                linestyle="none",
                markersize=6.5,
                markeredgewidth=1.3,
                color=COLORS[n],
                zorder=4,
            )

        # Full plant replication reference.
        original = float(
            gd.full_coverage_pct.iloc[0]
        )

        ax.scatter(
            original,
            i,
            marker="D",
            s=42,
            facecolor="#222222",
            edgecolor="white",
            linewidth=0.6,
            zorder=6,
        )

    ax.axvline(
        90,
        color="#555555",
        linestyle=(0, (4, 3)),
        linewidth=1.1,
        zorder=1,
    )

    ax.set_title(
        f"{horizon}-hour forecast",
        pad=13,
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.7, len(ORDER) - 0.3)

    ax.set_yticks(range(len(ORDER)))

    labels = [
        f"{g} (N={plant_counts[g]})"
        for g in ORDER
    ]

    if horizon == 24:
        ax.set_yticklabels(labels)
    else:
        ax.tick_params(
            axis="y",
            labelleft=False
        )

    ax.invert_yaxis()
    ax.set_xlabel("Empirical coverage (%)")

    ax.set_axisbelow(True)
    ax.grid(
        axis="x",
        color="#DADADA",
        alpha=0.7,
        linewidth=0.7
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)

# Panel labels.
for ax, label in zip(axes, ("(a)", "(b)")):
    ax.text(
        -0.07,
        1.06,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom"
    )

# -----------------------------------------------------
# 4. Shared legend
# -----------------------------------------------------

handles = []

for n in SAMPLE_SIZES:
    handles.append(
        Line2D(
            [0], [0],
            color=COLORS[n],
            linewidth=2.0,
            marker="|",
            markersize=8,
            label=f"{n} plants"
        )
    )

handles.append(
    Line2D(
        [0], [0],
        marker="D",
        color="none",
        markerfacecolor="#222222",
        markeredgecolor="white",
        markersize=7,
        label="Full replication",
    )
)

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.53, 1.005),
    ncol=5,
    frameon=False,
    columnspacing=1.4,
    handlelength=1.7,
)

fig.subplots_adjust(
    left=0.18,
    right=0.98,
    bottom=0.14,
    top=0.83,
    wspace=0.16,
)

# -----------------------------------------------------
# 5. Save publication files
# -----------------------------------------------------

STEM = "Figure_S3_Replication_Sensitivity"

fig.savefig(
    OUT / f"{STEM}.pdf",
    bbox_inches="tight"
)

fig.savefig(
    OUT / f"{STEM}.svg",
    bbox_inches="tight"
)

fig.savefig(
    OUT / f"{STEM}.png",
    dpi=450,
    bbox_inches="tight"
)

plt.close(fig)

print("=" * 68)
print("SUPPLEMENTARY FIGURE S3 — SUCCESS")
print("=" * 68)
print("Eight high-replication genotypes per horizon")
print("Subsampling sizes: 4, 5, 6, 7 plants")
print("Percentile ranges: 2.5%–97.5%")
print("Reference: original full-plant empirical coverage")
print("Outputs:", OUT)

print(
    "\nThe percentile ranges describe simulated subsampling "
    "variation conditional on observed plants."
)
print(
    "They are not confidence intervals for a new-genotype "
    "population or a proof that undercoverage is sampling noise."
)
