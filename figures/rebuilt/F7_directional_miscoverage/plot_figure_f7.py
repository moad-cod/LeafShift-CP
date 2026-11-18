#!/usr/bin/env python3
"""
Figure F7 — Directional Miscoverage by Genotype

Reads:
    experiments/P3/P3_2/P3_2_genotype_directional.csv

Writes:
    figures/rebuilt/F7_directional_miscoverage/
        Figure_F7_Directional_Miscoverage.pdf
        Figure_F7_Directional_Miscoverage.png
        Figure_F7_Directional_Miscoverage.svg
        Figure_F7_source.csv

Design:
- two-panel horizontal stacked bar chart
- panel (a): 24h
- panel (b): 48h
- bars split into lower-tail and upper-tail miscoverage
- genotype order fixed by 48h total miscoverage (descending)
- highlight Ei-6 and Ws-2
"""

from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]

INPUT = ROOT / "experiments" / "P3" / "P3_2" / "P3_2_genotype_directional.csv"
OUTDIR = ROOT / "figures" / "rebuilt" / "F7_directional_miscoverage"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#222222",
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "legend.fontsize": 9,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
})

# scientific palette
LOWER_COLOR = "#4C78A8"   # muted blue
UPPER_COLOR = "#F58518"   # orange
EDGE_COLOR = "#2F2F2F"
GRID_COLOR = "#D9D9D9"
HIGHLIGHT_GENOTYPES = {"Ei-6", "Ws-2"}

# ---------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------
if not INPUT.is_file():
    raise FileNotFoundError(f"Missing input file:\n{INPUT}")

df = pd.read_csv(INPUT)

required = {
    "horizon",
    "genotype",
    "plants",
    "lower_miss",
    "upper_miss",
    "coverage",
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

# Clean / standardize
df["horizon"] = pd.to_numeric(df["horizon"], errors="raise").astype(int)
df["plants"] = pd.to_numeric(df["plants"], errors="raise").astype(int)
for col in ["lower_miss", "upper_miss", "coverage"]:
    df[col] = pd.to_numeric(df[col], errors="raise")

df["total_miss"] = df["lower_miss"] + df["upper_miss"]

# keep only 24h and 48h
df = df[df["horizon"].isin([24, 48])].copy()

# there should be one row per genotype per horizon
dup = df.duplicated(["horizon", "genotype"])
if dup.any():
    raise ValueError("Duplicate genotype/horizon rows found")

# ---------------------------------------------------------------------
# Fixed genotype order: sort by 48h total miscoverage descending
# ---------------------------------------------------------------------
order48 = (
    df[df["horizon"] == 48]
    .sort_values(["total_miss", "upper_miss", "lower_miss"], ascending=[False, False, False])
    ["genotype"]
    .tolist()
)

# Make sure 24h has the same genotypes
g24 = set(df[df["horizon"] == 24]["genotype"])
g48 = set(df[df["horizon"] == 48]["genotype"])
if g24 != g48:
    raise ValueError("24h and 48h genotype sets do not match")

# Apply order
df["genotype"] = pd.Categorical(df["genotype"], categories=order48, ordered=True)
df = df.sort_values(["horizon", "genotype"]).reset_index(drop=True)

# Build display labels with plant counts
plant_map = (
    df[["genotype", "plants"]]
    .drop_duplicates()
    .set_index("genotype")["plants"]
    .to_dict()
)

label_map = {
    g: f"{g} (n={plant_map[g]})"
    for g in order48
}

# Convert to percentages for plotting and export
for col in ["lower_miss", "upper_miss", "coverage", "total_miss"]:
    df[f"{col}_pct"] = df[col] * 100.0

# Save sorted source data for the figure
source = df.copy()
source["genotype_label"] = source["genotype"].map(label_map)
source = source[
    [
        "horizon",
        "genotype",
        "genotype_label",
        "plants",
        "lower_miss_pct",
        "upper_miss_pct",
        "total_miss_pct",
        "coverage_pct",
    ]
].rename(columns={
    "lower_miss_pct": "lower_tail_miscoverage_pct",
    "upper_miss_pct": "upper_tail_miscoverage_pct",
    "total_miss_pct": "total_miscoverage_pct",
    "coverage_pct": "coverage_pct",
})
source.to_csv(OUTDIR / "Figure_F7_source.csv", index=False)

# ---------------------------------------------------------------------
# Plot helper
# ---------------------------------------------------------------------
def build_panel(ax, horizon, show_ylabels):
    d = (
        df[df["horizon"] == horizon]
        .set_index("genotype")
        .loc[order48]
        .reset_index()
    )

    y = np.arange(len(d))
    lower = d["lower_miss_pct"].to_numpy()
    upper = d["upper_miss_pct"].to_numpy()
    total = lower + upper

    # Main stacked bars
    ax.barh(
        y,
        lower,
        color=LOWER_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
        height=0.72,
        label="Lower-tail miss",
        zorder=3,
    )
    ax.barh(
        y,
        upper,
        left=lower,
        color=UPPER_COLOR,
        edgecolor=EDGE_COLOR,
        linewidth=0.5,
        height=0.72,
        label="Upper-tail miss",
        zorder=3,
    )

    # Highlight specific genotypes with thicker border + star
    for i, row in d.iterrows():
        g = row["genotype"]
        if g in HIGHLIGHT_GENOTYPES:
            ax.barh(
                i,
                total[i],
                left=0,
                facecolor="none",
                edgecolor="#111111",
                linewidth=1.4,
                height=0.72,
                zorder=4,
            )
            ax.text(
                total[i] + 0.55,
                i,
                "★",
                va="center",
                ha="left",
                fontsize=10.5,
                color="#111111",
                zorder=5,
            )

    ax.axvline(
        10,
        color="#666666",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        zorder=2,
    )

    ax.set_title(f"{horizon}-hour forecast", pad=10)
    ax.set_xlabel("Genotype-level empirical miscoverage (%)")
    ax.set_axisbelow(True)
    ax.grid(axis="x", color=GRID_COLOR, linewidth=0.7, alpha=0.85, zorder=1)

    ax.set_yticks(y)
    if show_ylabels:
        ax.set_yticklabels([label_map[g] for g in d["genotype"]])
        for tick in ax.get_yticklabels():
            if tick.get_text().startswith("Ei-6") or tick.get_text().startswith("Ws-2"):
                tick.set_fontweight("bold")
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.invert_yaxis()

    # subtle clean spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    return total.max()

# ---------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------
fig, axes = plt.subplots(
    1, 2,
    figsize=(12.6, 8.4),
    sharey=True,
    gridspec_kw={"wspace": 0.06}
)

max24 = build_panel(axes[0], 24, show_ylabels=True)
max48 = build_panel(axes[1], 48, show_ylabels=False)

# same x-limit in both panels
xmax = max(max24, max48, 10)
xmax = int(math.ceil((xmax + 1.5) / 2.0) * 2)  # round up nicely
for ax in axes:
    ax.set_xlim(0, xmax)

# Panel letters
axes[0].text(
    -0.12, 1.03, "(a)",
    transform=axes[0].transAxes,
    fontsize=12,
    fontweight="bold",
    ha="left", va="bottom"
)
axes[1].text(
    -0.12, 1.03, "(b)",
    transform=axes[1].transAxes,
    fontsize=12,
    fontweight="bold",
    ha="left", va="bottom"
)

# Shared y label on left side
axes[0].set_ylabel("Held-out genotype")

# Figure-level legend
handles = [
    plt.Rectangle((0, 0), 1, 1, facecolor=LOWER_COLOR, edgecolor=EDGE_COLOR, linewidth=0.5),
    plt.Rectangle((0, 0), 1, 1, facecolor=UPPER_COLOR, edgecolor=EDGE_COLOR, linewidth=0.5),
]
labels = ["Lower-tail miss", "Upper-tail miss"]
fig.legend(
    handles,
    labels,
    ncol=2,
    frameon=False,
    loc="upper center",
    bbox_to_anchor=(0.5, 1.01),
    columnspacing=1.8,
    handlelength=1.6,
)

# Figure title
fig.suptitle(
    "Directional miscoverage of U5 prediction intervals across held-out genotypes",
    y=1.055,
    fontsize=13,
    fontweight="bold",
)

# Small note
fig.text(
    0.5,
    0.008,
    "Bars show lower-tail and upper-tail genotype-level macro miscoverage. "
    "Dashed line = nominal 10% total miscoverage corresponding to 90% coverage.",
    ha="center",
    va="bottom",
    fontsize=9,
    color="#444444",
)

# ---------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------
pdf_path = OUTDIR / "Figure_F7_Directional_Miscoverage.pdf"
png_path = OUTDIR / "Figure_F7_Directional_Miscoverage.png"
svg_path = OUTDIR / "Figure_F7_Directional_Miscoverage.svg"

fig.savefig(pdf_path)
fig.savefig(svg_path)
fig.savefig(png_path, dpi=450)
plt.close(fig)

print("F7 created successfully.")
print("Saved to:")
print(" -", pdf_path)
print(" -", png_path)
print(" -", svg_path)
print(" -", OUTDIR / "Figure_F7_source.csv")
