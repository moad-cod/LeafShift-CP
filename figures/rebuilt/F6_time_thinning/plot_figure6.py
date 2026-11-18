from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[3]
TABLE_DIR = ROOT / "experiments" / "P2" / "P2_4" / "tables"
OUT_DIR = ROOT / "figures" / "rebuilt" / "F6_time_thinning"
OUT_DIR.mkdir(parents=True, exist_ok=True)

GENO_FILE = TABLE_DIR / "p2_4_genotype_comparison.csv"
SUMMARY_FILE = TABLE_DIR / "p2_4_summary.csv"

# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------
geno = pd.read_csv(GENO_FILE)
summary = pd.read_csv(SUMMARY_FILE)

# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------
required_geno_cols = {
    "horizon_hours", "test_genotype",
    "original_coverage", "thinned_coverage",
    "assignments_original", "assignments_thinned",
    "plants_original", "plants_thinned",
    "original_unique_windows", "selected_unique_windows",
    "coverage_change_pp"
}
required_summary_cols = {
    "horizon_hours",
    "original_unique_windows", "selected_unique_windows", "retention_fraction",
    "selected_plants",
    "original_macro_coverage", "thinned_macro_coverage",
    "original_worst_coverage", "thinned_worst_coverage",
    "original_worst_genotype", "thinned_worst_genotype",
    "original_genotypes_below_90", "thinned_genotypes_below_90",
    "coverage_rank_correlation"
}

missing_geno = required_geno_cols - set(geno.columns)
missing_summary = required_summary_cols - set(summary.columns)
if missing_geno:
    raise ValueError(f"Missing columns in genotype table: {sorted(missing_geno)}")
if missing_summary:
    raise ValueError(f"Missing columns in summary table: {sorted(missing_summary)}")

expected_horizons = {24, 48}
if set(geno["horizon_hours"].unique()) != expected_horizons:
    raise ValueError(f"Unexpected horizons in genotype table: {sorted(geno['horizon_hours'].unique())}")
if set(summary["horizon_hours"].unique()) != expected_horizons:
    raise ValueError(f"Unexpected horizons in summary table: {sorted(summary['horizon_hours'].unique())}")

counts = geno.groupby("horizon_hours")["test_genotype"].nunique().to_dict()
if counts != {24: 21, 48: 21}:
    raise ValueError(f"Expected 21 genotypes per horizon, got: {counts}")

if not (geno["assignments_original"] == 4).all():
    raise ValueError("Not all original genotype summaries have four assignments.")
if not (geno["assignments_thinned"] == 4).all():
    raise ValueError("Not all thinned genotype summaries have four assignments.")

if not (geno["plants_original"] == geno["plants_thinned"]).all():
    raise ValueError("Original and thinned plant counts differ for at least one genotype.")

if not ((geno["original_coverage"] >= 0) & (geno["original_coverage"] <= 1)).all():
    raise ValueError("Original coverage outside [0, 1].")
if not ((geno["thinned_coverage"] >= 0) & (geno["thinned_coverage"] <= 1)).all():
    raise ValueError("Thinned coverage outside [0, 1].")

# -----------------------------------------------------------------------------
# Cross-check archived summary values
# -----------------------------------------------------------------------------
tol = 1e-10

for horizon in [24, 48]:
    g = geno.loc[geno["horizon_hours"] == horizon].copy()
    s = summary.loc[summary["horizon_hours"] == horizon].iloc[0]

    macro_orig = g["original_coverage"].mean()
    macro_thin = g["thinned_coverage"].mean()

    worst_orig_row = g.loc[g["original_coverage"].idxmin()]
    worst_thin_row = g.loc[g["thinned_coverage"].idxmin()]

    below90_orig = int((g["original_coverage"] < 0.90).sum())
    below90_thin = int((g["thinned_coverage"] < 0.90).sum())

    if abs(macro_orig - s["original_macro_coverage"]) > tol:
        raise ValueError(f"{horizon}h original macro coverage mismatch.")
    if abs(macro_thin - s["thinned_macro_coverage"]) > tol:
        raise ValueError(f"{horizon}h thinned macro coverage mismatch.")
    if abs(worst_orig_row["original_coverage"] - s["original_worst_coverage"]) > tol:
        raise ValueError(f"{horizon}h original worst coverage mismatch.")
    if abs(worst_thin_row["thinned_coverage"] - s["thinned_worst_coverage"]) > tol:
        raise ValueError(f"{horizon}h thinned worst coverage mismatch.")
    if str(worst_orig_row["test_genotype"]) != str(s["original_worst_genotype"]):
        raise ValueError(f"{horizon}h original worst genotype mismatch.")
    if str(worst_thin_row["test_genotype"]) != str(s["thinned_worst_genotype"]):
        raise ValueError(f"{horizon}h thinned worst genotype mismatch.")
    if below90_orig != int(s["original_genotypes_below_90"]):
        raise ValueError(f"{horizon}h original below-90 count mismatch.")
    if below90_thin != int(s["thinned_genotypes_below_90"]):
        raise ValueError(f"{horizon}h thinned below-90 count mismatch.")

    print(
        f"{horizon}h: "
        f"retained={int(s['selected_unique_windows'])}/{int(s['original_unique_windows'])} "
        f"({100*s['retention_fraction']:.2f}%), "
        f"macro={100*macro_orig:.2f}% -> {100*macro_thin:.2f}%, "
        f"worst={s['original_worst_genotype']} {100*s['original_worst_coverage']:.2f}% -> "
        f"{100*s['thinned_worst_coverage']:.2f}% "
        f"— PASS"
    )

# -----------------------------------------------------------------------------
# Prepare plotting data
# Common order: original 48h coverage ascending
# -----------------------------------------------------------------------------
order_48 = (
    geno.loc[geno["horizon_hours"] == 48, ["test_genotype", "original_coverage"]]
    .sort_values(["original_coverage", "test_genotype"], ascending=[True, True])
    ["test_genotype"]
    .tolist()
)

plant_counts = (
    geno.loc[geno["horizon_hours"] == 48, ["test_genotype", "plants_original"]]
    .drop_duplicates()
    .set_index("test_genotype")["plants_original"]
    .to_dict()
)

label_map = {
    g: f"{g}  (n={plant_counts[g]})"
    for g in order_48
}

# -----------------------------------------------------------------------------
# Style
# -----------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 10.5,
    "ytick.labelsize": 11,
})

COL_ORIG = "#6e7781"     # open original points
COL_LINE = "#b0b7bf"     # pair connector
COL_THIN_OK = "#1f77b4"  # thinned >= 90
COL_THIN_LOW = "#d55e00" # thinned < 90
COL_THRESH = "#616973"

# -----------------------------------------------------------------------------
# Plot
# -----------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.2), sharey=True)
fig.subplots_adjust(left=0.21, right=0.985, top=0.87, bottom=0.18, wspace=0.12)

for ax, horizon, panel in zip(axes, [24, 48], ["(a)", "(b)"]):
    g = geno.loc[geno["horizon_hours"] == horizon].copy()
    g["test_genotype"] = pd.Categorical(g["test_genotype"], categories=order_48, ordered=True)
    g = g.sort_values("test_genotype").reset_index(drop=True)

    y = np.arange(len(g))

    orig = 100 * g["original_coverage"].to_numpy()
    thin = 100 * g["thinned_coverage"].to_numpy()

    # background / reference
    ax.axvspan(0, 90, color="#f6f7f9", zorder=0)
    ax.axvline(90, color=COL_THRESH, linestyle="--", linewidth=1.6, zorder=1)

    # paired horizontal segments
    for yi, x0, x1 in zip(y, orig, thin):
        ax.hlines(yi, min(x0, x1), max(x0, x1), color=COL_LINE, lw=1.8, zorder=2)

    # original points (open gray)
    ax.scatter(
        orig, y,
        s=34,
        facecolors="white",
        edgecolors=COL_ORIG,
        linewidths=1.0,
        zorder=3
    )

    # thinned points (filled, colored by target crossing)
    thin_colors = np.where(thin >= 90.0, COL_THIN_OK, COL_THIN_LOW)
    ax.scatter(
        thin, y,
        s=54,
        c=thin_colors,
        edgecolors=thin_colors,
        linewidths=1.0,
        zorder=4
    )

    s = summary.loc[summary["horizon_hours"] == horizon].iloc[0]
    retained_pct = 100 * s["retention_fraction"]
    macro_orig = 100 * s["original_macro_coverage"]
    macro_thin = 100 * s["thinned_macro_coverage"]
    below_orig = int(s["original_genotypes_below_90"])
    below_thin = int(s["thinned_genotypes_below_90"])

    ax.text(
        0.0, 1.05,
        f"{panel} {horizon}-hour forecast",
        transform=ax.transAxes,
        ha="left", va="bottom",
        fontsize=16, fontweight="bold"
    )
    ax.text(
        0.0, 1.015,
        f"Retained {retained_pct:.2f}% | "
        f"Macro {macro_orig:.2f}% → {macro_thin:.2f}%",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
        color="#4d4d4d"
    )

    ax.set_xlim(74, 100.5)
    ax.set_xticks([75, 80, 85, 90, 95, 100])
    ax.set_xticklabels([f"{t:d}%" for t in [75, 80, 85, 90, 95, 100]])
    ax.grid(axis="x", color="#d9dde3", linewidth=0.8)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Empirical coverage (%)")

    if ax is axes[0]:
        ax.set_yticks(y)
        ax.set_yticklabels([label_map[g] for g in g["test_genotype"]])
        ax.set_ylabel("Held-out genotype (plant count)")
    else:
        ax.set_yticks(y)
        ax.tick_params(axis="y", labelleft=False)

    ax.set_ylim(len(g) - 0.5, -0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9aa0a6")
    ax.spines["bottom"].set_color("#9aa0a6")

# Legend
legend_items = [
    Line2D([0], [0], marker="o", linestyle="none",
           markerfacecolor="white", markeredgecolor=COL_ORIG,
           markeredgewidth=1.2, markersize=7,
           label="Original evaluation"),
    Line2D([0], [0], marker="o", linestyle="none",
           markerfacecolor=COL_THIN_OK, markeredgecolor=COL_THIN_OK,
           markersize=7, label="Thinned evaluation ≥90%"),
    Line2D([0], [0], marker="o", linestyle="none",
           markerfacecolor=COL_THIN_LOW, markeredgecolor=COL_THIN_LOW,
           markersize=7, label="Thinned evaluation <90%"),
    Line2D([0], [0], color=COL_LINE, lw=1.8,
           label="Paired genotype change"),
    Line2D([0], [0], color=COL_THRESH, lw=1.6, linestyle="--",
           label="Nominal 90%"),
]
fig.legend(
    handles=legend_items,
    loc="lower center",
    ncol=5,
    frameon=False,
    fontsize=11,
    bbox_to_anchor=(0.5, 0.06),
    handletextpad=0.6,
    columnspacing=1.4
)

fig.text(
    0.5, 0.015,
    "Nonoverlapping-window sensitivity reuses the same fitted models and held-out genotypes; "
    "it tests evaluation robustness, not independent generalization.",
    ha="center", va="bottom", fontsize=10.5, color="#4d4d4d"
)

# -----------------------------------------------------------------------------
# Save source data used in the figure
# -----------------------------------------------------------------------------
source = geno.copy()
source["plot_order"] = source["test_genotype"].map({g: i for i, g in enumerate(order_48)})
source = source.sort_values(["horizon_hours", "plot_order"]).drop(columns=["plot_order"])
source.to_csv(OUT_DIR / "F6_time_thinning_source.csv", index=False)

# -----------------------------------------------------------------------------
# Save figure
# -----------------------------------------------------------------------------
pdf_path = OUT_DIR / "F6_nonoverlapping_window_sensitivity.pdf"
svg_path = OUT_DIR / "F6_nonoverlapping_window_sensitivity.svg"
png_path = OUT_DIR / "F6_nonoverlapping_window_sensitivity.png"

fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")
fig.savefig(png_path, dpi=360, bbox_inches="tight")
plt.close(fig)

print("VALIDATION PASS: 21 genotypes × 2 horizons; original/thinned summary cross-check passed.")
print(f"Saved source: {OUT_DIR / 'F6_time_thinning_source.csv'}")
print("Saved:")
print(f"  {pdf_path}")
print(f"  {svg_path}")
print(f"  {png_path}")
