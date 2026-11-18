from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path.cwd()
OUT = ROOT / "figures/rebuilt/F4_genotype_reliability"
OUT.mkdir(parents=True, exist_ok=True)

E3 = ROOT / "experiments/E3/tables/E3_2_fold_metrics.csv"
CI = ROOT / "experiments/P1/P1_1/statistics/tables/p1_genotype_coverage_ci.csv"
PLANTS = ROOT / "experiments/P1/P1_1/tables/P1_1_U5_plant_assignment_coverage.csv"

METHOD = "U5_CQR_balanced"

EXPECTED = {
    24: (0.8997119769358646, "Ei-6", 0.8221703622232337, 8),
    48: (0.9004006113622590, "Ws-2", 0.7639306084000144, 6),
}

def check(condition, message):
    if not condition:
        raise ValueError(message)

fold = pd.read_csv(E3)
ci = pd.read_csv(CI)
plant = pd.read_csv(PLANTS)

fold = fold[fold["method"] == METHOD].copy()
ci = ci[ci["method"] == METHOD].copy()
plant = plant[plant["method"] == METHOD].copy()

keys = ["horizon_hours", "test_genotype"]

# Verify four assignments per test genotype.
counts = fold.groupby(keys)["inner_fold"].nunique()
check(len(counts) == 42 and (counts == 4).all(),
      "Expected four U5 assignments for every genotype/horizon.")

check(len(fold) == 168, "Expected 168 U5 fold rows.")
check(len(ci) == 42, "Expected 42 bootstrap rows.")
check(not ci.duplicated(keys).any(), "Duplicate bootstrap keys.")

# Reconstruction 1: average assignment-specific fold coverage.
fold_stats = (
    fold.groupby(keys, as_index=False)["coverage"]
    .mean()
    .rename(columns={"coverage": "fold_coverage"})
)

# Reconstruction 2: equal-plant means within each assignment,
# followed by averaging four assignments.
plant_stats = (
    plant.groupby(keys + ["inner_fold"], as_index=False)
    .agg(
        plant_mean=("coverage", "mean"),
        plant_count=("plant_id", "nunique")
    )
)

assignment_counts = plant_stats.groupby(keys)["inner_fold"].nunique()
check((assignment_counts == 4).all(),
      "Plant table does not have all four assignments.")

plant_recon = (
    plant_stats.groupby(keys, as_index=False)["plant_mean"]
    .mean()
    .rename(columns={"plant_mean": "plant_coverage"})
)

actual_plants = (
    plant.groupby(keys, as_index=False)["plant_id"]
    .nunique()
    .rename(columns={"plant_id": "actual_plants"})
)

data = (
    ci.merge(fold_stats, on=keys, validate="one_to_one")
      .merge(plant_recon, on=keys, validate="one_to_one")
      .merge(actual_plants, on=keys, validate="one_to_one")
)

check(len(data) == 42, "Merged data must contain exactly 42 rows.")

for column in ["fold_coverage", "plant_coverage"]:
    difference = np.abs(data["coverage"] - data[column]).max()
    print(f"Maximum difference vs {column}: {difference:.3e}")
    check(difference < 1e-9, f"Coverage mismatch: {column}")

check((data["plants"] == data["actual_plants"]).all(),
      "Bootstrap plant counts disagree with plant table.")

check(((data["ci_lower"] <= data["coverage"]) &
       (data["coverage"] <= data["ci_upper"])).all(),
      "A coverage estimate lies outside its bootstrap interval.")

genotypes_24 = set(data.loc[data.horizon_hours == 24, "test_genotype"])
genotypes_48 = set(data.loc[data.horizon_hours == 48, "test_genotype"])
check(genotypes_24 == genotypes_48 and len(genotypes_24) == 21,
      "Genotype identities differ across horizons.")

for horizon, (macro_expected, worst_name, worst_value, below_expected) in EXPECTED.items():
    subset = data[data.horizon_hours == horizon]
    macro = subset.coverage.mean()
    worst = subset.loc[subset.coverage.idxmin()]
    below = int((subset.coverage < 0.90).sum())

    check(abs(macro - macro_expected) < 1e-9,
          f"Macro coverage mismatch at {horizon}h.")
    check(worst.test_genotype == worst_name,
          f"Worst genotype mismatch at {horizon}h.")
    check(abs(worst.coverage - worst_value) < 1e-9,
          f"Worst coverage mismatch at {horizon}h.")
    check(below == below_expected,
          f"Below-90 count mismatch at {horizon}h.")

    print(
        f"{horizon}h: macro={macro:.8f}, "
        f"worst={worst_name} ({worst.coverage:.8f}), "
        f"below90={below}/21 — PASS"
    )

# Export validated, full-precision figure source data.
result = data.rename(columns={
    "test_genotype": "genotype",
    "horizon_hours": "horizon",
    "ci_lower": "ci_low",
    "ci_upper": "ci_high",
    "plants": "n_plants",
})

result = result[
    ["genotype", "horizon", "coverage",
     "ci_low", "ci_high", "n_plants", "status"]
].sort_values(["horizon", "genotype"])

result.to_csv(
    OUT / "F4_U5_genotype_reliability_input.csv",
    index=False,
    float_format="%.16g"
)

# Shared genotype ordering: lowest 48 h coverage first. Keeping one order across
# panels makes horizon-specific changes directly comparable row by row.
order = (
    result[result.horizon == 48]
    .sort_values("coverage")["genotype"]
    .tolist()
)

# Assignment-specific values are paired sensitivity evaluations, not independent
# replicates. They are shown as small open points behind the assignment-averaged
# genotype estimate.
assignment_data = fold[
    ["horizon_hours", "test_genotype", "inner_fold", "coverage"]
].rename(columns={
    "horizon_hours": "horizon",
    "test_genotype": "genotype",
})

# Plot settings: sized for a full-width, two-column journal figure. PDF and SVG
# retain vector geometry and editable text; PNG is exported at 600 dpi.
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 9.5,
    "axes.linewidth": 0.7,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.2,
    "pdf.fonttype": 42,
    "svg.fonttype": "none",
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
})

BLUE = "#0072B2"       # Okabe-Ito blue
VERMILLION = "#D55E00" # Okabe-Ito vermillion
DARK_GREY = "#5F6368"
MID_GREY = "#9AA0A6"
LIGHT_GREY = "#E6E9ED"

fig, axes = plt.subplots(
    1, 2, figsize=(9.0, 6.65), sharey=True,
    gridspec_kw={"wspace": 0.22}
)

minimum = min(result.ci_low.min(), result.coverage.min())
xmin = max(0.0, np.floor((minimum - 0.015) * 20) / 20)
xmax = 1.0
ticks = np.arange(max(0.0, np.ceil(xmin * 20) / 20), 1.001, 0.05)

for ax, horizon, panel in zip(axes, [24, 48], ["a", "b"]):
    sub = (
        result[result.horizon == horizon]
        .set_index("genotype")
        .loc[order]
        .reset_index()
    )

    y_positions = np.arange(len(sub))
    ax.axvspan(xmin, 0.90, color="#F7F7F7", zorder=0)
    ax.axvline(0.90, color=DARK_GREY, linestyle=(0, (4, 2)),
               linewidth=1.0, zorder=1)

    for pos, record in enumerate(sub.itertuples()):
        unstable = record.n_plants < 8
        under = record.coverage < 0.90
        color = VERMILLION if under else BLUE

        assignments = assignment_data[
            (assignment_data.horizon == horizon)
            & (assignment_data.genotype == record.genotype)
        ].sort_values("inner_fold")
        check(len(assignments) == 4,
              f"Expected four assignment values: {horizon}h/{record.genotype}")

        # Conditional plant-bootstrap percentile interval.
        ax.plot(
            [record.ci_low, record.ci_high],
            [pos, pos],
            color=MID_GREY, linewidth=1.2,
            zorder=2
        )
        ax.plot(
            [record.ci_low, record.ci_low],
            [pos - 0.10, pos + 0.10],
            color=MID_GREY, linewidth=0.75
        )
        ax.plot(
            [record.ci_high, record.ci_high],
            [pos - 0.10, pos + 0.10],
            color=MID_GREY, linewidth=0.75
        )

        # Four paired fitting/calibration assignments. Tiny deterministic vertical
        # offsets prevent coincident points from hiding one another.
        ax.scatter(
            assignments.coverage,
            pos + np.array([-0.15, -0.05, 0.05, 0.15]),
            s=10,
            marker="o",
            facecolors="white",
            edgecolors=DARK_GREY,
            linewidths=0.55,
            alpha=0.85,
            zorder=3,
        )

        # Assignment-averaged genotype estimate. Marker fill indicates whether the
        # observed estimate is below the nominal target; plant instability is
        # encoded in the row label rather than conflated with a second marker shape.
        ax.scatter(
            record.coverage, pos,
            s=28,
            marker="D",
            facecolors=color,
            edgecolors=color,
            linewidths=0.7,
            zorder=4
        )

    macro = sub.coverage.mean()
    below = int((sub.coverage < 0.90).sum())

    # A two-line axis title prevents the title/subtitle collision.
    # Use two decimals: 89.97% and 90.04% should not both display as 90.0%.
    ax.set_title(
        f"({panel}) {horizon}-hour forecast\n"
        f"Macro {macro:.2%}  ·  {below}/21 below 90%",
        loc="left", weight="bold", pad=11,
        fontsize=9.0, linespacing=1.35
    )

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(-0.65, len(sub) - 0.35)
    ax.invert_yaxis()
    ax.set_yticks(y_positions)

    if horizon == 24:
        labels = [
            f"{g}   {int(n)}{'†' if int(n) < 8 else ''}"
            for g, n in zip(sub.genotype, sub.n_plants)
        ]
        ax.set_yticklabels(labels)
        ax.tick_params(axis="y", length=0, labelleft=True)
        # Genotypes and plant counts are already given in the tick labels.
        # An additional vertical label is redundant and crowds the page.
    else:
        ax.tick_params(axis="y", left=False, labelleft=False)

    ax.set_xlabel("Empirical coverage (%)", labelpad=7)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(xmax=1, decimals=0)
    )
    # Hide only the two abutting end labels at the central panel boundary.
    # Both panels retain identical limits and internal ticks.
    if horizon == 24:
        tick_labels = [f"{round(100*t):.0f}%" if t < 0.999 else "" for t in ticks]
    else:
        tick_labels = ["" if np.isclose(t, ticks[0]) else f"{round(100*t):.0f}%" for t in ticks]
    ax.set_xticks(ticks, labels=tick_labels)
    ax.grid(axis="x", color=LIGHT_GREY, linewidth=0.55)
    ax.set_axisbelow(True)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(MID_GREY)
    ax.tick_params(axis="x", width=0.6, length=3, color=DARK_GREY)

legend = [
    Line2D([0], [0], marker="D", linestyle="none",
           markerfacecolor=BLUE, markeredgecolor=BLUE,
           markersize=5, label="Genotype estimate ≥90%"),

    Line2D([0], [0], marker="D", linestyle="none",
           markerfacecolor=VERMILLION, markeredgecolor=VERMILLION,
           markersize=5, label="Genotype estimate <90%"),

    Line2D([0], [0], marker="o", linestyle="none",
           markerfacecolor="white", markeredgecolor=DARK_GREY,
           markeredgewidth=0.6, markersize=4,
           label="Paired assignment"),

    Line2D([0], [0], color=DARK_GREY, linestyle=(0, (4, 2)),
           label="Nominal 90%"),

    Line2D([0], [0], color=MID_GREY, linewidth=1.2,
           label="Conditional 95% bootstrap interval"),
]

# Keep the legend below both axis labels. The statistical qualification belongs
# in the journal caption, not as unreadably small text inside the artwork.
fig.subplots_adjust(
    left=0.14,
    right=0.985,
    top=0.86,
    bottom=0.19,
    wspace=0.22,
)
fig.legend(
    handles=legend,
    loc="lower center",
    bbox_to_anchor=(0.5, 0.027),
    ncol=3,
    frameon=False,
    fontsize=7.1,
    handlelength=1.9,
    handletextpad=0.6,
    columnspacing=1.4,
)

stem = OUT / "F4_U5_genotype_reliability"
pdf_metadata = {
    "Title": "U5 empirical coverage across held-out Arabidopsis genotypes",
    "Subject": "Genotype-specific conformal prediction reliability",
    "Creator": "leaf-growth-generalization publication figure pipeline",
}
svg_metadata = {
    "Title": pdf_metadata["Title"],
    "Description": pdf_metadata["Subject"],
    "Creator": pdf_metadata["Creator"],
}

fig.savefig(stem.with_suffix(".pdf"), metadata=pdf_metadata, bbox_inches="tight")
fig.savefig(stem.with_suffix(".svg"), metadata=svg_metadata, bbox_inches="tight")
fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
plt.close(fig)

print(f"Saved publication figure: {stem}.{{pdf,svg,png}}")
