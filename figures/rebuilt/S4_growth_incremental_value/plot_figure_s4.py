
#!/usr/bin/env python3
"""
Supplementary Figure S4 — Incremental Value of Historical Growth

Comparison: SGA-CQR minus SA-CQR.

Inputs:
    experiments/P3/P3_8/P3_8_genotype_differences.csv
    experiments/P3/P3_8/P3_8_direct_comparison_summary.csv

Outputs:
    Figure_S4_Growth_Incremental_Value.pdf
    Figure_S4_Growth_Incremental_Value.png
    Figure_S4_Growth_Incremental_Value.svg
    Figure_S4_source.csv
    Figure_S4_summary.csv

Updates:
    - Fixed legend and panel-label overlap
    - Removed annotations overlapping lower-panel reference lines
    - Improved panel spacing and typography
    - Preserved all scientific data and statistics
    - Added strict numerical checks

No retraining, recalibration or new simulations.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# ============================================================
# 1. PATHS AND SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT_GENOTYPES = (
    ROOT
    / "experiments/P3/P3_8"
    / "P3_8_genotype_differences.csv"
)

INPUT_SUMMARY = (
    ROOT
    / "experiments/P3/P3_8"
    / "P3_8_direct_comparison_summary.csv"
)

OUT = ROOT / "figures/rebuilt/S4_growth_incremental_value"
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = (24, 48)

COLORS = {
    24: "#4C78A8",
    48: "#E68627",
}

REFERENCE = {
    24: -1.8779,
    48: -5.4134,
}

STEM = "Figure_S4_Growth_Incremental_Value"


# ============================================================
# 2. PUBLICATION STYLE
# ============================================================

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


# ============================================================
# 3. LOAD AND VALIDATE DATA
# ============================================================

if not INPUT_GENOTYPES.is_file():
    raise FileNotFoundError(INPUT_GENOTYPES)

if not INPUT_SUMMARY.is_file():
    raise FileNotFoundError(INPUT_SUMMARY)

df = pd.read_csv(INPUT_GENOTYPES)
summary = pd.read_csv(INPUT_SUMMARY)

required_genotype = {
    "horizon",
    "genotype",
    "delta_score",
}

required_summary = {
    "horizon",
    "metric",
    "mean_delta",
    "bootstrap_low",
    "bootstrap_high",
    "n_genotypes",
}

missing = required_genotype - set(df.columns)
if missing:
    raise ValueError(f"Missing genotype columns: {sorted(missing)}")

missing = required_summary - set(summary.columns)
if missing:
    raise ValueError(f"Missing summary columns: {sorted(missing)}")

for col in ["horizon", "delta_score"]:
    df[col] = pd.to_numeric(df[col], errors="raise")

for col in [
    "horizon",
    "mean_delta",
    "bootstrap_low",
    "bootstrap_high",
    "n_genotypes",
]:
    summary[col] = pd.to_numeric(
        summary[col], errors="raise"
    )

df = df[df["horizon"].isin(HORIZONS)].copy()

summary = summary[
    summary["horizon"].isin(HORIZONS)
    & summary["metric"].eq("score")
].copy()

if df.duplicated(["horizon", "genotype"]).any():
    raise ValueError("Duplicate genotype/horizon records")

if summary.duplicated(["horizon", "metric"]).any():
    raise ValueError("Duplicate summary records")

if df["delta_score"].isna().any():
    raise ValueError("Missing genotype score differences")

numeric_cols = [
    "mean_delta",
    "bootstrap_low",
    "bootstrap_high",
]

if summary[numeric_cols].isna().any().any():
    raise ValueError("Missing summary values")

if not np.isfinite(df["delta_score"]).all():
    raise ValueError("Nonfinite genotype differences")

if not np.isfinite(summary[numeric_cols]).all().all():
    raise ValueError("Nonfinite summary statistics")

if len(summary) != 2:
    raise ValueError("Expected exactly two score-summary rows")


# ============================================================
# 4. VERIFY P3.8 RESULTS
# ============================================================

for horizon in HORIZONS:

    group = df[df["horizon"] == horizon]
    row = summary[summary["horizon"] == horizon].iloc[0]

    if len(group) != 21:
        raise ValueError(
            f"{horizon}h: expected 21 genotypes, got {len(group)}"
        )

    if group["genotype"].nunique() != 21:
        raise ValueError("Duplicate or missing genotype")

    if int(row["n_genotypes"]) != 21:
        raise ValueError(
            f"{horizon}h: expected 21 bootstrap units"
        )

    observed_mean = float(group["delta_score"].mean())

    if not np.isclose(
        observed_mean,
        float(row["mean_delta"]),
        atol=1e-7,
        rtol=0,
    ):
        raise ValueError(
            f"{horizon}h mean differs from P3.8 summary"
        )

    if abs(observed_mean - REFERENCE[horizon]) > 0.01:
        raise ValueError(
            f"{horizon}h unexpected reference result"
        )

    if not (
        row["bootstrap_low"]
        <= row["mean_delta"]
        <= row["bootstrap_high"]
    ):
        raise ValueError("Invalid bootstrap interval")

    print(
        f"{horizon}h: mean difference "
        f"{observed_mean:+.4f} px² — PASS"
    )

g24 = set(df[df["horizon"] == 24]["genotype"])
g48 = set(df[df["horizon"] == 48]["genotype"])

if g24 != g48:
    raise ValueError("Genotype sets differ between horizons")


# ============================================================
# 5. CONSISTENT GENOTYPE ORDERING
# ============================================================

# Strongest 48h improvement appears at the bottom,
# maintaining the visual order used in the previous figure.

ORDER = (
    df[df["horizon"] == 48]
    .sort_values(
        ["delta_score", "genotype"],
        ascending=[False, True],
    )["genotype"]
    .tolist()
)

df["genotype"] = pd.Categorical(
    df["genotype"],
    categories=ORDER,
    ordered=True,
)

df = df.sort_values(
    ["horizon", "genotype"]
).reset_index(drop=True)

df["growth_improves_score"] = df["delta_score"] < 0

df.to_csv(
    OUT / "Figure_S4_source.csv",
    index=False,
)

summary.to_csv(
    OUT / "Figure_S4_summary.csv",
    index=False,
)


# ============================================================
# 6. FIGURE LAYOUT
# ============================================================

fig = plt.figure(figsize=(11.8, 9.2))

gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[5.0, 1.25],
    hspace=0.29,
    wspace=0.20,
)

ax24 = fig.add_subplot(gs[0, 0])
ax48 = fig.add_subplot(gs[0, 1], sharey=ax24)

bottom24 = fig.add_subplot(gs[1, 0], sharex=ax24)
bottom48 = fig.add_subplot(gs[1, 1], sharex=ax48)

TOP = {
    24: ax24,
    48: ax48,
}

BOTTOM = {
    24: bottom24,
    48: bottom48,
}


# ============================================================
# 7. TOP PANELS — GENOTYPE-LEVEL DIFFERENCES
# ============================================================

for horizon in HORIZONS:

    ax = TOP[horizon]

    group = (
        df[df["horizon"] == horizon]
        .set_index("genotype")
        .loc[ORDER]
    )

    values = group["delta_score"].to_numpy(dtype=float)
    y = np.arange(len(ORDER))

    # Zero-to-value guide lines.
    for i, value in enumerate(values):
        ax.plot(
            [0, value],
            [i, i],
            color=COLORS[horizon],
            alpha=0.30,
            linewidth=1.1,
            zorder=2,
        )

    improved = values < 0
    worsened = values > 0
    tied = values == 0

    # Filled markers: improved interval score.
    ax.scatter(
        values[improved],
        y[improved],
        s=40,
        marker="o",
        facecolor=COLORS[horizon],
        edgecolor="white",
        linewidth=0.7,
        zorder=4,
    )

    # Hollow markers: worse interval score.
    ax.scatter(
        values[worsened],
        y[worsened],
        s=40,
        marker="o",
        facecolor="white",
        edgecolor=COLORS[horizon],
        linewidth=1.5,
        zorder=4,
    )

    if tied.any():
        ax.scatter(
            values[tied],
            y[tied],
            s=40,
            marker="s",
            color="#333333",
            zorder=4,
        )

    ax.axvline(
        0,
        color="#444444",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
        zorder=1,
    )

    ax.set_yticks(y)

    if horizon == 24:
        ax.set_yticklabels(ORDER)
        ax.set_ylabel("Held-out genotype")
    else:
        ax.tick_params(axis="y", labelleft=False)

    ax.invert_yaxis()

    ax.set_title(
        f"{horizon}-hour forecast",
        pad=11,
        fontweight="medium",
    )

    ax.set_xlabel(
        "SGA-CQR minus SA-CQR interval score (px²)",
        labelpad=8,
    )

    ax.set_axisbelow(True)
    ax.grid(
        axis="x",
        color="#DADADA",
        linewidth=0.65,
        alpha=0.75,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="y", length=0)


# ============================================================
# 8. BOTTOM PANELS — GENOTYPE-MACRO SUMMARIES
# ============================================================

for horizon in HORIZONS:

    ax = BOTTOM[horizon]

    row = summary[
        summary["horizon"] == horizon
    ].iloc[0]

    estimate = float(row["mean_delta"])
    lower = float(row["bootstrap_low"])
    upper = float(row["bootstrap_high"])

    xerr = np.array([
        [estimate - lower],
        [upper - estimate],
    ])

    ax.errorbar(
        estimate,
        0,
        xerr=xerr,
        fmt="D",
        color=COLORS[horizon],
        ecolor=COLORS[horizon],
        markersize=8,
        markeredgecolor="white",
        markeredgewidth=0.75,
        elinewidth=1.9,
        capsize=4,
        capthick=1.3,
        zorder=5,
    )

    ax.axvline(
        0,
        color="#444444",
        linestyle=(0, (4, 3)),
        linewidth=1.0,
    )

    ax.set_ylim(-0.70, 0.70)
    ax.set_yticks([])

    ax.set_title(
        "Genotype-macro mean and 95% bootstrap interval",
        fontsize=9.5,
        pad=10,
    )

    ax.set_xlabel(
        "Mean interval-score difference (px²)",
        labelpad=7,
    )

    ax.grid(
        axis="x",
        color="#DADADA",
        linewidth=0.65,
        alpha=0.75,
    )

    ax.set_axisbelow(True)

    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)

    ax.tick_params(axis="y", length=0)


# ============================================================
# 9. COORDINATED AXIS LIMITS
# ============================================================

for horizon in HORIZONS:

    values = df.loc[
        df["horizon"] == horizon,
        "delta_score",
    ].to_numpy(dtype=float)

    row = summary[
        summary["horizon"] == horizon
    ].iloc[0]

    low = min(
        values.min(),
        float(row["bootstrap_low"]),
        0.0,
    )

    high = max(
        values.max(),
        float(row["bootstrap_high"]),
        0.0,
    )

    padding = max(
        0.75,
        0.10 * (high - low),
    )

    limits = (
        low - padding,
        high + padding,
    )

    TOP[horizon].set_xlim(*limits)
    BOTTOM[horizon].set_xlim(*limits)


# ============================================================
# 10. PANEL LABELS
# ============================================================

# Labels placed within reserved inter-panel whitespace.

for ax, letter in (
    (ax24, "(a)"),
    (ax48, "(b)"),
    (bottom24, "(c)"),
    (bottom48, "(d)"),
):
    ax.text(
        -0.065,
        1.035,
        letter,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


# ============================================================
# 11. SHARED LEGEND
# ============================================================

handles = [
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor=COLORS[24],
        markeredgecolor="white",
        markersize=7,
        label="Lower interval score",
    ),
    Line2D(
        [0], [0],
        marker="o",
        linestyle="none",
        markerfacecolor="white",
        markeredgecolor=COLORS[24],
        markersize=7,
        label="Higher interval score",
    ),
    Line2D(
        [0], [0],
        marker="D",
        linestyle="none",
        markerfacecolor="#444444",
        markeredgecolor="white",
        markersize=7,
        label="Genotype-macro mean",
    ),
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.55, 0.985),
    frameon=False,
    ncol=3,
    columnspacing=2.0,
    handletextpad=0.5,
)

# Leave dedicated vertical room for the legend.
fig.subplots_adjust(
    left=0.13,
    right=0.98,
    top=0.87,
    bottom=0.08,
)


# ============================================================
# 12. EXPORT
# ============================================================

for suffix in ("pdf", "svg"):
    fig.savefig(
        OUT / f"{STEM}.{suffix}",
        bbox_inches="tight",
    )

fig.savefig(
    OUT / f"{STEM}.png",
    dpi=450,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# 13. NUMERICAL REPORT
# ============================================================

print("\n" + "=" * 70)
print("SUPPLEMENTARY FIGURE S4 — SUCCESS")
print("=" * 70)

for horizon in HORIZONS:

    row = summary[
        summary["horizon"] == horizon
    ].iloc[0]

    n_improved = int(
        (
            df.loc[
                df["horizon"] == horizon,
                "delta_score",
            ] < 0
        ).sum()
    )

    print(f"\n{horizon}-HOUR FORECAST")
    print(
        f"Mean score difference: "
        f"{row['mean_delta']:+.4f} px²"
    )
    print(
        f"Bootstrap percentile interval: "
        f"[{row['bootstrap_low']:+.4f}, "
        f"{row['bootstrap_high']:+.4f}] px²"
    )
    print(f"Genotypes improved: {n_improved}/21")

print("\nSaved to:")
print(OUT)

print(
    "\nNegative differences favor SGA-CQR."
)
print(
    "Percentile intervals are retrospective and exploratory."
)
