
#!/usr/bin/env python3
"""
Figure 9 — Adaptive CQR interval-score performance.

Inputs:
  experiments/P3/P3_6/P3_6_genotype_metrics.csv

Outputs:
  figures/rebuilt/F9_adaptive_CQR_performance/
    Figure_F9_Adaptive_CQR_Performance.pdf
    Figure_F9_Adaptive_CQR_Performance.png
    Figure_F9_Adaptive_CQR_Performance.svg
    Figure_F9_source.csv
    Figure_F9_summary.csv

Panels:
(a) 24h genotype-paired score differences
(b) 48h genotype-paired score differences
(c) 24h mean differences with bootstrap intervals
(d) 48h mean differences with bootstrap intervals

Negative score differences favor adaptive calibration.
No retraining, recalibration, or modification of P3 data.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ==========================================================
# 1. PATHS AND SETTINGS
# ==========================================================

ROOT = Path(__file__).resolve().parents[3]

INPUT = (
    ROOT / "experiments/P3/P3_6/"
    "P3_6_genotype_metrics.csv"
)

OUT = (
    ROOT / "figures/rebuilt/"
    "F9_adaptive_CQR_performance"
)

OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261116
BOOTSTRAPS = 20000
rng = np.random.default_rng(SEED)

METHODS = ["SA_CQR", "SGA_CQR"]
HORIZONS = [24, 48]

COLORS = {
    "SA_CQR": "#4C78A8",
    "SGA_CQR": "#E68627",
}

DISPLAY = {
    "SA_CQR": "SA-CQR",
    "SGA_CQR": "SGA-CQR",
}

MARKERS = {
    "SA_CQR": "o",
    "SGA_CQR": "D",
}

# ==========================================================
# 2. PUBLICATION STYLE
# ==========================================================

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.labelsize": 8.5,
    "ytick.labelsize": 8.5,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})

# ==========================================================
# 3. LOAD AND VERIFY DATA
# ==========================================================

if not INPUT.exists():
    raise FileNotFoundError(INPUT)

df = pd.read_csv(INPUT)

required = {
    "horizon_hours",
    "test_genotype",
    "method",
    "score",
    "width",
    "coverage",
}

missing = required - set(df.columns)

if missing:
    raise ValueError(
        f"Missing columns: {sorted(missing)}"
    )

if df.duplicated([
    "horizon_hours",
    "test_genotype",
    "method",
]).any():
    raise ValueError(
        "Duplicate genotype/method/horizon records"
    )

expected_methods = {
    "U5", "SA_CQR", "SGA_CQR"
}

if set(df["method"]) != expected_methods:
    raise ValueError(
        "Expected U5, SA_CQR and SGA_CQR"
    )

if set(df["horizon_hours"]) != {24, 48}:
    raise ValueError(
        "Expected both 24h and 48h horizons"
    )

for col in ["score", "width", "coverage"]:
    df[col] = pd.to_numeric(
        df[col], errors="raise"
    )

if df[["score", "width", "coverage"]].isna().any().any():
    raise ValueError("Invalid numerical data")

if len(df) != 21 * 3 * 2:
    raise ValueError(
        f"Expected 126 genotype-method rows, found {len(df)}"
    )

expected_u5_coverage = {
    24: 0.899712,
    48: 0.9004006,
}

for h in HORIZONS:
    sub = df[df.horizon_hours == h]

    for method in expected_methods:
        if sub[sub.method == method].test_genotype.nunique() != 21:
            raise ValueError(
                f"Incomplete {method} at {h}h"
            )

    observed = sub[
        sub.method == "U5"
    ]["coverage"].mean()

    if abs(observed - expected_u5_coverage[h]) > 0.00015:
        raise ValueError(
            f"U5 reproduction failed at {h}h"
        )

    print(
        f"{h}h U5 macro coverage: "
        f"{observed:.5%} — PASS"
    )

# ==========================================================
# 4. PAIRED GENOTYPE DIFFERENCES
# ==========================================================

source_rows = []
summary_rows = []

genotypes = sorted(
    df[df.horizon_hours == 24].test_genotype.unique()
)

if set(genotypes) != set(
    df[df.horizon_hours == 48].test_genotype.unique()
):
    raise ValueError("Genotype sets differ between horizons")

data_for_plot = {}

for h in HORIZONS:
    sub = df[
        df.horizon_hours == h
    ]

    wide = sub.pivot(
        index="test_genotype",
        columns="method",
        values=["score", "width", "coverage"],
    ).loc[genotypes]

    if wide.isna().any().any():
        raise ValueError("Incomplete paired data")

    differences = pd.DataFrame(
        index=genotypes
    )

    # Same bootstrap genotype indices for both methods,
    # preserving their paired comparison structure.
    n = len(genotypes)
    boot_indices = rng.integers(
        0, n, size=(BOOTSTRAPS, n)
    )

    for method in METHODS:
        delta_score = (
            wide[("score", method)]
            - wide[("score", "U5")]
        ).to_numpy(dtype=float)

        delta_width = (
            wide[("width", method)]
            - wide[("width", "U5")]
        ).to_numpy(dtype=float)

        delta_coverage = (
            wide[("coverage", method)]
            - wide[("coverage", "U5")]
        ).to_numpy(dtype=float)

        differences[method] = delta_score

        boot = delta_score[
            boot_indices
        ].mean(axis=1)

        ci_low, ci_high = np.quantile(
            boot, [0.025, 0.975]
        )

        summary_rows.append({
            "horizon_hours": h,
            "method": method,
            "comparison": f"{method} minus U5",
            "genotypes": n,
            "mean_score_difference": delta_score.mean(),
            "median_score_difference": np.median(delta_score),
            "bootstrap_low": ci_low,
            "bootstrap_high": ci_high,
            "genotypes_improved": int(
                (delta_score < 0).sum()
            ),
            "mean_width_difference": delta_width.mean(),
            "mean_coverage_difference_pp": (
                100 * delta_coverage.mean()
            ),
        })

        for i, g in enumerate(genotypes):
            source_rows.append({
                "horizon_hours": h,
                "genotype": g,
                "method": method,
                "U5_score": wide.loc[g, ("score", "U5")],
                "adaptive_score": wide.loc[
                    g, ("score", method)
                ],
                "delta_score": delta_score[i],
                "delta_width": delta_width[i],
                "delta_coverage_pp": (
                    100 * delta_coverage[i]
                ),
            })

    data_for_plot[h] = differences

source = pd.DataFrame(source_rows)
summary = pd.DataFrame(summary_rows)

source.to_csv(
    OUT / "Figure_F9_source.csv",
    index=False
)

summary.to_csv(
    OUT / "Figure_F9_summary.csv",
    index=False
)

# Check consistency with existing P3.7 experiment.
expected_differences = {
    (24, "SA_CQR"): -7.171,
    (24, "SGA_CQR"): -9.049,
    (48, "SA_CQR"): -18.720,
    (48, "SGA_CQR"): -24.133,
}

for row in summary.itertuples():
    target = expected_differences[
        (row.horizon_hours, row.method)
    ]

    if abs(row.mean_score_difference - target) > 0.05:
        raise ValueError(
            "P3.7 score comparison mismatch: "
            f"{row.horizon_hours}h {row.method}"
        )

print("P3.7 comparison consistency: PASS")

# ==========================================================
# 5. FIGURE LAYOUT
# ==========================================================

fig = plt.figure(
    figsize=(12.8, 10.0)
)

gs = fig.add_gridspec(
    2,
    2,
    height_ratios=[4.8, 1.2],
    hspace=0.22,
    wspace=0.27,
)

ax_a = fig.add_subplot(gs[0, 0])
ax_b = fig.add_subplot(gs[0, 1])
ax_c = fig.add_subplot(gs[1, 0], sharex=ax_a)
ax_d = fig.add_subplot(gs[1, 1], sharex=ax_b)

top_axes = {
    24: ax_a,
    48: ax_b,
}

bottom_axes = {
    24: ax_c,
    48: ax_d,
}

Y = np.arange(len(genotypes))

OFFSETS = {
    "SA_CQR": -0.15,
    "SGA_CQR": +0.15,
}

# ==========================================================
# 6. TOP PANELS: GENOTYPE DIFFERENCES
# ==========================================================

for h in HORIZONS:
    ax = top_axes[h]
    data = data_for_plot[h]

    sa = data["SA_CQR"].to_numpy()
    sga = data["SGA_CQR"].to_numpy()

    # Thin connectors show the incremental effect
    # of adding historical growth to size adaptation.
    for i in range(len(genotypes)):
        ax.plot(
            [sa[i], sga[i]],
            [Y[i], Y[i]],
            color="#B8B8B8",
            linewidth=1.0,
            zorder=2,
        )

    for method in METHODS:
        ax.scatter(
            data[method],
            Y + OFFSETS[method],
            s=32,
            color=COLORS[method],
            marker=MARKERS[method],
            edgecolors="white",
            linewidths=0.55,
            zorder=4,
        )

    ax.axvline(
        0,
        color="#333333",
        linestyle=(0, (4, 3)),
        linewidth=1.15,
        zorder=1,
    )

    ax.set_yticks(Y)

    if h == 24:
        ax.set_yticklabels(genotypes)
    else:
        ax.tick_params(
            axis="y",
            labelleft=False
        )

    ax.invert_yaxis()

    ax.set_title(
        f"{h}-hour forecast",
        pad=12,
    )

    ax.set_axisbelow(True)

    ax.grid(
        axis="x",
        color="#DADADA",
        linewidth=0.65,
        alpha=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0,
    )

    ax.set_xlabel(
        "Interval-score difference from U5 (px²)",
        labelpad=9,
    )

# ==========================================================
# 7. BOTTOM PANELS: BOOTSTRAP SUMMARY
# ==========================================================

for h in HORIZONS:
    ax = bottom_axes[h]

    rows = summary[
        summary.horizon_hours == h
    ].set_index("method")

    for j, method in enumerate(METHODS):
        row = rows.loc[method]

        estimate = float(
            row.mean_score_difference
        )

        ci_low = float(
            row.bootstrap_low
        )
        ci_high = float(
            row.bootstrap_high
        )

        errors = np.array([
            [max(0, estimate - ci_low)],
            [max(0, ci_high - estimate)],
        ])

        ax.errorbar(
            estimate,
            j,
            xerr=errors,
            fmt=MARKERS[method],
            color=COLORS[method],
            ecolor=COLORS[method],
            markersize=6.5,
            markeredgecolor="white",
            markeredgewidth=0.5,
            elinewidth=1.8,
            capsize=3.5,
            capthick=1.2,
            zorder=4,
        )

    ax.axvline(
        0,
        color="#333333",
        linestyle=(0, (4, 3)),
        linewidth=1.15,
    )

    ax.set_yticks(
        [0, 1]
    )

    ax.set_yticklabels(
        ["SA-CQR", "SGA-CQR"]
    )

    ax.set_ylim(
        1.6, -0.6
    )

    ax.set_title(
        "Genotype-macro paired difference",
        fontsize=10,
        pad=9,
    )

    ax.set_xlabel(
        "Mean interval-score difference (px²)"
    )

    ax.set_axisbelow(True)

    ax.grid(
        axis="x",
        color="#DADADA",
        linewidth=0.65,
        alpha=0.8,
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)

    ax.tick_params(
        axis="y",
        length=0,
    )

# ==========================================================
# 8. COORDINATED AXES
# ==========================================================

for h in HORIZONS:
    values = source[
        source.horizon_hours == h
    ]["delta_score"]

    intervals = summary[
        summary.horizon_hours == h
    ]

    left_limit = min(
        values.min(),
        intervals.bootstrap_low.min(),
        0.0,
    )

    right_limit = max(
        values.max(),
        intervals.bootstrap_high.max(),
        0.0,
    )

    padding = max(
        3.0,
        (right_limit - left_limit) * 0.09
    )

    limits = (
        left_limit - padding,
        right_limit + padding,
    )

    top_axes[h].set_xlim(*limits)
    bottom_axes[h].set_xlim(*limits)

# Panel labels
for ax, label in zip(
    [ax_a, ax_b, ax_c, ax_d],
    ["(a)", "(b)", "(c)", "(d)"]
):
    ax.text(
        -0.075,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=12,
        fontweight="bold",
        ha="left",
        va="bottom",
    )

# Compact shared legend
handles = [
    Line2D(
        [0], [0],
        marker=MARKERS[m],
        linestyle="none",
        markerfacecolor=COLORS[m],
        markeredgecolor="white",
        markersize=7.5,
        label=DISPLAY[m],
    )
    for m in METHODS
]

fig.legend(
    handles=handles,
    loc="upper center",
    bbox_to_anchor=(0.5, 0.995),
    ncol=2,
    frameon=False,
    columnspacing=2.5,
)

fig.subplots_adjust(
    left=0.145,
    right=0.975,
    top=0.88,
    bottom=0.075,
)

# ==========================================================
# 9. EXPORT
# ==========================================================

stem = "Figure_F9_Adaptive_CQR_Performance"

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

print("\n" + "=" * 72)
print("FIGURE 9 — SUCCESS")
print("=" * 72)

print(
    summary[
        [
            "horizon_hours",
            "method",
            "mean_score_difference",
            "bootstrap_low",
            "bootstrap_high",
            "genotypes_improved",
            "mean_width_difference",
            "mean_coverage_difference_pp",
        ]
    ].to_string(index=False)
)

print("\nFiles saved in:")
print(OUT)

print(
    "\nInterpretation: negative interval-score "
    "differences favor adaptive CQR."
)

print(
    "Bootstrap intervals are retrospective and exploratory."
)
