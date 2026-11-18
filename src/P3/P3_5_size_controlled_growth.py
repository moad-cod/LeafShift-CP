
#!/usr/bin/env python3
"""
P3.5 — Size-controlled historical growth and U5 miscoverage.

Within genotype:
  1. Divide forecast windows into 3 historical leaf-size strata.
  2. Inside each stratum, identify low/high historical growth.
  3. Pair the same plants across low/high groups.
  4. Average assignments, then eligible plant contrasts.
  5. Average strata within genotype, then genotypes equally.

Exploratory analysis. No training or recalibration.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
PRED = ROOT / "experiments/P1/P1_1/tables"
MANIFEST = ROOT / "data/processed/forecasting/forecast_windows.csv"
OUT = ROOT / "experiments/P3/P3_5"
OUT.mkdir(parents=True, exist_ok=True)

MIN_WINDOWS = 5
FEATURES = {
    "absolute_growth": "history_growth_abs_px_per_hour",
    "relative_growth": "history_growth_relative_per_hour",
}
METRICS = ["covered", "upper_miss", "lower_miss"]

# 1. Load archived U5 predictions.
frames = []
for h in [24, 48]:
    f = PRED / f"P1_1_U5_window_predictions_{h}h.csv"
    d = pd.read_csv(f)
    if set(d.horizon_hours.unique()) != {h}:
        raise ValueError(f"Unexpected horizon in {f}")
    frames.append(d)

df = pd.concat(frames, ignore_index=True)

key = ["horizon_hours", "window_id"]
prediction_key = key + ["inner_fold"]

if df.duplicated(prediction_key).any():
    raise ValueError("Duplicate window/assignment predictions")

if not df.groupby(key)["inner_fold"].nunique().eq(4).all():
    raise ValueError("Expected four assignments per window")

for col in ["target_area_pixels", "lower_pixels", "upper_pixels"]:
    df[col] = pd.to_numeric(df[col], errors="raise")

if df[
    ["target_area_pixels", "lower_pixels", "upper_pixels"]
].isna().any().any():
    raise ValueError("Invalid numerical prediction data")

if (df.lower_pixels > df.upper_pixels).any():
    raise ValueError("Invalid interval endpoints")

df["covered"] = (
    (df.target_area_pixels >= df.lower_pixels) &
    (df.target_area_pixels <= df.upper_pixels)
).astype(int)

df["upper_miss"] = (
    df.target_area_pixels > df.upper_pixels
).astype(int)

df["lower_miss"] = (
    df.target_area_pixels < df.lower_pixels
).astype(int)

# 2. Add cutoff leaf size from the archived manifest.
manifest = pd.read_csv(
    MANIFEST,
    usecols=[
        "horizon_hours",
        "window_id",
        "genotype",
        "plant_id",
        "last_observed_area_pixels",
    ],
).rename(columns={
    "genotype": "manifest_genotype",
    "plant_id": "manifest_plant",
})

if manifest.duplicated(key).any():
    raise ValueError("Duplicate manifest keys")

df = df.merge(
    manifest,
    on=key,
    how="left",
    validate="many_to_one",
    indicator=True,
)

if not df["_merge"].eq("both").all():
    raise ValueError("Unmatched prediction windows")

if not df.test_genotype.astype(str).equals(
    df.manifest_genotype.astype(str)
):
    raise ValueError("Genotype identity mismatch")

if not df.plant_id.astype(str).equals(
    df.manifest_plant.astype(str)
):
    raise ValueError("Plant identity mismatch")

df = df.drop(columns=[
    "_merge", "manifest_genotype", "manifest_plant"
])

if df["last_observed_area_pixels"].isna().any():
    raise ValueError("Missing historical leaf sizes")

# Original macro-coverage verification.
pa = df.groupby(
    ["horizon_hours", "test_genotype",
     "plant_id", "inner_fold"]
)["covered"].mean().reset_index()

plants = pa.groupby(
    ["horizon_hours", "test_genotype", "plant_id"]
)["covered"].mean().reset_index()

gen = plants.groupby(
    ["horizon_hours", "test_genotype"]
)["covered"].mean().reset_index()

for h, expected in [(24, .899712), (48, .9004006)]:
    actual = gen.loc[
        gen.horizon_hours == h, "covered"
    ].mean()

    if abs(actual - expected) > 0.00015:
        raise ValueError("Failed original coverage check")

    print(f"Original {h}h coverage: {actual:.5%} PASS")

# 3. Assign size and growth strata using unique
# historical windows only, not repeated assignments.
unique = df.drop_duplicates(key).copy()

label_frames = []
support = []

for feature_name, feature_col in FEATURES.items():

    labels = []

    for (h, g), block in unique.groupby(
        ["horizon_hours", "test_genotype"], sort=True
    ):
        b = block.copy()

        size = pd.to_numeric(
            b["last_observed_area_pixels"], errors="coerce"
        )
        growth = pd.to_numeric(
            b[feature_col], errors="coerce"
        )

        valid = size.notna() & growth.notna()
        b = b.loc[valid].copy()
        size = size.loc[valid]

        if len(b) < 30:
            continue

        # Quantile strata based on historical area.
        # Rank only resolves tied cutoff areas consistently.
        size_rank = size.rank(method="average", pct=True)
        b["size_stratum"] = np.minimum(
            np.floor(size_rank.to_numpy() * 3).astype(int), 2
        )

        for size_bin, part in b.groupby("size_stratum"):
            values = pd.to_numeric(
                part[feature_col], errors="coerce"
            )

            if len(part) < 15:
                continue

            q1, q2 = values.quantile([1/3, 2/3])

            if not np.isfinite(q1) or not q1 < q2:
                continue

            part = part.copy()
            part["growth_group"] = np.select(
                [values <= q1, values >= q2],
                ["low", "high"],
                default="middle",
            )

            part = part[
                part.growth_group.isin(["low", "high"])
            ]

            part["feature"] = feature_name
            labels.append(part[
                key + [
                    "feature", "size_stratum",
                    "growth_group",
                ]
            ])

    if labels:
        label_frames.append(pd.concat(
            labels, ignore_index=True
        ))

if not label_frames:
    raise RuntimeError("No valid historical-feature strata")

labels = pd.concat(label_frames, ignore_index=True)

# 4. Join labels with predictions and compare
# the same plant's low- and high-growth windows.
all_pairs = []

for feature_name in FEATURES:

    current_labels = labels[
        labels.feature == feature_name
    ].drop(columns="feature")

    current = df.merge(
        current_labels,
        on=key,
        how="inner",
        validate="many_to_one",
    )

    group_cols = [
        "horizon_hours", "test_genotype", "plant_id",
        "size_stratum", "growth_group", "inner_fold",
    ]

    counts = current.groupby(group_cols).size()
    eligible = counts[
        counts >= MIN_WINDOWS
    ].index

    per_assignment = (
        current.groupby(group_cols)[METRICS]
        .mean()
        .loc[eligible]
        .reset_index()
    )

    complete_cols = group_cols[:-1]
    complete = (
        per_assignment.groupby(complete_cols)
        ["inner_fold"].nunique()
    )
    complete = complete[
        complete == 4
    ].index

    per_assignment = (
        per_assignment.set_index(complete_cols)
        .loc[complete]
        .reset_index()
    )

    # Each plant/size/group: mean of 4 assignments.
    plant_group = (
        per_assignment.groupby(complete_cols)[METRICS]
        .mean().reset_index()
    )

    # Only include plants with BOTH growth groups
    # within the same genotype and leaf-size stratum.
    wide = plant_group.pivot(
        index=[
            "horizon_hours", "test_genotype",
            "plant_id", "size_stratum",
        ],
        columns="growth_group",
        values=METRICS,
    )

    wide.columns = [
        f"{metric}_{group}"
        for metric, group in wide.columns
    ]
    wide = wide.dropna().reset_index()

    if wide.empty:
        continue

    for m in METRICS:
        wide[f"{m}_high_minus_low"] = (
            wide[f"{m}_high"] - wide[f"{m}_low"]
        )

    wide["feature"] = feature_name
    all_pairs.append(wide)

if not all_pairs:
    raise RuntimeError("No eligible within-size paired plants")

pairs = pd.concat(all_pairs, ignore_index=True)
pairs.to_csv(
    OUT / "P3_5_paired_plant_size.csv", index=False
)

# 5. Equal-weight eligible plants within each
# size stratum, then size strata within genotype.
difference_cols = [
    f"{m}_high_minus_low" for m in METRICS
]

size_results = (
    pairs.groupby([
        "feature", "horizon_hours", "test_genotype",
        "size_stratum",
    ])
    .agg(
        paired_plants=("plant_id", "nunique"),
        **{m: (m, "mean") for m in difference_cols},
    )
    .reset_index()
)

size_results.to_csv(
    OUT / "P3_5_size_stratum_results.csv",
    index=False,
)

genotype_results = (
    size_results.groupby(
        ["feature", "horizon_hours", "test_genotype"]
    )
    .agg(
        eligible_size_strata=("size_stratum", "nunique"),
        paired_plant_stratum_units=("paired_plants", "sum"),
        **{m: (m, "mean") for m in difference_cols},
    )
    .reset_index()
)

genotype_results.to_csv(
    OUT / "P3_5_genotype_results.csv",
    index=False,
)

# 6. Compare size-controlled findings to the
# earlier unadjusted P3.3 genotype contrasts.
previous = pd.read_csv(
    ROOT / "experiments/P3/P3_3/"
    "P3_3_genotype_contrasts.csv"
)

rows = []

for (feature, h), sub in genotype_results.groupby(
    ["feature", "horizon_hours"]
):
    old = previous[
        (previous.feature == feature) &
        (previous.horizon_hours == h)
    ]

    paired = sub.merge(
        old[
            ["test_genotype",
             "upper_miss_high_minus_low"]
        ].rename(columns={
            "upper_miss_high_minus_low": "previous_upper"
        }),
        on="test_genotype",
        validate="one_to_one",
    )

    rows.append({
        "feature": feature,
        "horizon": h,
        "eligible_genotypes": len(sub),
        "positive_genotypes": int(
            (sub.upper_miss_high_minus_low > 0).sum()
        ),
        "size_controlled_upper_pp": (
            sub.upper_miss_high_minus_low.mean() * 100
        ),
        "previous_upper_same_genotypes_pp": (
            paired.previous_upper.mean() * 100
        ),
        "size_controlled_lower_pp": (
            sub.lower_miss_high_minus_low.mean() * 100
        ),
        "size_controlled_coverage_pp": (
            sub.covered_high_minus_low.mean() * 100
        ),
    })

summary = pd.DataFrame(rows)
summary.to_csv(
    OUT / "P3_5_summary.csv", index=False
)

# 7. One figure per horizon.
for h in (24, 48):
    current = genotype_results[
        (genotype_results.horizon_hours == h) &
        (genotype_results.feature == "absolute_growth")
    ].sort_values("upper_miss_high_minus_low")

    if current.empty:
        continue

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(
        np.arange(len(current)),
        current.upper_miss_high_minus_low * 100,
    )
    ax.set_yticks(
        np.arange(len(current)),
        [
            f"{g} ({int(n)} strata)"
            for g, n in zip(
                current.test_genotype,
                current.eligible_size_strata,
            )
        ],
    )
    ax.axvline(0, color="black", linestyle="--")
    ax.set_xlabel(
        "Within-size high minus low growth\n"
        "Upper-tail miss rate (percentage points)"
    )
    ax.set_title(
        f"P3.5 — Size-controlled growth association ({h}h)"
    )
    ax.grid(axis="x", alpha=.15)
    fig.tight_layout()

    fig.savefig(
        OUT / f"P3_5_size_controlled_{h}h.png",
        dpi=250, bbox_inches="tight"
    )
    fig.savefig(
        OUT / f"P3_5_size_controlled_{h}h.pdf",
        bbox_inches="tight"
    )
    plt.close(fig)

# 8. Plain-text interpretation report.
report = [
    "# P3.5 — Size-Controlled Historical Growth",
    "",
    "Exploratory retrospective, within-genotype comparison.",
    "Historical cutoff area defines size strata.",
    "Growth tertiles are defined within each size stratum.",
    "Only paired plants with sufficient windows are retained.",
    "",
]

for _, row in summary.iterrows():
    report += [
        f"## {row.feature} — {int(row.horizon)}h",
        f"- Eligible genotypes: {int(row.eligible_genotypes)}",
        f"- Genotypes with positive upper contrast: "
        f"{int(row.positive_genotypes)}",
        f"- Size-controlled upper-tail contrast: "
        f"{row.size_controlled_upper_pp:+.3f} pp",
        f"- Previous unadjusted upper contrast "
        f"(same eligible genotypes): "
        f"{row.previous_upper_same_genotypes_pp:+.3f} pp",
        f"- Size-controlled lower-tail contrast: "
        f"{row.size_controlled_lower_pp:+.3f} pp",
        f"- Size-controlled coverage contrast: "
        f"{row.size_controlled_coverage_pp:+.3f} pp",
        "",
    ]

report += [
    "## Limitations",
    "",
    "- Size stratification reduces, but does not remove,",
    "  confounding by historical leaf area.",
    "- Historical cutoff features were used for grouping.",
    "- Size bins and growth groups are retrospective.",
    "- Eligible genotype/plant sets differ across analyses.",
    "- Equal weighting of eligible strata defines a new",
    "  exploratory estimand, not the original macro coverage.",
    "- No inference, causal effect, or coverage guarantee",
    "  is established by this experiment.",
]

text = "\n".join(report) + "\n"
(OUT / "P3_5_results.md").write_text(
    text, encoding="utf-8"
)

print("\n" + text)
print("Results saved to:", OUT)