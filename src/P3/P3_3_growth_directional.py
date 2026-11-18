
#!/usr/bin/env python3
"""
P3.3 — Historical growth and directional U5 reliability.

Research design:
- Uses archived U5 test predictions.
- Joins forecast-time metadata from the original manifest.
- Defines low/high historical-feature groups within genotype.
- Compares directional failures on the SAME plants.
- Preserves the four paired development assignments.
- Does not retrain, recalibrate, or modify earlier results.

Retrospective exploratory analysis, not causal inference.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

PRED = ROOT / "experiments/P1/P1_1/tables"
MANIFEST = (
    ROOT / "data/processed/forecasting/forecast_windows.csv"
)
OUT = ROOT / "experiments/P3/P3_3"
OUT.mkdir(parents=True, exist_ok=True)

MIN_WINDOWS_PER_GROUP = 5

FEATURES = {
    "absolute_growth": "history_growth_abs_px_per_hour",
    "relative_growth": "history_growth_relative_per_hour",
    "cutoff_area": "last_observed_area_pixels",
    "history_gap": "max_history_observation_gap_hours",
}

METRICS = ["covered", "lower_miss", "upper_miss"]

# ---------------------------------------------------------
# 1. Load archived predictions
# ---------------------------------------------------------

frames = []

for horizon in (24, 48):
    path = PRED / (
        f"P1_1_U5_window_predictions_{horizon}h.csv"
    )

    df = pd.read_csv(path)
    df["horizon_hours"] = horizon
    frames.append(df)

data = pd.concat(frames, ignore_index=True)

required = [
    "horizon_hours",
    "test_genotype",
    "plant_id",
    "inner_fold",
    "window_id",
    "target_area_pixels",
    "lower_pixels",
    "upper_pixels",
    "covered",
]

missing = set(required) - set(data.columns)

if missing:
    raise ValueError(f"Missing prediction columns: {missing}")

keys = ["horizon_hours", "window_id"]

if data.duplicated(keys + ["inner_fold"]).any():
    raise ValueError("Duplicate prediction window/assignment")

for c in ("target_area_pixels", "lower_pixels", "upper_pixels"):
    data[c] = pd.to_numeric(data[c], errors="raise")

if data[
    ["target_area_pixels", "lower_pixels", "upper_pixels"]
].isna().any().any():
    raise ValueError("Missing prediction values")

if (data.lower_pixels > data.upper_pixels).any():
    raise ValueError("Invalid interval endpoints")

data["lower_miss"] = (
    data.target_area_pixels < data.lower_pixels
).astype(int)

data["upper_miss"] = (
    data.target_area_pixels > data.upper_pixels
).astype(int)

recalculated = (
    (data.target_area_pixels >= data.lower_pixels)
    & (data.target_area_pixels <= data.upper_pixels)
).astype(int)

if not np.array_equal(
    recalculated.to_numpy(),
    data.covered.astype(int).to_numpy()
):
    raise ValueError("Archived and recalculated coverage differ")

data["covered"] = recalculated

# Validate four assignments per forecasting window.
counts = data.groupby(keys)["inner_fold"].nunique()

if not counts.eq(4).all():
    raise ValueError("Expected four assignments per window")

# ---------------------------------------------------------
# 2. Join historical data
# ---------------------------------------------------------

manifest_columns = [
    "window_id",
    "horizon_hours",
    "genotype",
    "plant_id",
    "last_observed_area_pixels",
    "max_history_observation_gap_hours",
    "history_skipped_image_frames",
]

manifest = pd.read_csv(
    MANIFEST,
    usecols=manifest_columns,
)

if manifest.duplicated(keys).any():
    raise ValueError("Duplicate manifest window IDs")

data = data.merge(
    manifest,
    on=keys,
    how="left",
    validate="many_to_one",
    suffixes=("", "_manifest"),
    indicator=True,
)

if not data["_merge"].eq("both").all():
    raise ValueError("Unmatched windows in manifest")

data = data.drop(columns="_merge")

if not data.test_genotype.astype(str).equals(
    data.genotype.astype(str)
):
    raise ValueError("Genotype identity mismatch")

if not data.plant_id.astype(str).equals(
    data.plant_id_manifest.astype(str)
):
    raise ValueError("Plant identity mismatch")

for feature in FEATURES.values():
    if feature not in data.columns:
        raise ValueError(f"Missing feature: {feature}")

    data[feature] = pd.to_numeric(
        data[feature],
        errors="coerce",
    )

# Reproduce the original overall macro coverage.
pa = (
    data.groupby(
        [
            "horizon_hours",
            "test_genotype",
            "plant_id",
            "inner_fold",
        ]
    )["covered"]
    .mean()
    .reset_index()
)

plant_original = (
    pa.groupby(
        ["horizon_hours", "test_genotype", "plant_id"]
    )["covered"]
    .mean()
    .reset_index()
)

gen_original = (
    plant_original.groupby(
        ["horizon_hours", "test_genotype"]
    )["covered"]
    .mean()
    .reset_index()
)

for h, expected in [(24, 0.899712), (48, 0.9004006)]:
    observed = gen_original.loc[
        gen_original.horizon_hours == h,
        "covered",
    ].mean()

    if abs(observed - expected) > 0.00015:
        raise ValueError("Original U5 reproduction failed")

    print(f"{h}h original macro coverage: {observed:.5%} PASS")

# ---------------------------------------------------------
# 3. Build cutoff-feature groups without target outcomes
# ---------------------------------------------------------

all_pairs = []
support_rows = []

for feature_name, feature_col in FEATURES.items():

    # One row per unique forecasting window.
    unique = data[
        [
            "horizon_hours",
            "test_genotype",
            "window_id",
            feature_col,
        ]
    ].drop_duplicates(
        ["horizon_hours", "test_genotype", "window_id"]
    ).copy()

    labels = []

    for (h, g), group in unique.groupby(
        ["horizon_hours", "test_genotype"]
    ):
        valid = group[feature_col].dropna()

        if len(valid) < 10:
            continue

        q_low, q_high = valid.quantile([1/3, 2/3])

        # Avoid overlapping groups when a feature is constant.
        if not q_low < q_high:
            continue

        current = group.copy()
        current["growth_group"] = np.select(
            [
                current[feature_col] <= q_low,
                current[feature_col] >= q_high,
            ],
            ["low", "high"],
            default="middle",
        )

        current["q_low"] = q_low
        current["q_high"] = q_high

        labels.append(current)

    if not labels:
        print(f"Skipping {feature_name}: insufficient variation")
        continue

    labels = pd.concat(labels, ignore_index=True)

    subset = data.merge(
        labels[
            [
                "horizon_hours",
                "test_genotype",
                "window_id",
                "growth_group",
            ]
        ],
        on=[
            "horizon_hours",
            "test_genotype",
            "window_id",
        ],
        how="inner",
        validate="many_to_one",
    )

    subset = subset[
        subset.growth_group.isin(["low", "high"])
    ].copy()

    # First average within plant, group and assignment.
    per_assignment = (
        subset.groupby(
            [
                "horizon_hours",
                "test_genotype",
                "plant_id",
                "growth_group",
                "inner_fold",
            ]
        )[METRICS]
        .agg(["mean", "size"])
    )

    per_assignment.columns = [
        f"{a}_{b}" for a, b in per_assignment.columns
    ]

    per_assignment = per_assignment.reset_index()

    # Require enough unique forecasting windows per group.
    # The four assignments share the same test windows.
    valid_pa = per_assignment[
        per_assignment["covered_size"]
        >= MIN_WINDOWS_PER_GROUP
    ].copy()

    # Include only complete four-assignment plant/group units.
    complete = (
        valid_pa.groupby(
            [
                "horizon_hours",
                "test_genotype",
                "plant_id",
                "growth_group",
            ]
        )["inner_fold"]
        .nunique()
        .reset_index(name="n_assignments")
    )

    complete = complete[
        complete.n_assignments == 4
    ].drop(columns="n_assignments")

    valid_pa = valid_pa.merge(
        complete,
        on=[
            "horizon_hours",
            "test_genotype",
            "plant_id",
            "growth_group",
        ],
        how="inner",
    )

    plant_group = (
        valid_pa.groupby(
            [
                "horizon_hours",
                "test_genotype",
                "plant_id",
                "growth_group",
            ]
        )[
            [f"{m}_mean" for m in METRICS]
        ]
        .mean()
        .reset_index()
    )

    # Pair the same plant's low and high outcomes.
    wide = plant_group.pivot(
        index=[
            "horizon_hours",
            "test_genotype",
            "plant_id",
        ],
        columns="growth_group",
        values=[f"{m}_mean" for m in METRICS],
    )

    wide.columns = [
        f"{metric}_{group}"
        for metric, group in wide.columns
    ]

    wide = wide.dropna().reset_index()

    for metric in METRICS:
        wide[f"{metric}_high_minus_low"] = (
            wide[f"{metric}_mean_high"]
            - wide[f"{metric}_mean_low"]
        )

    wide["feature"] = feature_name
    all_pairs.append(wide)

    for (h, g), group in wide.groupby(
        ["horizon_hours", "test_genotype"]
    ):
        support_rows.append({
            "feature": feature_name,
            "horizon": h,
            "genotype": g,
            "paired_plants": len(group),
        })

if not all_pairs:
    raise RuntimeError("No eligible paired growth comparisons")

pairs = pd.concat(all_pairs, ignore_index=True)
pairs.to_csv(
    OUT / "P3_3_paired_plant_results.csv",
    index=False,
)

support = pd.DataFrame(support_rows)
support.to_csv(
    OUT / "P3_3_support.csv",
    index=False,
)

# ---------------------------------------------------------
# 4. Genotype-level paired contrasts
# ---------------------------------------------------------

difference_cols = [
    f"{m}_high_minus_low" for m in METRICS
]

genotype = (
    pairs.groupby(
        ["feature", "horizon_hours", "test_genotype"]
    )
    .agg(
        paired_plants=("plant_id", "nunique"),
        **{
            col: (col, "mean")
            for col in difference_cols
        },
    )
    .reset_index()
)

genotype.to_csv(
    OUT / "P3_3_genotype_contrasts.csv",
    index=False,
)

summary_rows = []

for (feature, h), sub in genotype.groupby(
    ["feature", "horizon_hours"]
):
    row = {
        "feature": feature,
        "horizon": h,
        "eligible_genotypes": len(sub),
        "paired_plants_total": int(
            sub.paired_plants.sum()
        ),
    }

    for col in difference_cols:
        row[col] = float(sub[col].mean())
        row[f"{col}_median"] = float(sub[col].median())

    summary_rows.append(row)

summary = pd.DataFrame(summary_rows)

summary.to_csv(
    OUT / "P3_3_summary.csv",
    index=False,
)

# ---------------------------------------------------------
# 5. Publication figures
# ---------------------------------------------------------

for h in (24, 48):
    sub = genotype[
        (genotype.horizon_hours == h)
        & (genotype.feature == "absolute_growth")
    ].sort_values(
        "upper_miss_high_minus_low"
    )

    if sub.empty:
        continue

    fig, ax = plt.subplots(figsize=(9, 8))

    y = np.arange(len(sub))
    values = sub.upper_miss_high_minus_low * 100

    ax.barh(y, values)

    ax.set_yticks(
        y,
        [
            f"{g} (n={n})"
            for g, n in zip(
                sub.test_genotype,
                sub.paired_plants,
            )
        ],
    )

    ax.axvline(
        0,
        linestyle="--",
        color="black",
        linewidth=1,
    )

    ax.set_xlabel(
        "High minus low historical growth\n"
        "Upper-tail miss rate (percentage points)"
    )

    ax.set_title(
        f"P3.3 — U5 growth-associated upper-tail misses ({h}h)"
    )

    ax.grid(axis="x", alpha=0.15)

    fig.tight_layout()

    for extension in ("png", "pdf"):
        fig.savefig(
            OUT / f"P3_3_upper_growth_{h}h.{extension}",
            dpi=250,
            bbox_inches="tight",
        )

    plt.close(fig)

# ---------------------------------------------------------
# 6. Report
# ---------------------------------------------------------

report = [
    "# P3.3 — Historical Growth and Directional Reliability",
    "",
    "Retrospective exploratory analysis.",
    "",
    "High/low groups use within-genotype feature tertiles.",
    "Thresholds use historical features, not forecast outcomes.",
    "",
    "Each comparison pairs low/high windows from the same",
    "plant, with four development assignments averaged.",
    "",
    "Genotype summaries are equally weighted across eligible",
    "genotypes; they are not guaranteed population effects.",
    "",
]

for _, row in summary.iterrows():

    report += [
        f"## {row['feature']} — {int(row['horizon'])}h",
        "",
        f"- Eligible genotypes: {int(row['eligible_genotypes'])}",
        f"- Paired plants: {int(row['paired_plants_total'])}",
        "- High minus low coverage: "
        f"{100*row['covered_high_minus_low']:+.3f} pp",
        "- High minus low upper misses: "
        f"{100*row['upper_miss_high_minus_low']:+.3f} pp",
        "- High minus low lower misses: "
        f"{100*row['lower_miss_high_minus_low']:+.3f} pp",
        "",
    ]

report += [
    "## Scientific limitations",
    "",
    "- Groups are exploratory and defined within each genotype.",
    "- Historical growth does not identify physiological stage.",
    "- Directional failures do not establish their cause.",
    "- Results are conditional on eligible paired plants.",
    "- Overlapping windows are not independent replicates.",
    "- No new calibration or finite-sample guarantee is claimed.",
    "- No inferential p-values are computed in this stage.",
]

text = "\n".join(report) + "\n"

(OUT / "P3_3_results.md").write_text(
    text,
    encoding="utf-8",
)

print("\n" + text)
print("Saved results to:", OUT)
