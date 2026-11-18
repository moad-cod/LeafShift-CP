
#!/usr/bin/env python3
"""P3.2: Directional U5 miscoverage, using archived predictions.

No retraining, recalibration, or changes to manuscript files.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = ROOT / "experiments/P1/P1_1/tables"
OUT = ROOT / "experiments/P3/P3_2"
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    24: INPUT_DIR / "P1_1_U5_window_predictions_24h.csv",
    48: INPUT_DIR / "P1_1_U5_window_predictions_48h.csv",
}

EXPECTED = {
    24: (0.8997120, 8),
    48: (0.9004006, 6),
}

REQUIRED = [
    "test_genotype",
    "plant_id",
    "inner_fold",
    "target_area_pixels",
    "lower_pixels",
    "upper_pixels",
]

frames = []

for horizon, path in FILES.items():
    if not path.is_file():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing {missing}")

    df = df.rename(columns={
        "test_genotype": "genotype",
        "plant_id": "plant",
        "inner_fold": "assignment",
        "target_area_pixels": "y",
        "lower_pixels": "lower",
        "upper_pixels": "upper",
    })

    for col in ["y", "lower", "upper"]:
        df[col] = pd.to_numeric(df[col], errors="raise")

    if df[["y", "lower", "upper"]].isna().any().any():
        raise ValueError(f"Missing interval data at {horizon} h")

    if (df["lower"] > df["upper"]).any():
        raise ValueError(f"Crossed interval endpoints at {horizon} h")

    df["horizon"] = horizon

    # The three events are mutually exclusive.
    df["lower_miss"] = (df["y"] < df["lower"]).astype(int)
    df["upper_miss"] = (df["y"] > df["upper"]).astype(int)
    df["covered_calc"] = (
        (df["y"] >= df["lower"]) &
        (df["y"] <= df["upper"])
    ).astype(int)

    assert (
        df["lower_miss"] +
        df["upper_miss"] +
        df["covered_calc"]
    ).eq(1).all()

    if "covered" in df.columns:
        if not df["covered_calc"].eq(df["covered"]).all():
            raise ValueError(
                f"Archived coverage disagreement at {horizon} h"
            )

    frames.append(df)

data = pd.concat(frames, ignore_index=True)

# Equal-window average within each plant and assignment.
metrics = ["lower_miss", "upper_miss", "covered_calc"]

plant_assignment = (
    data.groupby(
        ["horizon", "genotype", "plant", "assignment"]
    )[metrics].mean().reset_index()
)

# Check the same four assignments for every plant.
assignment_counts = (
    plant_assignment.groupby(
        ["horizon", "genotype", "plant"]
    )["assignment"].nunique()
)

if not assignment_counts.eq(4).all():
    raise ValueError("Incomplete fitting/calibration assignments")

# Average the four paired assignments per plant.
plant = (
    plant_assignment.groupby(
        ["horizon", "genotype", "plant"]
    )[metrics].mean().reset_index()
)

# Equal-weight plants within genotype.
genotype = (
    plant.groupby(["horizon", "genotype"])
    .agg(
        plants=("plant", "nunique"),
        lower_miss=("lower_miss", "mean"),
        upper_miss=("upper_miss", "mean"),
        coverage=("covered_calc", "mean"),
    )
    .reset_index()
)

genotype["total_miss"] = (
    genotype["lower_miss"] + genotype["upper_miss"]
)
genotype["replication_group"] = np.where(
    genotype["plants"] < 8,
    "<8 plants",
    ">=8 plants",
)

# Verify original manuscript estimand.
for h, (expected_coverage, expected_below) in EXPECTED.items():
    sub = genotype[genotype["horizon"] == h]
    macro = sub["coverage"].mean()
    below = int((sub["coverage"] < 0.9).sum())

    if len(sub) != 21:
        raise ValueError("Expected 21 genotypes")

    if abs(macro - expected_coverage) > 0.00015:
        raise ValueError(
            f"{h} h coverage reproduction failed: {macro}"
        )

    if below != expected_below:
        raise ValueError(
            f"{h} h undercovered genotype count mismatch"
        )

    print(
        f"{h} h: macro coverage={macro:.5%}, "
        f"below 90%={below} — PASS"
    )

genotype.to_csv(
    OUT / "P3_2_genotype_directional.csv",
    index=False,
)
plant.to_csv(
    OUT / "P3_2_plant_directional.csv",
    index=False,
)

# Report exact directional components.
report = [
    "# P3.2 — Directional Miscoverage",
    "",
    "Retrospective exploratory analysis of archived U5 predictions.",
    "",
]

for h in [24, 48]:
    sub = genotype[genotype["horizon"] == h].copy()

    lower = sub["lower_miss"].mean()
    upper = sub["upper_miss"].mean()
    coverage = sub["coverage"].mean()

    report += [
        f"## {h}-hour horizon",
        "",
        f"- Macro coverage: {coverage:.5%}",
        f"- Macro lower-tail misses: {lower:.5%}",
        f"- Macro upper-tail misses: {upper:.5%}",
        f"- Total macro miscoverage: {lower + upper:.5%}",
        "",
        "### Genotypes with lowest coverage",
        "",
    ]

    worst = sub.nsmallest(6, "coverage")

    for _, row in worst.iterrows():
        report.append(
            f"- {row['genotype']} ({int(row['plants'])} plants): "
            f"coverage={row['coverage']:.3%}, "
            f"lower={row['lower_miss']:.3%}, "
            f"upper={row['upper_miss']:.3%}"
        )

    report += ["", "### Replication groups", ""]

    for group in ["<8 plants", ">=8 plants"]:
        s = sub[sub["replication_group"] == group]
        report.append(
            f"- {group}: {len(s)} genotypes; "
            f"lower={s['lower_miss'].mean():.3%}; "
            f"upper={s['upper_miss'].mean():.3%}"
        )

    # Separate publication-quality genotype figure.
    sub = sub.sort_values("coverage")
    y = np.arange(len(sub))

    fig, ax = plt.subplots(figsize=(9, 8))

    ax.barh(
        y,
        sub["lower_miss"] * 100,
        label="Lower-tail miss",
    )
    ax.barh(
        y,
        sub["upper_miss"] * 100,
        left=sub["lower_miss"] * 100,
        label="Upper-tail miss",
    )

    labels = [
        f"{g} (n={int(n)})"
        for g, n in zip(sub["genotype"], sub["plants"])
    ]

    ax.set_yticks(y, labels)
    ax.invert_yaxis()

    ax.axvline(
        10,
        linestyle="--",
        linewidth=1,
        color="black",
        label="10% nominal miscoverage",
    )

    ax.set_xlabel("Genotype-level empirical miscoverage (%)")
    ax.set_title(
        f"U5 directional miscoverage — {h}-hour forecast"
    )
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.15)

    fig.tight_layout()

    fig.savefig(
        OUT / f"P3_2_directional_{h}h.png",
        dpi=250,
        bbox_inches="tight",
    )
    fig.savefig(
        OUT / f"P3_2_directional_{h}h.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)

report += [
    "",
    "## Interpretation limits",
    "",
    "- Directional failures do not identify biological causes.",
    "- The same test plants are reused across four assignments.",
    "- Results are retrospective, not independent confirmation.",
    "- All original models and intervals remain unchanged.",
]

(OUT / "P3_2_results.md").write_text(
    "\n".join(report) + "\n",
    encoding="utf-8",
)

print("\n" + "\n".join(report))
print(f"\nResults saved to: {OUT}")
