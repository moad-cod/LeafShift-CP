
#!/usr/bin/env python3
"""P3.7: Paired exploratory comparison of adaptive CQR and U5."""

from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / "experiments/P3/P3_6/P3_6_genotype_metrics.csv"
OUT = ROOT / "experiments/P3/P3_7"
OUT.mkdir(parents=True, exist_ok=True)

SEED = 20261116
B = 20000
rng = np.random.default_rng(SEED)

df = pd.read_csv(INPUT)

required = {
    "horizon_hours", "method", "test_genotype",
    "coverage", "width", "score",
    "upper_miss", "lower_miss",
}
if missing := required - set(df.columns):
    raise ValueError(f"Missing fields: {missing}")

if df.duplicated(
    ["horizon_hours", "method", "test_genotype"]
).any():
    raise ValueError("Duplicate genotype-method records")

metrics = ["score", "width", "coverage",
           "upper_miss", "lower_miss"]

rows = []
paired_rows = []

for h in (24, 48):
    part = df[df.horizon_hours == h]

    if set(part.method) != {"U5", "SA_CQR", "SGA_CQR"}:
        raise ValueError(f"Missing methods at {h}h")

    wide = part.pivot(
        index="test_genotype",
        columns="method",
        values=metrics,
    ).sort_index()

    if len(wide) != 21 or wide.isna().any().any():
        raise ValueError(f"Incomplete genotype data: {h}h")

    for method in ("SA_CQR", "SGA_CQR"):
        for metric in metrics:
            # Negative score/width differences are favorable.
            diff = (
                wide[(metric, method)]
                - wide[(metric, "U5")]
            ).to_numpy(dtype=float)

            n = len(diff)
            draws = rng.integers(0, n, size=(B, n))
            boot = diff[draws].mean(axis=1)

            ci_lo, ci_hi = np.quantile(
                boot, [.025, .975]
            )

            loo = (
                diff.sum() - diff
            ) / (n - 1)

            scale = (
                100.0 if metric in
                ("coverage", "upper_miss", "lower_miss")
                else 1.0
            )

            rows.append({
                "horizon": h,
                "method": method,
                "metric": metric,
                "mean_delta": diff.mean() * scale,
                "median_delta": np.median(diff) * scale,
                "bootstrap_low": ci_lo * scale,
                "bootstrap_high": ci_hi * scale,
                "genotypes_lower": int((diff < 0).sum()),
                "genotypes_higher": int((diff > 0).sum()),
                "loo_min": loo.min() * scale,
                "loo_max": loo.max() * scale,
                "units": (
                    "percentage points"
                    if scale == 100 else "px²"
                ),
            })

        # Save every genotype's paired differences.
        for g in wide.index:
            paired_rows.append({
                "horizon": h,
                "method": method,
                "genotype": g,
                **{
                    f"delta_{m}": (
                        wide.loc[g, (m, method)]
                        - wide.loc[g, (m, "U5")]
                    )
                    for m in metrics
                },
            })

summary = pd.DataFrame(rows)
paired = pd.DataFrame(paired_rows)

summary.to_csv(
    OUT / "P3_7_paired_summary.csv", index=False
)
paired.to_csv(
    OUT / "P3_7_genotype_differences.csv", index=False
)

print("=" * 72)
print("P3.7 — PAIRED GENOTYPE COMPARISON")
print("=" * 72)

for h in (24, 48):
    print(f"\n{h}-HOUR FORECAST")
    s = summary[
        (summary.horizon == h)
        & (summary.metric == "score")
    ]

    for _, row in s.iterrows():
        print(f"\n{row.method} versus U5")
        print(f"Mean score difference: {row.mean_delta:+.3f} px²")
        print(
            "Bootstrap percentile interval: "
            f"[{row.bootstrap_low:+.3f}, "
            f"{row.bootstrap_high:+.3f}]"
        )
        print(
            f"Genotypes with lower score: "
            f"{row.genotypes_lower}/21"
        )
        print(
            "Leave-one-genotype-out mean range: "
            f"[{row.loo_min:+.3f}, {row.loo_max:+.3f}]"
        )

print("\nInterpretation:")
print("Negative score differences favor the adaptive method.")
print("Intervals are exploratory genotype-bootstrap summaries.")
print("No independent validation or coverage guarantee is implied.")
print(f"\nSaved results to {OUT}")
