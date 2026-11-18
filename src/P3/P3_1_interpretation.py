
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DIR = ROOT / "experiments/P3/P3_1"

samples = pd.read_csv(
    DIR / "P3_1_subsampling_summary.csv"
)
genotypes = pd.read_csv(
    DIR / "P3_1_genotype_summary.csv"
)

EXPECTED_MIN = {24: 0.8222, 48: 0.7639}

print("=" * 75)
print("P3.1 — PLANT REPLICATION SENSITIVITY INTERPRETATION")
print("=" * 75)

for horizon in [24, 48]:
    data = samples[samples["horizon"] == horizon]
    original = genotypes[
        genotypes["horizon"] == horizon
    ]

    print(f"\n{'=' * 75}")
    print(f"{horizon}-HOUR FORECAST")
    print(f"{'=' * 75}")

    print("\nObserved full-genotype coverage:")
    print(
        original[
            ["genotype", "plants", "coverage"]
        ].sort_values("coverage").to_string(index=False)
    )

    print("\nSubsampling results by plant count:")

    for n in [4, 5, 6, 7]:
        group = data[
            data["sample_plants"] == n
        ]

        print(f"\nSample size: {n} plants")
        print(f"Source genotypes: {len(group)}")

        print(
            "Mean probability of coverage <90%: "
            f"{group['frac_below_90'].mean():.4%}"
        )

        print(
            "Mean probability of coverage <= "
            f"{EXPECTED_MIN[horizon]:.2%}: "
            f"{group['frac_at_or_below_original_worst'].mean():.4%}"
        )

        print(
            "Mean subsample SD: "
            f"{group['subsample_sd'].mean():.5f}"
        )

        print(
            "Mean 95% percentile range: "
            f"[{group['p025'].mean():.4%}, "
            f"{group['p975'].mean():.4%}]"
        )

    print("\nPer-genotype sensitivity at n=4:")

    small = data[
        data["sample_plants"] == 4
    ].sort_values(
        "frac_at_or_below_original_worst",
        ascending=False
    )

    print(
        small[
            [
                "genotype",
                "full_plants",
                "full_coverage",
                "subsample_sd",
                "frac_below_90",
                "frac_at_or_below_original_worst",
            ]
        ].to_string(index=False)
    )

print("\n" + "=" * 75)
print("IMPORTANT INTERPRETATION LIMITATIONS")
print("=" * 75)
print(
    "Tail fractions are conditional on the observed "
    "plants in well-replicated genotypes."
)
print(
    "Averages across source genotypes are descriptive "
    "and are not population probabilities."
)
print(
    "This analysis cannot distinguish biological genotype "
    "effects from measurement and sampling variability."
)
print("\nAnalysis completed. No archived files modified.")
