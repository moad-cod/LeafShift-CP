
#!/usr/bin/env python3
"""P3.3 - Discover historical features and window keys.

Read-only audit. Does not modify archived experiments.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

FILES = [
    ROOT / "experiments/P1/P1_1/tables/P1_1_U5_window_predictions_24h.csv",
    ROOT / "experiments/P1/P1_1/tables/P1_1_U5_window_predictions_48h.csv",
]

SEARCH_DIRS = [
    ROOT / "data/processed/forecasting",
    ROOT / "experiments/E1/tables",
    ROOT / "experiments/E3/tables",
    ROOT / "experiments/P1/P1_2",
]

KEYWORDS = [
    "area", "history", "growth", "rate",
    "time", "gap", "window", "cutoff",
    "feature", "trajectory", "leaf",
    "target", "timestamp",
]

def inspect(path):
    print("\n" + "=" * 72)
    print("FILE:", path.relative_to(ROOT))
    print("=" * 72)

    try:
        if path.suffix == ".csv":
            df = pd.read_csv(path, nrows=5)
        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq
            schema = pq.read_schema(path)
            print("Parquet columns:")
            for name in schema.names:
                print("  ", name)
            return
        else:
            return

        print("Columns:")
        for col in df.columns:
            print("  ", col)

        print("\nPotential historical features:")
        for col in df.columns:
            if any(k in col.lower() for k in KEYWORDS):
                values = df[col].head(3).tolist()
                print(f"  {col}: {values}")

    except Exception as exc:
        print("ERROR:", repr(exc))


print("\nP3.3 HISTORICAL FEATURE DISCOVERY")

for path in FILES:
    if path.exists():
        inspect(path)
    else:
        print("MISSING:", path)

for folder in SEARCH_DIRS:
    print("\nSEARCHING:", folder.relative_to(ROOT))

    if not folder.exists():
        print("Directory missing")
        continue

    candidates = [
        p for p in folder.rglob("*")
        if p.is_file()
        and p.suffix.lower() in {".csv", ".parquet"}
    ]

    candidates = sorted(
        candidates,
        key=lambda p: (
            not any(k in p.name.lower() for k in KEYWORDS),
            str(p),
        ),
    )

    print("Candidate tables:", len(candidates))

    for path in candidates[:12]:
        inspect(path)

print("\nDiscovery completed. No files modified.")