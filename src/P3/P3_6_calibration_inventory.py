
#!/usr/bin/env python3
"""
P3.6 — Locate development-only CQR calibration artifacts.

Read-only repository inspection.
Does not train models or modify any existing experiments.
"""

from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]

SEARCH_DIRS = [
    ROOT / "experiments" / "E2",
    ROOT / "experiments" / "E3",
    ROOT / "experiments" / "P0",
    ROOT / "experiments" / "P1",
    ROOT / "data" / "processed",
]

KEYWORDS = (
    "calibration",
    "calib",
    "cqr",
    "quantile",
    "score",
    "prediction",
    "residual",
    "threshold",
    "fold",
)

EXTENSIONS = {".csv", ".parquet", ".feather", ".npz"}

print("=" * 75)
print("P3.6 — DEVELOPMENT CALIBRATION ARTIFACT DISCOVERY")
print("=" * 75)

candidates = []

for folder in SEARCH_DIRS:
    if not folder.exists():
        continue

    for path in folder.rglob("*"):
        if (
            path.is_file()
            and path.suffix.lower() in EXTENSIONS
            and any(
                word in path.name.lower()
                for word in KEYWORDS
            )
        ):
            candidates.append(path)

candidates = sorted(set(candidates))

print(f"\nMatching artifacts: {len(candidates)}")

for path in candidates:
    relative = path.relative_to(ROOT)
    size_mb = path.stat().st_size / 1024**2

    print(f"\nFILE: {relative}")
    print(f"Size: {size_mb:.2f} MB")

    try:
        if path.suffix == ".csv":
            frame = pd.read_csv(path, nrows=2)
            columns = frame.columns.tolist()

        elif path.suffix == ".parquet":
            import pyarrow.parquet as pq
            columns = pq.read_schema(path).names

        elif path.suffix == ".feather":
            import pyarrow.feather as feather
            columns = feather.read_table(path).column_names

        else:
            with __import__("numpy").load(
                path, allow_pickle=False
            ) as data:
                columns = list(data.files)

        print("Columns/arrays:", columns)

        relevant = [
            col for col in columns
            if any(
                keyword in col.lower()
                for keyword in (
                    "cal", "score", "quantile",
                    "lower", "upper", "genotype",
                    "plant", "fold", "weight",
                )
            )
        ]

        if relevant:
            print("Relevant fields:", relevant)

    except Exception as error:
        print("Inspection error:", repr(error))

print("\n" + "=" * 75)
print("CQR IMPLEMENTATION FILES")
print("=" * 75)

SOURCE_FILES = [
    ROOT / "src/E3/run_cqr_baselines.py",
    ROOT / "src/P0/cqr_clipping_sensitivity.py",
    ROOT / "src/P1/p11_u5_prediction_export.py",
]

TERMS = (
    "calibrat",
    "quantile",
    "predict(",
    "score",
    "sample_weight",
    "joblib",
    "pickle",
    "save",
    "to_csv",
)

for path in SOURCE_FILES:
    if not path.exists():
        continue

    lines = path.read_text(
        encoding="utf-8", errors="replace"
    ).splitlines()

    print(f"\nSOURCE: {path.relative_to(ROOT)}")
    print(f"Total lines: {len(lines)}")

    matches = [
        (i, line.strip())
        for i, line in enumerate(lines, 1)
        if any(term in line.lower() for term in TERMS)
    ]

    for line_no, line in matches[:55]:
        print(f"{line_no:4}: {line[:140]}")

    if len(matches) > 55:
        print(
            f"... {len(matches)-55} additional matching lines"
        )

print("\nRead-only discovery completed.")
