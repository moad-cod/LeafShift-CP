
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
import csv
import re
from statistics import median

# ============================================================
# E0.1 — PHENOLEAF-TS DATASET STRUCTURE AUDIT
#
# Purpose:
#   Verify genotype structure, candidate plant identities,
#   image/label pairing, acquisition timestamps, and
#   sequence completeness before designing forecasting splits.
#
# Dataset remains READ-ONLY.
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT / "data" / "raw" / "PhenoLeaf-TS_DS"
)

RAW_ROOT = DATASET_ROOT / "01_raw_dataset"
LABEL_ROOT = DATASET_ROOT / "02_leaf_labels"

OUTPUT_ROOT = PROJECT_ROOT / "experiments" / "E0"

TABLES_DIR = OUTPUT_ROOT / "tables"
REPORTS_DIR = OUTPUT_ROOT / "reports"
FIGURES_DIR = OUTPUT_ROOT / "figures"

for directory in (TABLES_DIR, REPORTS_DIR, FIGURES_DIR):
    directory.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. UTILITIES
# ============================================================

TIMESTAMP_PATTERN = re.compile(
    r"(?<!\d)"
    r"(\d{4})_(\d{2})_(\d{2})_"
    r"(\d{2})_(\d{2})_(\d{2})"
    r"(?!\d)"
)


def extract_timestamp(filename):
    """Parse YYYY_MM_DD_HH_MM_SS from an image filename."""

    match = TIMESTAMP_PATTERN.search(filename)

    if match is None:
        return None

    try:
        return datetime(
            *map(int, match.groups())
        )
    except ValueError:
        return None


def get_png_files(directory):
    """Return PNG files indexed by filename."""

    if not directory.is_dir():
        return {}

    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() == ".png"
    }


def save_csv(path, rows, fieldnames):
    """Write reproducible CSV output."""

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames
        )

        writer.writeheader()
        writer.writerows(rows)


def fmt(value, digits=2):
    if value is None:
        return "N/A"
    return f"{value:.{digits}f}"


# ============================================================
# 2. CHECK DATASET DIRECTORIES
# ============================================================

print("=" * 65)
print("E0.1 — PHENOLEAF-TS STRUCTURE AUDIT")
print("=" * 65)

if not RAW_ROOT.is_dir():
    raise FileNotFoundError(
        f"Missing raw dataset: {RAW_ROOT}"
    )

if not LABEL_ROOT.is_dir():
    raise FileNotFoundError(
        f"Missing label dataset: {LABEL_ROOT}"
    )

raw_genotypes = {
    p.name: p
    for p in RAW_ROOT.iterdir()
    if p.is_dir()
}

label_genotypes = {
    p.name: p
    for p in LABEL_ROOT.iterdir()
    if p.is_dir()
}

all_genotypes = sorted(
    set(raw_genotypes) | set(label_genotypes)
)

print(f"\nRaw genotype directories: {len(raw_genotypes)}")
print(f"Label genotype directories: {len(label_genotypes)}")

print(
    "Genotype directories matching: ",
    set(raw_genotypes) == set(label_genotypes)
)


# ============================================================
# 3. BUILD IMAGE AND PLANT MANIFEST
# ============================================================

image_rows = []
plant_rows = []
genotype_rows = []

all_timestamp_gaps = []
all_dates = []

total_raw_images = 0
total_label_images = 0
total_matched_images = 0

for genotype in all_genotypes:

    raw_genotype_dir = RAW_ROOT / genotype
    label_genotype_dir = LABEL_ROOT / genotype

    raw_reps = {
        p.name
        for p in raw_genotype_dir.iterdir()
        if p.is_dir()
    } if raw_genotype_dir.is_dir() else set()

    label_reps = {
        p.name
        for p in label_genotype_dir.iterdir()
        if p.is_dir()
    } if label_genotype_dir.is_dir() else set()

    all_reps = sorted(raw_reps | label_reps)

    genotype_plants = []
    genotype_raw_count = 0
    genotype_label_count = 0
    genotype_matched_count = 0

    print(f"\nProcessing genotype: {genotype}")

    for rep in all_reps:

        plant_id = f"{genotype}/{rep}"

        raw_dir = raw_genotype_dir / rep
        label_dir = label_genotype_dir / rep

        raw_files = get_png_files(raw_dir)
        label_files = get_png_files(label_dir)

        raw_names = set(raw_files)
        label_names = set(label_files)

        matched = raw_names & label_names
        missing_labels = raw_names - label_names
        missing_raw = label_names - raw_names

        genotype_raw_count += len(raw_names)
        genotype_label_count += len(label_names)
        genotype_matched_count += len(matched)

        timestamp_entries = []
        invalid_timestamps = 0

        for filename in sorted(raw_names | label_names):

            timestamp = extract_timestamp(filename)

            in_raw = filename in raw_names
            in_label = filename in label_names

            if in_raw and timestamp is None:
                invalid_timestamps += 1

            if in_raw and timestamp is not None:
                timestamp_entries.append(
                    (timestamp, filename)
                )

            image_rows.append({
                "genotype": genotype,
                "replicate": rep,
                "plant_id": plant_id,
                "filename": filename,
                "timestamp": (
                    timestamp.isoformat()
                    if timestamp else ""
                ),
                "raw_exists": int(in_raw),
                "label_exists": int(in_label),
                "paired": int(in_raw and in_label),
            })

        timestamp_entries.sort()

        dates = [
            entry[0]
            for entry in timestamp_entries
        ]

        duplicates = (
            len(dates) - len(set(dates))
        )

        gaps_hours = [
            (dates[i + 1] - dates[i]).total_seconds()
            / 3600
            for i in range(len(dates) - 1)
        ]

        positive_gaps = [
            gap for gap in gaps_hours
            if gap > 0
        ]

        all_timestamp_gaps.extend(positive_gaps)
        all_dates.extend(dates)

        duration_hours = (
            (dates[-1] - dates[0]).total_seconds()
            / 3600
            if len(dates) >= 2 else None
        )

        median_gap = (
            median(positive_gaps)
            if positive_gaps else None
        )

        plant = {
            "genotype": genotype,
            "replicate": rep,
            "plant_id": plant_id,
            "raw_images": len(raw_names),
            "label_images": len(label_names),
            "matched_pairs": len(matched),
            "missing_labels": len(missing_labels),
            "missing_raw_images": len(missing_raw),
            "invalid_raw_timestamps": invalid_timestamps,
            "duplicate_raw_timestamps": duplicates,
            "first_timestamp": (
                dates[0].isoformat()
                if dates else ""
            ),
            "last_timestamp": (
                dates[-1].isoformat()
                if dates else ""
            ),
            "duration_hours": (
                round(duration_hours, 3)
                if duration_hours is not None else ""
            ),
            "median_gap_hours": (
                round(median_gap, 3)
                if median_gap is not None else ""
            ),
            "complete_pairing": (
                int(
                    raw_names == label_names
                    and len(raw_names) > 0
                )
            ),
        }

        plant_rows.append(plant)
        genotype_plants.append(plant)

    genotype_rows.append({
        "genotype": genotype,
        "candidate_plants": len(genotype_plants),
        "raw_images": genotype_raw_count,
        "label_images": genotype_label_count,
        "matched_pairs": genotype_matched_count,
        "plants_with_complete_pairing": sum(
            p["complete_pairing"]
            for p in genotype_plants
        ),
        "median_raw_frames_per_plant": (
            median([
                p["raw_images"]
                for p in genotype_plants
            ])
            if genotype_plants else ""
        ),
        "median_duration_hours": (
            round(median([
                p["duration_hours"]
                for p in genotype_plants
                if p["duration_hours"] != ""
            ]), 2)
            if any(
                p["duration_hours"] != ""
                for p in genotype_plants
            )
            else ""
        ),
    })

    total_raw_images += genotype_raw_count
    total_label_images += genotype_label_count
    total_matched_images += genotype_matched_count


# ============================================================
# 4. SAVE CSV TABLES
# ============================================================

save_csv(
    TABLES_DIR / "E0_1_genotype_summary.csv",
    genotype_rows,
    [
        "genotype",
        "candidate_plants",
        "raw_images",
        "label_images",
        "matched_pairs",
        "plants_with_complete_pairing",
        "median_raw_frames_per_plant",
        "median_duration_hours",
    ],
)

save_csv(
    TABLES_DIR / "E0_1_plant_manifest.csv",
    plant_rows,
    list(plant_rows[0].keys()) if plant_rows else [
        "genotype", "replicate", "plant_id"
    ],
)

save_csv(
    TABLES_DIR / "E0_1_image_pairing.csv",
    image_rows,
    list(image_rows[0].keys()) if image_rows else [
        "genotype", "replicate", "plant_id",
        "filename", "timestamp",
        "raw_exists", "label_exists", "paired"
    ],
)


# ============================================================
# 5. SUMMARY STATISTICS
# ============================================================

total_plants = len(plant_rows)

complete_plants = sum(
    p["complete_pairing"] for p in plant_rows
)

missing_labels_total = sum(
    p["missing_labels"] for p in plant_rows
)

missing_raw_total = sum(
    p["missing_raw_images"] for p in plant_rows
)

invalid_timestamp_total = sum(
    p["invalid_raw_timestamps"] for p in plant_rows
)

duplicate_timestamp_total = sum(
    p["duplicate_raw_timestamps"] for p in plant_rows
)

plant_counts = [
    row["candidate_plants"]
    for row in genotype_rows
]

frame_counts = [
    row["raw_images"]
    for row in plant_rows
]

durations = [
    row["duration_hours"]
    for row in plant_rows
    if row["duration_hours"] != ""
]

summary = {
    "genotypes": len(all_genotypes),
    "candidate_plants": total_plants,
    "raw_images": total_raw_images,
    "label_images": total_label_images,
    "matched_pairs": total_matched_images,
    "complete_pairing_plants": complete_plants,
    "missing_labels": missing_labels_total,
    "missing_raw_images": missing_raw_total,
    "invalid_raw_timestamps": invalid_timestamp_total,
    "duplicate_raw_timestamps": duplicate_timestamp_total,
    "median_plants_per_genotype": (
        median(plant_counts) if plant_counts else None
    ),
    "median_frames_per_plant": (
        median(frame_counts) if frame_counts else None
    ),
    "median_duration_hours": (
        median(durations) if durations else None
    ),
    "median_sampling_gap_hours": (
        median(all_timestamp_gaps)
        if all_timestamp_gaps else None
    ),
}


# ============================================================
# 6. SAVE FIGURE
# ============================================================

try:
    import matplotlib.pyplot as plt

    names = [
        row["genotype"] for row in genotype_rows
    ]

    counts = [
        row["candidate_plants"]
        for row in genotype_rows
    ]

    fig, ax = plt.subplots(
        figsize=(12, max(6, len(names) * 0.35))
    )

    ax.barh(names, counts)
    ax.invert_yaxis()

    ax.set_xlabel("Candidate plant replicates")
    ax.set_ylabel("Genotype")
    ax.set_title(
        "PhenoLeaf-TS — Candidate Plants per Genotype"
    )

    fig.tight_layout()

    figure_path = (
        FIGURES_DIR /
        "E0_1_plants_per_genotype.png"
    )

    fig.savefig(
        figure_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close(fig)

    print(f"\nFigure saved: {figure_path}")

except ImportError:
    print(
        "\nMatplotlib not installed; "
        "skipping figure."
    )


# ============================================================
# 7. AUTOMATIC MARKDOWN REPORT
# ============================================================

all_paired = (
    total_raw_images == total_matched_images
    and total_label_images == total_matched_images
)

report = f"""# E0.1 — PhenoLeaf-TS Dataset Structure Audit

## Objective

Assess whether the dataset provides sufficiently organized
genotype, plant, image, annotation, and timestamp metadata
for genotype-held-out growth forecasting.

## Dataset

- Root: `data/raw/PhenoLeaf-TS_DS`
- Raw images: `01_raw_dataset`
- Leaf masks: `02_leaf_labels`

## Results

| Metric | Value |
|---|---:|
| Genotypes | {summary['genotypes']} |
| Candidate plant replicates | {total_plants} |
| Raw images | {total_raw_images} |
| Leaf-label images | {total_label_images} |
| Exact filename pairs | {total_matched_images} |
| Plants with complete pairing | {complete_plants} |
| Missing labels | {missing_labels_total} |
| Labels without raw images | {missing_raw_total} |
| Invalid raw timestamps | {invalid_timestamp_total} |
| Duplicate raw timestamps | {duplicate_timestamp_total} |
| Median plants per genotype | {fmt(summary['median_plants_per_genotype'])} |
| Median frames per plant | {fmt(summary['median_frames_per_plant'])} |
| Median observation duration (hours) | {fmt(summary['median_duration_hours'])} |
| Median within-plant sampling gap (hours) | {fmt(summary['median_sampling_gap_hours'])} |

## Initial assessment

Genotype directories are available: {bool(raw_genotypes)}.

Raw and label genotype sets match:
{set(raw_genotypes) == set(label_genotypes)}.

All indexed images have matching annotations:
{all_paired}.

## Interpretation

The genotype directories provide candidate genotype identifiers.
Replicate directories provide candidate independent plant IDs.

These assumptions must be checked against the dataset README
and original data-acquisition documentation.

Chronological sequences can be reconstructed from filename
timestamps where timestamps are valid.

## Limitations

This audit does not verify:

1. Biological independence of replicate directories.
2. Temporal persistence of individual leaf labels.
3. Pixel-level annotation correctness.
4. Genotype-specific usable forecasting windows.
5. Availability of suitable uncertainty-calibration groups.

A genotype-held-out forecasting protocol is not yet approved.

## Next experiment

E0.2 — Leaf Identity and Area Trajectory Audit.
"""

report_path = (
    REPORTS_DIR /
    "E0_1_dataset_structure_report.md"
)

report_path.write_text(
    report,
    encoding="utf-8"
)


# ============================================================
# 8. TERMINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("E0.1 — RESULTS")
print("=" * 65)

for key, value in summary.items():
    print(f"{key:34s}: {value}")

print("\nOutputs saved to:")
print(f"  {TABLES_DIR}")
print(f"  {REPORTS_DIR}")

print("\nE0.1 complete.")
print("Next: E0.2 — Leaf Identity and Trajectory Audit")
