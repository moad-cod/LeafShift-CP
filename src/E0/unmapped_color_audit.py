
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from PIL import Image, ImageDraw
import csv

# ============================================================
# E0.2c — UNMAPPED COLOR LOCALIZATION
#
# Locate the seven colors not documented in the official
# palette and inspect their distribution over genotype,
# plant, and time.
#
# Original dataset is READ-ONLY.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

DATA = ROOT / "data/raw/PhenoLeaf-TS_DS"
RAW = DATA / "01_raw_dataset"
LABELS = DATA / "02_leaf_labels"

OUT = ROOT / "experiments/E0"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures" / "E0_2c_examples"

for folder in (TABLES, REPORTS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

UNKNOWN_COLORS = {
    (37, 3, 193),
    (255, 0, 124),
    (218, 28, 13),
    (181, 220, 171),
    (91, 20, 124),
    (209, 136, 154),
    (191, 155, 14),
}


def timestamp_from_filename(name):
    parts = name.split("_")
    try:
        return datetime(
            *map(int, parts[1:7])
        )
    except (ValueError, IndexError):
        return None


def write_csv(path, rows, columns):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def inspect_mask(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        counts = rgb.getcolors(
            maxcolors=rgb.width * rgb.height
        )

    if counts is None:
        raise RuntimeError(
            f"Unable to count colors: {path}"
        )

    return dict(
        (color, count)
        for count, color in counts
    )


def save_example(raw_path, mask_path, rgb, output):
    """
    Save side-by-side raw image and corresponding
    annotation for visual investigation.
    """
    with Image.open(raw_path) as im:
        raw = im.convert("RGB")
        raw.thumbnail((600, 600))

    with Image.open(mask_path) as im:
        mask = im.convert("RGB")
        mask.thumbnail((600, 600))

    height = max(raw.height, mask.height) + 60
    width = raw.width + mask.width + 30

    canvas = Image.new(
        "RGB", (width, height), "white"
    )

    canvas.paste(raw, (0, 45))
    canvas.paste(mask, (raw.width + 20, 45))

    draw = ImageDraw.Draw(canvas)
    draw.text((5, 8), "Raw image", fill="black")
    draw.text(
        (raw.width + 25, 8),
        f"Label mask | unmapped RGB {rgb}",
        fill="black",
    )

    canvas.save(output)


def main():
    print("=" * 65)
    print("E0.2c — UNMAPPED COLOR LOCALIZATION")
    print("=" * 65)

    occurrences = []
    by_color = defaultdict(list)
    best_example = {}

    total_masks = 0

    for genotype_dir in sorted(LABELS.iterdir()):
        if not genotype_dir.is_dir():
            continue

        genotype = genotype_dir.name

        for rep_dir in sorted(genotype_dir.iterdir()):
            if not rep_dir.is_dir():
                continue

            replicate = rep_dir.name
            plant_id = f"{genotype}/{replicate}"

            files = sorted(
                (
                    p for p in rep_dir.iterdir()
                    if p.is_file()
                    and p.suffix.lower() == ".png"
                ),
                key=lambda p: (
                    timestamp_from_filename(p.name)
                    or datetime.min,
                    p.name,
                ),
            )

            for frame_index, path in enumerate(files):
                total_masks += 1
                counts = inspect_mask(path)

                matched_colors = (
                    UNKNOWN_COLORS & counts.keys()
                )

                if not matched_colors:
                    continue

                timestamp = timestamp_from_filename(
                    path.name
                )

                for rgb in sorted(matched_colors):
                    pixels = counts[rgb]

                    record = {
                        "genotype": genotype,
                        "replicate": replicate,
                        "plant_id": plant_id,
                        "frame_index": frame_index,
                        "timestamp": (
                            timestamp.isoformat()
                            if timestamp else ""
                        ),
                        "filename": path.name,
                        "rgb": str(rgb),
                        "pixels": pixels,
                    }

                    occurrences.append(record)
                    by_color[rgb].append(record)

                    previous_best = best_example.get(rgb)

                    if (
                        previous_best is None
                        or pixels > previous_best[0]
                    ):
                        raw_path = (
                            RAW / genotype
                            / replicate / path.name
                        )

                        best_example[rgb] = (
                            pixels,
                            raw_path,
                            path,
                            plant_id,
                        )

        print(f"Processed genotype: {genotype}")

    columns = [
        "genotype", "replicate", "plant_id",
        "frame_index", "timestamp", "filename",
        "rgb", "pixels"
    ]

    write_csv(
        TABLES / "E0_2c_color_occurrences.csv",
        occurrences,
        columns,
    )

    summaries = []

    for rgb in sorted(UNKNOWN_COLORS):
        records = by_color[rgb]

        plants = sorted({
            r["plant_id"] for r in records
        })

        genotypes = sorted({
            r["genotype"] for r in records
        })

        frames_by_plant = defaultdict(list)

        for record in records:
            frames_by_plant[
                record["plant_id"]
            ].append(record["frame_index"])

        adjacent_pairs = 0
        repeated_plants = 0

        for indices in frames_by_plant.values():
            indices = sorted(indices)

            if len(indices) >= 2:
                repeated_plants += 1

            adjacent_pairs += sum(
                b == a + 1
                for a, b in zip(
                    indices, indices[1:]
                )
            )

        summaries.append({
            "rgb": str(rgb),
            "total_pixels": sum(
                r["pixels"] for r in records
            ),
            "frames_present": len(records),
            "plants_present": len(plants),
            "genotypes_present": len(genotypes),
            "plants_with_repeated_occurrence": (
                repeated_plants
            ),
            "adjacent_frame_pairs": (
                adjacent_pairs
            ),
            "genotypes": ";".join(genotypes),
            "plants": ";".join(plants),
        })

        example = best_example.get(rgb)

        if example:
            pixels, raw_path, mask_path, plant_id = example

            if raw_path.is_file():
                color_id = "_".join(
                    map(str, rgb)
                )

                save_example(
                    raw_path,
                    mask_path,
                    rgb,
                    FIGURES / (
                        f"unknown_{color_id}.png"
                    ),
                )

    summary_columns = [
        "rgb",
        "total_pixels",
        "frames_present",
        "plants_present",
        "genotypes_present",
        "plants_with_repeated_occurrence",
        "adjacent_frame_pairs",
        "genotypes",
        "plants",
    ]

    write_csv(
        TABLES / "E0_2c_color_summary.csv",
        summaries,
        summary_columns,
    )

    report = [
        "# E0.2c — Unmapped Color Localization",
        "",
        f"Total masks scanned: {total_masks}",
        "",
        "| RGB | Pixels | Frames | Plants | Genotypes | Adjacent pairs |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for r in summaries:
        report.append(
            f"| {r['rgb']} "
            f"| {r['total_pixels']} "
            f"| {r['frames_present']} "
            f"| {r['plants_present']} "
            f"| {r['genotypes_present']} "
            f"| {r['adjacent_frame_pairs']} |"
        )

    report.extend([
        "",
        "## Interpretation",
        "",
        "Repeated occurrence can indicate a persistent "
        "annotation category, but does not prove that "
        "the color corresponds to a biological leaf.",
        "",
        "Inspect example images and the dataset README "
        "before updating the leaf palette.",
        "",
        "The existing E0.2 trajectories are unchanged.",
    ])

    (
        REPORTS / "E0_2c_unmapped_color_report.md"
    ).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 65)
    print("E0.2c — SUMMARY")
    print("=" * 65)

    print("Masks scanned:", total_masks)

    for row in summaries:
        print(
            f"{row['rgb']:20s} "
            f"frames={row['frames_present']:5d} "
            f"plants={row['plants_present']:4d} "
            f"genotypes={row['genotypes_present']:3d} "
            f"adjacent_pairs={row['adjacent_frame_pairs']:5d}"
        )

    print("\nSaved CSV tables, report, and example images.")
    print("E0.2c complete.")


if __name__ == "__main__":
    main()
