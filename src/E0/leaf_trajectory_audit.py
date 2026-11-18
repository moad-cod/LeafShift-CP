
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median
import csv
import re

from PIL import Image
from openpyxl import load_workbook

# ============================================================
# E0.2 — LEAF IDENTITY AND TRAJECTORY AUDIT
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/PhenoLeaf-TS_DS"
RAW = DATA / "01_raw_dataset"
LABELS = DATA / "02_leaf_labels"
PALETTE_FILE = DATA / "leaves_unique_colours.xlsx"

OUT = ROOT / "experiments/E0"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"

for folder in (TABLES, REPORTS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

# First-pass sampling; change to None for all plants
MAX_PLANTS_PER_GENOTYPE = None

# Pixels below this threshold are treated as tiny detections
MIN_LEAF_PIXELS = 20

STAMP_RE = re.compile(
    r"(\d{4})_(\d{2})_(\d{2})_"
    r"(\d{2})_(\d{2})_(\d{2})"
)

RGB_RE = re.compile(
    r"\(?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)?"
)


def timestamp_from_name(name):
    match = STAMP_RE.search(name)
    if not match:
        return None
    try:
        return datetime(*map(int, match.groups()))
    except ValueError:
        return None


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_palette():
    wb = load_workbook(
        PALETTE_FILE,
        read_only=True,
        data_only=True
    )

    ws = wb["Leaf Colour Palette"]
    mapping = {}

    for row in ws.iter_rows(min_row=2, values_only=True):
        if len(row) < 3:
            continue

        rgb_text = row[1]
        leaf_number = row[2]

        if rgb_text is None or leaf_number is None:
            continue

        match = RGB_RE.fullmatch(str(rgb_text).strip())
        if not match:
            print("Skipped palette row:", rgb_text, leaf_number)
            continue

        rgb = tuple(map(int, match.groups()))
        leaf_number = int(leaf_number)

        if any(value > 255 for value in rgb):
            raise ValueError(f"Invalid RGB: {rgb}")

        if rgb in mapping and mapping[rgb] != leaf_number:
            raise ValueError(f"Conflicting palette color: {rgb}")

        mapping[rgb] = leaf_number

    wb.close()

    if not mapping:
        raise RuntimeError("No valid leaf colors in workbook")

    return mapping


def color_counts(mask_path):
    """
    Count the exact RGB colors present in a PNG.

    This deliberately does not assume the background is black.
    Unknown colors are recorded, not silently assigned to leaves.
    """
    with Image.open(mask_path) as image:
        rgb_image = image.convert("RGB")
        total_pixels = rgb_image.width * rgb_image.height
        items = rgb_image.getcolors(maxcolors=total_pixels)

    if items is None:
        raise RuntimeError(
            f"Could not enumerate colors: {mask_path}"
        )

    return {
        color: count
        for count, color in items
    }, total_pixels


def main():
    print("=" * 62)
    print("E0.2 — INDIVIDUAL-LEAF TRAJECTORY AUDIT")
    print("=" * 62)

    palette = load_palette()
    leaf_ids = sorted(set(palette.values()))

    print(f"Palette entries: {len(palette)}")
    print(f"Leaf IDs: {leaf_ids[:5]} ... {leaf_ids[-5:]}")

    trajectory_rows = []
    plant_rows = []
    unknown_colors = Counter()
    mask_modes = Counter()

    all_plants = []
    skipped_masks = 0
    total_masks = 0

    for genotype_dir in sorted(RAW.iterdir()):
        if not genotype_dir.is_dir():
            continue

        genotype = genotype_dir.name

        reps = sorted(
            p for p in genotype_dir.iterdir()
            if p.is_dir()
        )

        selected_reps = (
            reps[:MAX_PLANTS_PER_GENOTYPE]
            if MAX_PLANTS_PER_GENOTYPE is not None
            else reps
        )

        for rep_dir in selected_reps:
            all_plants.append(
                (genotype, rep_dir.name, rep_dir)
            )

    print(f"Plants selected: {len(all_plants)}")

    for index, (genotype, rep, rep_dir) in enumerate(
        all_plants, start=1
    ):
        plant_id = f"{genotype}/{rep}"
        label_dir = LABELS / genotype / rep

        images = sorted(
            (
                (timestamp_from_name(p.name), p.name)
                for p in rep_dir.iterdir()
                if p.is_file() and p.suffix.lower() == ".png"
            ),
            key=lambda x: (
                x[0] or datetime.min,
                x[1]
            )
        )

        per_leaf = defaultdict(list)
        valid_frames = 0
        unknown_fraction_sum = 0.0
        largest_unknown_fraction = 0.0
        dimensions = set()

        for timestamp, filename in images:
            mask_path = label_dir / filename

            if timestamp is None or not mask_path.is_file():
                skipped_masks += 1
                continue

            with Image.open(mask_path) as image:
                mask_modes[image.mode] += 1
                dimensions.add(image.size)

            counts, total_pixels = color_counts(mask_path)

            valid_frames += 1
            total_masks += 1

            identified_pixels = sum(
                counts.get(rgb, 0)
                for rgb in palette
            )

            unidentified_pixels = (
                total_pixels - identified_pixels
            )

            unknown_fraction = (
                unidentified_pixels / total_pixels
            )

            unknown_fraction_sum += unknown_fraction

            largest_unknown_fraction = max(
                largest_unknown_fraction,
                unknown_fraction
            )

            for rgb, count in counts.items():
                if rgb not in palette:
                    unknown_colors[rgb] += count

            for rgb, leaf_id in palette.items():
                pixels = counts.get(rgb, 0)

                # Each color maps to one numbered leaf
                per_leaf[leaf_id].append(
                    (timestamp, pixels)
                )

                if pixels >= MIN_LEAF_PIXELS:
                    trajectory_rows.append({
                        "genotype": genotype,
                        "replicate": rep,
                        "plant_id": plant_id,
                        "timestamp": timestamp.isoformat(),
                        "filename": filename,
                        "leaf_id": leaf_id,
                        "area_pixels": pixels,
                        "image_pixels": total_pixels,
                        "area_fraction": round(
                            pixels / total_pixels, 9
                        ),
                    })

        usable_leaf_ids = []
        disappearing_reappearing = 0
        sharp_drops = 0
        usable_leaf_observations = 0

        for leaf_id, series in per_leaf.items():
            if len(series) != valid_frames:
                continue

            areas = [pixels for _, pixels in series]

            visible = [
                area >= MIN_LEAF_PIXELS
                for area in areas
            ]

            n_visible = sum(visible)

            if n_visible >= 4:
                usable_leaf_ids.append(leaf_id)
                usable_leaf_observations += n_visible

            # A leaf visible, then absent, then visible again.
            # Could indicate occlusion or an identity issue.
            if any(visible):
                first = visible.index(True)
                last = len(visible) - 1 - visible[::-1].index(True)
                if not all(visible[first:last + 1]):
                    disappearing_reappearing += 1

            # Flag large drops; these are NOT automatically errors.
            for previous, current in zip(areas, areas[1:]):
                if (
                    previous >= MIN_LEAF_PIXELS
                    and current >= MIN_LEAF_PIXELS
                    and current < previous * 0.5
                ):
                    sharp_drops += 1

        plant_rows.append({
            "genotype": genotype,
            "replicate": rep,
            "plant_id": plant_id,
            "raw_frames": len(images),
            "valid_mask_frames": valid_frames,
            "leaf_ids_visible_4plus_frames": len(
                usable_leaf_ids
            ),
            "usable_leaf_observations": (
                usable_leaf_observations
            ),
            "reappearing_leaf_ids": (
                disappearing_reappearing
            ),
            "sharp_area_drops": sharp_drops,
            "mean_nonpalette_fraction": round(
                unknown_fraction_sum / valid_frames, 6
            ) if valid_frames else "",
            "max_nonpalette_fraction": round(
                largest_unknown_fraction, 6
            ) if valid_frames else "",
            "mask_dimensions": ";".join(
                f"{w}x{h}" for w, h in sorted(dimensions)
            ),
        })

        print(
            f"[{index}/{len(all_plants)}] "
            f"{plant_id}: "
            f"{valid_frames} masks, "
            f"{len(usable_leaf_ids)} candidate leaf tracks"
        )

    save_csv(
        TABLES / "E0_2_leaf_area_trajectories.csv",
        trajectory_rows,
        [
            "genotype", "replicate", "plant_id",
            "timestamp", "filename", "leaf_id",
            "area_pixels", "image_pixels",
            "area_fraction"
        ]
    )

    save_csv(
        TABLES / "E0_2_plant_quality_summary.csv",
        plant_rows,
        [
            "genotype", "replicate", "plant_id",
            "raw_frames", "valid_mask_frames",
            "leaf_ids_visible_4plus_frames",
            "usable_leaf_observations",
            "reappearing_leaf_ids",
            "sharp_area_drops",
            "mean_nonpalette_fraction",
            "max_nonpalette_fraction",
            "mask_dimensions"
        ]
    )

    save_csv(
        TABLES / "E0_2_unknown_colors.csv",
        [
            {
                "rgb": str(rgb),
                "total_pixels": count
            }
            for rgb, count in unknown_colors.most_common()
        ],
        ["rgb", "total_pixels"]
    )

    total_tracks = sum(
        row["leaf_ids_visible_4plus_frames"]
        for row in plant_rows
    )

    report = f"""# E0.2 — Leaf Identity and Trajectory Audit

## Scope
- Dataset: PhenoLeaf-TS
- Plants sampled: {len(plant_rows)}
- Genotypes sampled: {len(set(p['genotype'] for p in plant_rows))}
- Masks decoded: {total_masks}
- Missing/unreadable filename pairs: {skipped_masks}
- Leaf palette entries: {len(palette)}
- Leaf tracks with at least four visible frames: {total_tracks}
- Mask image modes: {dict(mask_modes)}

## Interpretation

Areas are reported in **pixels**, not cm².

Leaf identity is provisionally inferred from the official RGB
palette. A color-to-leaf match alone does not demonstrate
biologically correct identity tracking.

Non-palette pixels include background and potentially
unrecognized annotations. They must be inspected before
any conclusion about annotation quality.

Sharp area drops and disappear/reappear events are diagnostic
flags. Occlusion, leaf movement, annotation artifacts, and
real biological changes may all contribute.

## Limitations

This script samples at most {MAX_PLANTS_PER_GENOTYPE}
plants per genotype.

It does not establish forecast accuracy, genotype-disjoint
generalization, or calibrated uncertainty.

It does not validate that 'rep_*' groups represent independent
biological individuals.

## Next decision

Inspect decoded leaf trajectories and mask color distribution.
Then audit all 318 plants before freezing forecast splits.
"""

    (REPORTS / "E0_2_leaf_trajectory_audit.md").write_text(
        report,
        encoding="utf-8"
    )

    # Optional visualization
    try:
        import matplotlib.pyplot as plt

        if trajectory_rows:
            example_plant = trajectory_rows[0]["plant_id"]
            tracks = defaultdict(list)

            for row in trajectory_rows:
                if row["plant_id"] == example_plant:
                    tracks[row["leaf_id"]].append(row)

            fig, ax = plt.subplots(figsize=(11, 6))

            for leaf_id, rows in sorted(tracks.items()):
                if len(rows) < 4:
                    continue

                dates = [
                    datetime.fromisoformat(r["timestamp"])
                    for r in rows
                ]
                areas = [r["area_pixels"] for r in rows]

                ax.plot(
                    dates, areas, marker=".",
                    label=f"Leaf {leaf_id}"
                )

            ax.set_title(
                f"Individual-leaf area trajectories: {example_plant}"
            )
            ax.set_xlabel("Acquisition time")
            ax.set_ylabel("Visible projected area (pixels)")
            ax.legend(fontsize=8, ncol=3)
            fig.autofmt_xdate()
            fig.tight_layout()
            fig.savefig(
                FIGURES / "E0_2_example_leaf_trajectories.png",
                dpi=200
            )
            plt.close(fig)

    except ImportError:
        print("Matplotlib unavailable: plot skipped.")

    print("\n" + "=" * 62)
    print("E0.2 — SUMMARY")
    print("=" * 62)
    print("Genotypes sampled:", len(set(
        p["genotype"] for p in plant_rows
    )))
    print("Plants sampled:", len(plant_rows))
    print("Masks decoded:", total_masks)
    print("Leaf tracks (4+ visible frames):", total_tracks)
    print("Skipped mask pairs:", skipped_masks)
    print("Most common non-palette colors:")
    for rgb, count in unknown_colors.most_common(8):
        print("  ", rgb, count)

    print("\nSaved to:", OUT)
    print("E0.2 complete — review diagnostics before E0.3.")


if __name__ == "__main__":
    main()
