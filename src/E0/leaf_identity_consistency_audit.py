from pathlib import Path
from collections import Counter
import csv

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/PhenoLeaf-TS_DS"
LABELS = DATA / "02_leaf_labels"

EVENTS = (
    ROOT / "experiments/E0/tables"
    / "E0_7_suspicious_leaf_events.csv"
)

OUTPUT = (
    ROOT / "experiments/E0/tables"
    / "E0_9_identity_consistency.csv"
)

# Focus first on extreme examples from different plants.
MAX_EVENTS = 100
MIN_LARGE_REGION_PIXELS = 500
OTHER_COLOR_OVERLAP_THRESHOLD = 0.60


def pack_rgb(image):
    arr = np.asarray(image.convert("RGB"), dtype=np.uint32)

    return (
        (arr[:, :, 0] << 16)
        | (arr[:, :, 1] << 8)
        | arr[:, :, 2]
    )


def load_mask(path):
    with Image.open(path) as image:
        return pack_rgb(image)


def compare_regions(before, after):
    """
    Examine how pixels from a large colored region
    are labeled in the next frame.

    This is spatial consistency, NOT verified
    biological leaf tracking.
    """
    before_values, before_counts = np.unique(
        before, return_counts=True
    )

    candidates = [
        (int(color), int(count))
        for color, count in zip(
            before_values, before_counts
        )
        if color != 0 and count >= MIN_LARGE_REGION_PIXELS
    ]

    rows = []

    for color, pixels in candidates:
        region = before == color
        after_colors = after[region]

        nonbackground = after_colors[after_colors != 0]

        if nonbackground.size == 0:
            continue

        unique, counts = np.unique(
            nonbackground, return_counts=True
        )

        best_idx = int(np.argmax(counts))
        best_color = int(unique[best_idx])
        best_count = int(counts[best_idx])

        same_count = int(np.count_nonzero(
            after_colors == color
        ))

        best_overlap = best_count / pixels
        same_overlap = same_count / pixels

        rows.append({
            "before_color_hex": f"#{color:06X}",
            "best_after_color_hex": f"#{best_color:06X}",
            "before_pixels": pixels,
            "best_other_overlap": round(
                best_overlap, 4
            ),
            "same_color_overlap": round(
                same_overlap, 4
            ),
            "possible_reassignment": int(
                best_color != color
                and best_overlap >=
                    OTHER_COLOR_OVERLAP_THRESHOLD
            ),
        })

    return rows


def main():
    with EVENTS.open(newline="", encoding="utf-8") as f:
        events = list(csv.DictReader(f))

    # Compare the most severe events, but avoid having
    # every result come from only one plant.
    selected = []
    plant_counts = Counter()

    for event in events:
        plant = event["plant_id"]

        if plant_counts[plant] >= 3:
            continue

        selected.append(event)
        plant_counts[plant] += 1

        if len(selected) >= MAX_EVENTS:
            break

    output_rows = []

    for event in selected:
        plant = event["plant_id"]
        genotype, replicate = plant.split("/")

        folder = LABELS / genotype / replicate

        before = load_mask(
            folder / event["previous_filename"]
        )
        after = load_mask(
            folder / event["current_filename"]
        )

        if before.shape != after.shape:
            print("Skipping shape mismatch:", plant)
            continue

        comparisons = compare_regions(
            before, after
        )

        for result in comparisons:
            output_rows.append({
                "plant_id": plant,
                "event_leaf_id": event["leaf_id"],
                "event_type": event["event"],
                "previous_filename": event[
                    "previous_filename"
                ],
                "current_filename": event[
                    "current_filename"
                ],
                **result,
            })

    columns = [
        "plant_id",
        "event_leaf_id",
        "event_type",
        "previous_filename",
        "current_filename",
        "before_color_hex",
        "best_after_color_hex",
        "before_pixels",
        "best_other_overlap",
        "same_color_overlap",
        "possible_reassignment",
    ]

    OUTPUT.parent.mkdir(
        parents=True, exist_ok=True
    )

    with OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(output_rows)

    flagged = [
        row for row in output_rows
        if row["possible_reassignment"]
    ]

    affected_plants = {
        row["plant_id"] for row in flagged
    }

    print("=" * 65)
    print("E0.9 — TEMPORAL LEAF IDENTITY CONSISTENCY")
    print("=" * 65)
    print("Event transitions examined:", len(selected))
    print("Large color regions examined:", len(output_rows))
    print("Potential reassignment signals:", len(flagged))
    print("Plants with signals:", len(affected_plants))

    print("\nFirst 10 flagged examples:")

    for row in flagged[:10]:
        print(
            row["plant_id"],
            row["before_color_hex"],
            "->",
            row["best_after_color_hex"],
            "overlap:",
            row["best_other_overlap"],
        )

    print("\nSaved:", OUTPUT)
    print("E0.9 complete.")


if __name__ == "__main__":
    main()
