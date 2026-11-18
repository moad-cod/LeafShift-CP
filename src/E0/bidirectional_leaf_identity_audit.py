
from pathlib import Path
from collections import defaultdict
import csv

import numpy as np
from PIL import Image

# ============================================================
# E0.10 — BIDIRECTIONAL LEAF IDENTITY VALIDATION
#
# Re-examines the E0.9 selected image transitions.
# Read-only: original images and masks are untouched.
#
# A strong flag is NOT a confirmed biological ID error.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

LABELS = (
    ROOT / "data/raw/PhenoLeaf-TS_DS/02_leaf_labels"
)

TABLES = ROOT / "experiments/E0/tables"
REPORTS = ROOT / "experiments/E0/reports"

INPUT = TABLES / "E0_9_identity_consistency.csv"
OUTPUT = TABLES / "E0_10_bidirectional_matches.csv"
SUMMARY = TABLES / "E0_10_transition_summary.csv"
REPORT = REPORTS / "E0_10_identity_validation.md"

# Diagnostic thresholds — not final exclusion rules.
MIN_REGION_PIXELS = 150
MIN_IOU = 0.55
MAX_SAME_COLOR_IOU = 0.15
MIN_BIDIRECTIONAL_COVERAGE = 0.65
MAX_NORMALIZED_CENTROID_SHIFT = 0.12


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def load_mask(path):
    with Image.open(path) as image:
        rgb = np.asarray(
            image.convert("RGB"), dtype=np.uint32
        )

    return (
        (rgb[:, :, 0] << 16)
        | (rgb[:, :, 1] << 8)
        | rgb[:, :, 2]
    )


def region_stats(mask):
    stats = {}

    colors, counts = np.unique(
        mask, return_counts=True
    )

    for color, area in zip(colors, counts):
        color = int(color)
        area = int(area)

        if color == 0 or area < MIN_REGION_PIXELS:
            continue

        yy, xx = np.nonzero(mask == color)

        stats[color] = {
            "area": area,
            "cx": float(xx.mean()),
            "cy": float(yy.mean()),
        }

    return stats


def overlap_matrix(before, after):
    """
    Count pixel intersections for all color pairs.
    Uses packed 24-bit RGB labels, not RGB proximity.
    """
    pairs = (
        (before.astype(np.uint64) << 24)
        | after.astype(np.uint64)
    )

    codes, counts = np.unique(
        pairs, return_counts=True
    )

    intersections = {}

    for code, count in zip(codes, counts):
        code = int(code)
        old = code >> 24
        new = code & 0xFFFFFF

        if old != 0 and new != 0:
            intersections[(old, new)] = int(count)

    return intersections


def compare_transition(before, after):
    if before.shape != after.shape:
        raise ValueError("Mask dimensions differ")

    old_stats = region_stats(before)
    new_stats = region_stats(after)
    intersections = overlap_matrix(before, after)

    height, width = before.shape
    diagonal = (width**2 + height**2) ** 0.5

    candidates = []

    # Evaluate every pair of sufficiently large regions.
    for (old, new), intersection in intersections.items():
        if old not in old_stats or new not in new_stats:
            continue

        old_area = old_stats[old]["area"]
        new_area = new_stats[new]["area"]

        union = old_area + new_area - intersection

        if union <= 0:
            continue

        iou = intersection / union

        dx = (
            old_stats[old]["cx"]
            - new_stats[new]["cx"]
        )
        dy = (
            old_stats[old]["cy"]
            - new_stats[new]["cy"]
        )

        shift = (dx**2 + dy**2) ** 0.5 / diagonal

        candidates.append({
            "old": old,
            "new": new,
            "old_area": old_area,
            "new_area": new_area,
            "intersection": intersection,
            "iou": iou,
            "old_coverage": intersection / old_area,
            "new_coverage": intersection / new_area,
            "centroid_shift": shift,
        })

    # Best IoU match for each old region and each new region.
    best_new_for_old = {}
    best_old_for_new = {}

    for match in candidates:
        old = match["old"]
        new = match["new"]

        if (
            old not in best_new_for_old
            or match["iou"] >
            best_new_for_old[old]["iou"]
        ):
            best_new_for_old[old] = match

        if (
            new not in best_old_for_new
            or match["iou"] >
            best_old_for_new[new]["iou"]
        ):
            best_old_for_new[new] = match

    # Same-color IoU provides an important control.
    same_iou = {
        m["old"]: m["iou"]
        for m in candidates
        if m["old"] == m["new"]
    }

    rows = []

    for match in candidates:
        old = match["old"]
        new = match["new"]

        if old == new:
            continue

        mutual = (
            best_new_for_old.get(old) is match
            and best_old_for_new.get(new) is match
        )

        original_iou = same_iou.get(old, 0.0)

        strong = (
            mutual
            and match["iou"] >= MIN_IOU
            and original_iou <= MAX_SAME_COLOR_IOU
            and match["old_coverage"] >=
                MIN_BIDIRECTIONAL_COVERAGE
            and match["new_coverage"] >=
                MIN_BIDIRECTIONAL_COVERAGE
            and match["centroid_shift"] <=
                MAX_NORMALIZED_CENTROID_SHIFT
        )

        if not strong:
            continue

        rows.append({
            "old_color": f"#{old:06X}",
            "new_color": f"#{new:06X}",
            "old_area_pixels": match["old_area"],
            "new_area_pixels": match["new_area"],
            "intersection_pixels": match["intersection"],
            "cross_color_iou": round(
                match["iou"], 5
            ),
            "same_color_iou": round(
                original_iou, 5
            ),
            "old_region_coverage": round(
                match["old_coverage"], 5
            ),
            "new_region_coverage": round(
                match["new_coverage"], 5
            ),
            "normalized_centroid_shift": round(
                match["centroid_shift"], 5
            ),
            "mutual_best_match": 1,
            "strong_candidate": 1,
        })

    return rows


def main():
    print("=" * 66)
    print("E0.10 — BIDIRECTIONAL LEAF IDENTITY VALIDATION")
    print("=" * 66)

    input_rows = read_csv(INPUT)

    # E0.9 contains multiple leaf-region records for
    # each before/after image transition.
    transitions = {}

    for row in input_rows:
        key = (
            row["plant_id"],
            row["previous_filename"],
            row["current_filename"],
        )
        transitions[key] = {
            "plant_id": row["plant_id"],
            "previous_filename": row["previous_filename"],
            "current_filename": row["current_filename"],
        }

    results = []
    transition_summary = []

    for key, item in sorted(transitions.items()):
        plant = item["plant_id"]
        genotype, replicate = plant.split("/")

        folder = LABELS / genotype / replicate

        before = load_mask(
            folder / item["previous_filename"]
        )
        after = load_mask(
            folder / item["current_filename"]
        )

        matches = compare_transition(before, after)

        transition_summary.append({
            **item,
            "strong_candidate_matches": len(matches),
            "has_strong_candidate": int(
                len(matches) > 0
            ),
        })

        for match in matches:
            results.append({
                **item,
                **match,
            })

    match_columns = [
        "plant_id",
        "previous_filename",
        "current_filename",
        "old_color",
        "new_color",
        "old_area_pixels",
        "new_area_pixels",
        "intersection_pixels",
        "cross_color_iou",
        "same_color_iou",
        "old_region_coverage",
        "new_region_coverage",
        "normalized_centroid_shift",
        "mutual_best_match",
        "strong_candidate",
    ]

    write_csv(OUTPUT, results, match_columns)

    write_csv(
        SUMMARY,
        transition_summary,
        [
            "plant_id",
            "previous_filename",
            "current_filename",
            "strong_candidate_matches",
            "has_strong_candidate",
        ],
    )

    affected_transitions = sum(
        r["has_strong_candidate"]
        for r in transition_summary
    )

    affected_plants = {
        r["plant_id"] for r in results
    }

    print("Unique transitions:", len(transitions))
    print("Strong candidate color matches:", len(results))
    print("Transitions with candidates:", affected_transitions)
    print("Plants with candidates:", len(affected_plants))

    print("\nTop candidate matches:")

    ranked = sorted(
        results,
        key=lambda r: r["cross_color_iou"],
        reverse=True,
    )

    for row in ranked[:15]:
        print(
            f"{row['plant_id']:20s} "
            f"{row['old_color']} -> "
            f"{row['new_color']} "
            f"IoU={row['cross_color_iou']:.3f} "
            f"same={row['same_color_iou']:.3f}"
        )

    report = [
        "# E0.10 — Bidirectional Leaf Identity Audit",
        "",
        f"- Unique selected transitions: {len(transitions)}",
        f"- Strong candidate matches: {len(results)}",
        f"- Transitions with candidates: {affected_transitions}",
        f"- Plants with candidates: {len(affected_plants)}",
        "",
        "## Method",
        "",
        "Candidates require reciprocal best spatial IoU, "
        "high bidirectional coverage, low same-color IoU, "
        "and limited centroid displacement.",
        "",
        "## Limitations",
        "",
        "This is a targeted sample selected from anomalous "
        "area transitions, not an unbiased dataset sample.",
        "",
        "High spatial consistency under a changed color "
        "suggests annotation reassignment but does not "
        "confirm biological identity.",
        "",
        "No automatic correction, interpolation, or exclusion "
        "has been applied.",
        "",
        "Visual verification and annotation documentation "
        "are still required before formal tracking claims.",
    ]

    REPORTS.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    print("\nSaved:", OUTPUT)
    print("Saved:", SUMMARY)
    print("Saved:", REPORT)
    print("\nE0.10 complete.")


if __name__ == "__main__":
    main()
