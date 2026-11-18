from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv
import random

import numpy as np
from PIL import Image

# ============================================================
# E0.11 — UNBIASED LEAF-IDENTITY STABILITY AUDIT
#
# Randomly samples ordinary adjacent image pairs.
# Reuses the strong bidirectional matching definition
# from E0.10.
#
# No original data is modified.
# The results estimate a SPATIAL reassignment signal,
# NOT a verified biological tracking error rate.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data/raw/PhenoLeaf-TS_DS"
LABELS = DATA / "02_leaf_labels"

TABLES = ROOT / "experiments/E0/tables"
REPORTS = ROOT / "experiments/E0/reports"

PAIRING = TABLES / "E0_1_image_pairing.csv"

SELECTION_CSV = TABLES / "E0_11_sampled_transitions.csv"
MATCHES_CSV = TABLES / "E0_11_candidate_matches.csv"
SUMMARY_CSV = TABLES / "E0_11_genotype_summary.csv"
REPORT = REPORTS / "E0_11_identity_stability_report.md"

RANDOM_SEED = 20261108
MAX_TRANSITIONS_PER_GENOTYPE = 10
MAX_TRANSITIONS_PER_PLANT = 2

MIN_REGION_PIXELS = 150
MIN_IOU = 0.55
MAX_SAME_COLOR_IOU = 0.15
MIN_BIDIRECTIONAL_COVERAGE = 0.65
MAX_CENTROID_SHIFT = 0.12

rng = random.Random(RANDOM_SEED)


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
        arr = np.asarray(
            image.convert("RGB"),
            dtype=np.uint32
        )

    return (
        (arr[:, :, 0] << 16)
        | (arr[:, :, 1] << 8)
        | arr[:, :, 2]
    )


def sample_transitions():
    plants = defaultdict(list)

    for row in read_csv(PAIRING):
        if row["paired"] != "1" or not row["timestamp"]:
            continue

        plants[
            (row["genotype"], row["plant_id"])
        ].append((
            datetime.fromisoformat(row["timestamp"]),
            row["filename"]
        ))

    genotype_candidates = defaultdict(list)

    for (genotype, plant_id), frames in sorted(plants.items()):
        frames.sort()

        candidates = []

        for before, after in zip(frames, frames[1:]):
            candidates.append({
                "genotype": genotype,
                "plant_id": plant_id,
                "previous_filename": before[1],
                "current_filename": after[1],
                "gap_hours": round(
                    (after[0] - before[0]).total_seconds()
                    / 3600, 4
                )
            })

        # Random transitions selected WITHOUT looking
        # at leaf-area change or annotation content.
        picked = rng.sample(
            candidates,
            min(MAX_TRANSITIONS_PER_PLANT, len(candidates))
        )

        genotype_candidates[genotype].extend(picked)

    selected = []

    for genotype in sorted(genotype_candidates):
        candidates = genotype_candidates[genotype]

        selected.extend(
            rng.sample(
                candidates,
                min(
                    MAX_TRANSITIONS_PER_GENOTYPE,
                    len(candidates)
                )
            )
        )

    selected.sort(
        key=lambda r: (
            r["genotype"],
            r["plant_id"],
            r["previous_filename"]
        )
    )

    return selected


def get_region_stats(mask):
    stats = {}

    colors, counts = np.unique(
        mask, return_counts=True
    )

    for color, count in zip(colors, counts):
        color = int(color)
        count = int(count)

        if color == 0 or count < MIN_REGION_PIXELS:
            continue

        yy, xx = np.nonzero(mask == color)

        stats[color] = {
            "area": count,
            "cx": float(xx.mean()),
            "cy": float(yy.mean())
        }

    return stats


def find_strong_matches(before, after):
    if before.shape != after.shape:
        raise ValueError("Mask shapes differ")

    old_stats = get_region_stats(before)
    new_stats = get_region_stats(after)

    # Count intersections between every old/new color.
    pair_codes = (
        (before.astype(np.uint64) << 24)
        | after.astype(np.uint64)
    )

    codes, counts = np.unique(
        pair_codes, return_counts=True
    )

    diagonal = np.hypot(*before.shape)

    candidates = []

    for code, count in zip(codes, counts):
        code = int(code)

        old = code >> 24
        new = code & 0xFFFFFF

        if old not in old_stats or new not in new_stats:
            continue

        intersection = int(count)

        old_area = old_stats[old]["area"]
        new_area = new_stats[new]["area"]

        union = old_area + new_area - intersection
        if union <= 0:
            continue

        dx = old_stats[old]["cx"] - new_stats[new]["cx"]
        dy = old_stats[old]["cy"] - new_stats[new]["cy"]

        candidates.append({
            "old": old,
            "new": new,
            "iou": intersection / union,
            "old_coverage": intersection / old_area,
            "new_coverage": intersection / new_area,
            "centroid_shift": np.hypot(dx, dy) / diagonal,
            "old_area": old_area,
            "new_area": new_area
        })

    best_old = {}
    best_new = {}
    same_iou = {}

    for match in candidates:
        old = match["old"]
        new = match["new"]

        if (
            old not in best_old
            or match["iou"] > best_old[old]["iou"]
        ):
            best_old[old] = match

        if (
            new not in best_new
            or match["iou"] > best_new[new]["iou"]
        ):
            best_new[new] = match

        if old == new:
            same_iou[old] = match["iou"]

    strong = []

    for match in candidates:
        old = match["old"]
        new = match["new"]

        if old == new:
            continue

        if best_old.get(old) is not match:
            continue

        if best_new.get(new) is not match:
            continue

        if match["iou"] < MIN_IOU:
            continue

        if same_iou.get(old, 0) > MAX_SAME_COLOR_IOU:
            continue

        if (
            match["old_coverage"]
            < MIN_BIDIRECTIONAL_COVERAGE
        ):
            continue

        if (
            match["new_coverage"]
            < MIN_BIDIRECTIONAL_COVERAGE
        ):
            continue

        if match["centroid_shift"] > MAX_CENTROID_SHIFT:
            continue

        strong.append({
            "old_color": f"#{old:06X}",
            "new_color": f"#{new:06X}",
            "cross_color_iou": round(match["iou"], 5),
            "same_color_iou": round(
                same_iou.get(old, 0), 5
            ),
            "old_area": match["old_area"],
            "new_area": match["new_area"]
        })

    return strong


def main():
    print("=" * 70)
    print("E0.11 — RANDOM-SAMPLE IDENTITY STABILITY AUDIT")
    print("=" * 70)

    selected = sample_transitions()
    print("Transitions sampled:", len(selected))

    selection_rows = []
    match_rows = []

    for i, transition in enumerate(selected, start=1):
        plant = transition["plant_id"]
        genotype, replicate = plant.split("/")

        folder = LABELS / genotype / replicate

        before = load_mask(
            folder / transition["previous_filename"]
        )

        after = load_mask(
            folder / transition["current_filename"]
        )

        matches = find_strong_matches(before, after)

        selection_rows.append({
            **transition,
            "strong_matches": len(matches),
            "has_strong_match": int(bool(matches))
        })

        for match in matches:
            match_rows.append({
                **transition,
                **match
            })

        if i % 25 == 0:
            print(f"Processed {i}/{len(selected)}")

    write_csv(
        SELECTION_CSV,
        selection_rows,
        [
            "genotype", "plant_id",
            "previous_filename", "current_filename",
            "gap_hours", "strong_matches",
            "has_strong_match"
        ]
    )

    write_csv(
        MATCHES_CSV,
        match_rows,
        [
            "genotype", "plant_id",
            "previous_filename", "current_filename",
            "gap_hours", "old_color", "new_color",
            "cross_color_iou", "same_color_iou",
            "old_area", "new_area"
        ]
    )

    by_genotype = defaultdict(list)

    for row in selection_rows:
        by_genotype[row["genotype"]].append(row)

    summary = []

    for genotype, records in sorted(by_genotype.items()):
        flagged = sum(
            r["has_strong_match"] for r in records
        )

        summary.append({
            "genotype": genotype,
            "transitions_sampled": len(records),
            "transitions_flagged": flagged,
            "flagged_fraction": round(
                flagged / len(records), 5
            ),
            "plants_sampled": len({
                r["plant_id"] for r in records
            }),
            "plants_flagged": len({
                r["plant_id"]
                for r in records
                if r["has_strong_match"]
            })
        })

    write_csv(
        SUMMARY_CSV,
        summary,
        [
            "genotype",
            "transitions_sampled",
            "transitions_flagged",
            "flagged_fraction",
            "plants_sampled",
            "plants_flagged"
        ]
    )

    n_flagged = sum(
        r["has_strong_match"] for r in selection_rows
    )

    print("\n" + "=" * 70)
    print("E0.11 — RESULTS")
    print("=" * 70)

    print("Sampled transitions:", len(selected))
    print("Transitions with strong signals:", n_flagged)
    print("Strong color matches:", len(match_rows))

    if selected:
        print(
            "Sampled-transition flagged fraction:",
            f"{100 * n_flagged / len(selected):.2f}%"
        )

    print("\nGenotype breakdown:")
    print(
        f"{'Genotype':14s} "
        f"{'Sampled':>8s} "
        f"{'Flagged':>8s} "
        f"{'Plants':>8s}"
    )

    for row in summary:
        print(
            f"{row['genotype']:14s} "
            f"{row['transitions_sampled']:8d} "
            f"{row['transitions_flagged']:8d} "
            f"{row['plants_sampled']:8d}"
        )

    report = [
        "# E0.11 — Random-Sample Identity Stability Audit",
        "",
        "## Sampling protocol",
        "",
        f"- Fixed random seed: {RANDOM_SEED}",
        f"- Maximum transitions per genotype: "
        f"{MAX_TRANSITIONS_PER_GENOTYPE}",
        f"- Maximum transitions per plant: "
        f"{MAX_TRANSITIONS_PER_PLANT}",
        "- Selection does not use area anomalies.",
        "",
        "## Results",
        "",
        f"- Transitions sampled: {len(selected)}",
        f"- Transitions with strong spatial signals: {n_flagged}",
        f"- Strong color matches: {len(match_rows)}",
        "",
        "## Critical limitations",
        "",
        "This is a bounded, stratified random sample, "
        "not a plant- or frame-proportional census.",
        "",
        "The sampled fraction is not an unbiased estimator "
        "of the overall frame-weighted dataset rate.",
        "",
        "Strong spatial signals require manual review.",
        "",
        "Results alone do not establish biological identity.",
        "",
        "We must verify the original palette semantics "
        "and the independence of replicate folders.",
        "",
        "No corrections or exclusions were applied.",
    ]

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("\nSaved:", SELECTION_CSV)
    print("Saved:", MATCHES_CSV)
    print("Saved:", SUMMARY_CSV)
    print("Saved:", REPORT)


if __name__ == "__main__":
    main()