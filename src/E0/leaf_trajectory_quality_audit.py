
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from statistics import median
import csv

# ============================================================
# E0.6 — LEAF TRAJECTORY QUALITY AUDIT
#
# Input: E0.1 image manifest and E0.2 leaf trajectories
# Does not read or modify original dataset images.
#
# Purpose:
#   Quantify missing observations and large area changes
#   before choosing forecasting targets and QC rules.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "experiments/E0/tables"
REPORTS = ROOT / "experiments/E0/reports"

PAIRING = TABLES / "E0_1_image_pairing.csv"
AREAS = TABLES / "E0_2_leaf_area_trajectories.csv"

FLAGGED_PLANT = "Is-1/rep_24"

# Diagnostic thresholds, not exclusion criteria.
SHARP_DROP_RATIO = 0.50
SHARP_INCREASE_RATIO = 2.00
MIN_TRACK_OBSERVATIONS = 4


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def safe_median(values):
    return round(median(values), 4) if values else ""


def main():
    print("=" * 66)
    print("E0.6 — LEAF TRAJECTORY QUALITY AUDIT")
    print("=" * 66)

    # ----------------------------------------
    # 1. Build each plant's full image timeline
    # ----------------------------------------

    plant_frames = defaultdict(list)

    for row in read_csv(PAIRING):
        if row["paired"] != "1" or not row["timestamp"]:
            continue

        plant_frames[row["plant_id"]].append(
            (
                datetime.fromisoformat(row["timestamp"]),
                row["filename"],
            )
        )

    frame_index = {}

    for plant_id, frames in plant_frames.items():
        frames.sort()

        for index, (_, filename) in enumerate(frames):
            frame_index[(plant_id, filename)] = index

    # ----------------------------------------
    # 2. Reconstruct individual leaf tracks
    # ----------------------------------------

    tracks = defaultdict(list)

    for row in read_csv(AREAS):
        key = (
            row["plant_id"],
            row["filename"],
        )

        if key not in frame_index:
            raise ValueError(
                f"Unmatched observation: {key}"
            )

        track_key = (
            row["genotype"],
            row["plant_id"],
            int(row["leaf_id"]),
        )

        tracks[track_key].append({
            "index": frame_index[key],
            "timestamp": datetime.fromisoformat(
                row["timestamp"]
            ),
            "pixels": int(row["area_pixels"]),
        })

    # ----------------------------------------
    # 3. Quantify trajectory quality
    # ----------------------------------------

    track_rows = []

    for (genotype, plant_id, leaf_id), observations in (
        sorted(tracks.items())
    ):
        observations.sort(
            key=lambda x: x["index"]
        )

        if len(observations) < MIN_TRACK_OBSERVATIONS:
            continue

        indices = [x["index"] for x in observations]
        areas = [x["pixels"] for x in observations]

        first_index = indices[0]
        last_index = indices[-1]

        expected_frames = (
            last_index - first_index + 1
        )

        observed_frames = len(observations)

        missing_frames = max(
            0, expected_frames - observed_frames
        )

        coverage = (
            observed_frames / expected_frames
            if expected_frames else 0
        )

        # Compare areas only when observations
        # occur in adjacent image frames.
        adjacent_pairs = []
        observed_gaps = []

        for before, after in zip(
            observations, observations[1:]
        ):
            observed_gaps.append(
                (
                    after["timestamp"]
                    - before["timestamp"]
                ).total_seconds() / 3600
            )

            if after["index"] == before["index"] + 1:
                adjacent_pairs.append(
                    (before["pixels"], after["pixels"])
                )

        sharp_drops = sum(
            current < previous * SHARP_DROP_RATIO
            for previous, current in adjacent_pairs
        )

        sharp_increases = sum(
            current > previous * SHARP_INCREASE_RATIO
            for previous, current in adjacent_pairs
        )

        # The number of separate visible runs.
        visible_runs = 1 + sum(
            b > a + 1
            for a, b in zip(indices, indices[1:])
        )

        duration_hours = (
            observations[-1]["timestamp"]
            - observations[0]["timestamp"]
        ).total_seconds() / 3600

        track_rows.append({
            "genotype": genotype,
            "plant_id": plant_id,
            "leaf_id": leaf_id,
            "observed_frames": observed_frames,
            "frames_first_to_last": expected_frames,
            "missing_internal_frames": missing_frames,
            "observation_coverage": round(coverage, 5),
            "visible_runs": visible_runs,
            "adjacent_pairs": len(adjacent_pairs),
            "sharp_drops": sharp_drops,
            "sharp_increases": sharp_increases,
            "duration_hours": round(
                duration_hours, 3
            ),
            "median_observed_gap_hours": (
                safe_median(observed_gaps)
            ),
            "median_area_pixels": (
                safe_median(areas)
            ),
            "flagged_annotation_plant": int(
                plant_id == FLAGGED_PLANT
            ),
        })

    # ----------------------------------------
    # 4. Aggregate per genotype
    # ----------------------------------------

    by_genotype = defaultdict(list)

    for row in track_rows:
        by_genotype[row["genotype"]].append(row)

    genotype_rows = []

    for genotype, rows in sorted(by_genotype.items()):
        total_adjacent = sum(
            r["adjacent_pairs"] for r in rows
        )

        total_drops = sum(
            r["sharp_drops"] for r in rows
        )

        total_increases = sum(
            r["sharp_increases"] for r in rows
        )

        total_expected = sum(
            r["frames_first_to_last"] for r in rows
        )

        total_observed = sum(
            r["observed_frames"] for r in rows
        )

        genotype_rows.append({
            "genotype": genotype,
            "plants_with_tracks": len({
                r["plant_id"] for r in rows
            }),
            "leaf_tracks": len(rows),
            "median_track_coverage": safe_median([
                r["observation_coverage"] for r in rows
            ]),
            "pooled_observation_coverage": round(
                total_observed / total_expected, 5
            ) if total_expected else "",
            "tracks_with_missing_frames": sum(
                r["missing_internal_frames"] > 0
                for r in rows
            ),
            "tracks_with_sharp_drops": sum(
                r["sharp_drops"] > 0
                for r in rows
            ),
            "tracks_with_sharp_increases": sum(
                r["sharp_increases"] > 0
                for r in rows
            ),
            "adjacent_pairs": total_adjacent,
            "sharp_drops": total_drops,
            "sharp_increases": total_increases,
            "sharp_drop_rate": round(
                total_drops / total_adjacent, 5
            ) if total_adjacent else "",
            "sharp_increase_rate": round(
                total_increases / total_adjacent, 5
            ) if total_adjacent else "",
        })

    # ----------------------------------------
    # 5. Save tables
    # ----------------------------------------

    save_csv(
        TABLES / "E0_6_leaf_quality.csv",
        track_rows,
        list(track_rows[0].keys())
    )

    save_csv(
        TABLES / "E0_6_genotype_quality.csv",
        genotype_rows,
        list(genotype_rows[0].keys())
    )

    # ----------------------------------------
    # 6. Summary
    # ----------------------------------------

    n_tracks = len(track_rows)

    tracks_with_missing = sum(
        r["missing_internal_frames"] > 0
        for r in track_rows
    )

    tracks_with_drops = sum(
        r["sharp_drops"] > 0
        for r in track_rows
    )

    tracks_with_increases = sum(
        r["sharp_increases"] > 0
        for r in track_rows
    )

    print("\nOverall quality:")
    print("Candidate tracks:", n_tracks)
    print(
        "Tracks with missing internal frames:",
        tracks_with_missing
    )
    print(
        "Tracks with sharp area decreases:",
        tracks_with_drops
    )
    print(
        "Tracks with sharp area increases:",
        tracks_with_increases
    )

    print("\nGenotype quality:")
    print(
        f"{'Genotype':13s} {'Plants':>6s} "
        f"{'Tracks':>7s} {'Coverage':>9s} "
        f"{'Drop %':>8s} {'Missing':>8s}"
    )

    for r in genotype_rows:
        print(
            f"{r['genotype']:13s} "
            f"{r['plants_with_tracks']:6d} "
            f"{r['leaf_tracks']:7d} "
            f"{r['median_track_coverage']:9.3f} "
            f"{100*r['sharp_drop_rate']:7.2f}% "
            f"{r['tracks_with_missing_frames']:8d}"
        )

    # ----------------------------------------
    # 7. Research report
    # ----------------------------------------

    report = [
        "# E0.6 — Individual-Leaf Trajectory Quality",
        "",
        "## Results",
        "",
        f"- Candidate tracks: {n_tracks}",
        f"- Tracks with internal missing frames: "
        f"{tracks_with_missing}",
        f"- Tracks with sharp decreases: "
        f"{tracks_with_drops}",
        f"- Tracks with sharp increases: "
        f"{tracks_with_increases}",
        "",
        "## Definitions",
        "",
        "- Missing: no >=20-pixel observation in an "
        "image frame between first and last detection.",
        "- Sharp decrease: current area below 50% of "
        "previous area in adjacent image frames.",
        "- Sharp increase: current area above twice "
        "previous area in adjacent image frames.",
        "- Coverage: observed / available image frames "
        "between first and last detection.",
        "",
        "## Scientific limitations",
        "",
        "These are diagnostic flags, not confirmed "
        "segmentation errors.",
        "",
        "Apparent area decreases can result from "
        "occlusion, leaf movement or annotation changes.",
        "",
        "The outcome is visible projected area in pixels, "
        "not occlusion-corrected true leaf area.",
        "",
        "These analyses do not independently confirm "
        "the biological validity of leaf identities.",
        "",
        "Do not remove flagged observations without "
        "visual inspection and a documented QC policy.",
        "",
        "## Next",
        "",
        "Review high-risk tracks and validate a "
        "genotype-held-out experimental protocol.",
    ]

    REPORTS.mkdir(parents=True, exist_ok=True)

    (
        REPORTS / "E0_6_trajectory_quality_report.md"
    ).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("\nSaved E0.6 tables and report.")
    print("E0.6 complete.")


if __name__ == "__main__":
    main()
