
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
import csv

# ==========================================================
# E0.7 — LEAF AREA ANOMALY REVIEW
#
# Detect suspicious changes between adjacent image frames.
# Produce a ranked list for manual inspection.
#
# Diagnostic only: NO observations are excluded.
# ==========================================================

ROOT = Path(__file__).resolve().parents[2]

TABLES = ROOT / "experiments/E0/tables"
REPORTS = ROOT / "experiments/E0/reports"

PAIRING = TABLES / "E0_1_image_pairing.csv"
AREAS = TABLES / "E0_2_leaf_area_trajectories.csv"

OUT = TABLES / "E0_7_suspicious_leaf_events.csv"
SUMMARY = TABLES / "E0_7_event_summary.csv"
REPORT = REPORTS / "E0_7_anomaly_review.md"

DROP_RATIO = 0.5
INCREASE_RATIO = 2.0

# A small measured leaf can have very large percentage changes.
# Preserve the raw values for this reason.
TOP_EVENTS = 100


def load_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    print("=" * 65)
    print("E0.7 — LEAF AREA ANOMALY REVIEW")
    print("=" * 65)

    # Complete chronological frame index per plant.
    plants = defaultdict(list)

    for r in load_csv(PAIRING):
        if r["paired"] == "1" and r["timestamp"]:
            plants[r["plant_id"]].append((
                datetime.fromisoformat(r["timestamp"]),
                r["filename"]
            ))

    frame_info = {}

    for plant, frames in plants.items():
        frames.sort()

        for index, (time, filename) in enumerate(frames):
            frame_info[(plant, filename)] = (index, time)

    # Individual leaf observations.
    tracks = defaultdict(list)

    for r in load_csv(AREAS):
        plant = r["plant_id"]
        filename = r["filename"]
        key = (plant, filename)

        if key not in frame_info:
            raise ValueError(f"Missing frame: {key}")

        index, time = frame_info[key]

        tracks[(r["genotype"], plant, int(r["leaf_id"]))].append({
            "index": index,
            "time": time,
            "filename": filename,
            "pixels": int(r["area_pixels"])
        })

    events = []
    counts = Counter()

    for (genotype, plant, leaf_id), obs in tracks.items():
        obs.sort(key=lambda x: x["index"])

        for previous, current in zip(obs, obs[1:]):
            # Restrict to adjacent image frames.
            if current["index"] != previous["index"] + 1:
                continue

            old = previous["pixels"]
            new = current["pixels"]

            if old <= 0:
                continue

            ratio = new / old

            if ratio < DROP_RATIO:
                event = "SHARP_DECREASE"
            elif ratio > INCREASE_RATIO:
                event = "SHARP_INCREASE"
            else:
                continue

            gap_h = (
                current["time"] - previous["time"]
            ).total_seconds() / 3600

            # Symmetric multiplicative change magnitude.
            severity = max(ratio, 1.0 / ratio)

            row = {
                "genotype": genotype,
                "plant_id": plant,
                "leaf_id": leaf_id,
                "event": event,
                "previous_timestamp": previous["time"].isoformat(),
                "current_timestamp": current["time"].isoformat(),
                "time_gap_hours": round(gap_h, 3),
                "previous_area_pixels": old,
                "current_area_pixels": new,
                "area_ratio": round(ratio, 5),
                "absolute_change_pixels": new - old,
                "severity_factor": round(severity, 4),
                "previous_filename": previous["filename"],
                "current_filename": current["filename"],
                "flagged_palette_plant": int(
                    plant == "Is-1/rep_24"
                ),
            }

            events.append(row)
            counts[(genotype, event)] += 1

    events.sort(
        key=lambda r: r["severity_factor"],
        reverse=True
    )

    columns = [
        "genotype", "plant_id", "leaf_id", "event",
        "previous_timestamp", "current_timestamp",
        "time_gap_hours", "previous_area_pixels",
        "current_area_pixels", "area_ratio",
        "absolute_change_pixels", "severity_factor",
        "previous_filename", "current_filename",
        "flagged_palette_plant"
    ]

    save_csv(
        OUT,
        events,
        columns
    )

    summary_rows = []

    for genotype in sorted({k[0] for k in counts}):
        summary_rows.append({
            "genotype": genotype,
            "sharp_decreases": counts[
                (genotype, "SHARP_DECREASE")
            ],
            "sharp_increases": counts[
                (genotype, "SHARP_INCREASE")
            ],
        })

    save_csv(
        SUMMARY,
        summary_rows,
        ["genotype", "sharp_decreases", "sharp_increases"]
    )

    decrease_count = sum(
        r["event"] == "SHARP_DECREASE" for r in events
    )
    increase_count = len(events) - decrease_count

    report = [
        "# E0.7 — Leaf Area Anomaly Review",
        "",
        f"- Leaf tracks examined: {len(tracks)}",
        f"- Abrupt decrease events: {decrease_count}",
        f"- Abrupt increase events: {increase_count}",
        f"- Total suspicious adjacent-frame events: {len(events)}",
        "",
        "## Interpretation",
        "",
        "These events are candidates for visual inspection.",
        "They do not establish biological abnormalities or",
        "annotation errors.",
        "",
        "Relative changes are especially unstable for",
        "small leaf areas. Review absolute changes and",
        "elapsed time before deciding whether to flag",
        "an observation.",
        "",
        "The full event CSV contains both frame filenames.",
        "No observations have been excluded.",
    ]

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("Tracks examined:", len(tracks))
    print("Sharp decrease events:", decrease_count)
    print("Sharp increase events:", increase_count)
    print("Total suspicious events:", len(events))

    print("\nTop suspicious events:")

    for r in events[:10]:
        print(
            f"{r['plant_id']:20s} "
            f"leaf={r['leaf_id']:2d} "
            f"{r['event']:15s} "
            f"{r['previous_area_pixels']} -> "
            f"{r['current_area_pixels']} px "
            f"({r['time_gap_hours']}h)"
        )

    print("\nSaved:")
    print(OUT)
    print(SUMMARY)
    print(REPORT)
    print("\nE0.7 complete.")


if __name__ == "__main__":
    main()