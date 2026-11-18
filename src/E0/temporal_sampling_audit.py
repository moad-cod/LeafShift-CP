
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime
from statistics import median
import csv

# ============================================================
# E0.4 — TEMPORAL SAMPLING DIAGNOSIS
#
# Goal:
#   Explain why only 8/21 genotypes qualified for
#   six-hour forecasting in E0.3.
#
# Inputs: existing E0.1 and E0.3 CSV outputs
# No raw images are read or modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "experiments/E0/tables"
REPORTS = ROOT / "experiments/E0/reports"

PAIRING = TABLES / "E0_1_image_pairing.csv"
ELIGIBILITY = TABLES / "E0_3_plant_forecast_eligibility.csv"

OUTPUT = TABLES / "E0_4_temporal_sampling.csv"
REPORT = REPORTS / "E0_4_temporal_sampling_report.md"

HISTORY_FRAMES = 6
MIN_ACCEPTED_GAP = 3.0
MAX_ACCEPTED_GAP = 9.0


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)


def gap_category(hours):
    if hours < 3:
        return "under_3h"
    if hours <= 9:
        return "3_to_9h"
    if hours <= 15:
        return "over_9_to_15h"
    if hours <= 30:
        return "over_15_to_30h"
    return "over_30h"


def longest_regular_run(timestamps):
    """
    Longest run of frames connected by adjacent
    gaps in the 3–9 hour range.
    """
    if not timestamps:
        return 0

    longest = 1
    current = 1

    for first, second in zip(
        timestamps, timestamps[1:]
    ):
        gap = (second - first).total_seconds() / 3600

        if MIN_ACCEPTED_GAP <= gap <= MAX_ACCEPTED_GAP:
            current += 1
        else:
            longest = max(longest, current)
            current = 1

    return max(longest, current)


def main():
    plants = defaultdict(list)

    for row in read_csv(PAIRING):
        if row["paired"] != "1" or not row["timestamp"]:
            continue

        plants[
            (row["genotype"], row["plant_id"])
        ].append(datetime.fromisoformat(row["timestamp"]))

    eligibility = {}

    for row in read_csv(ELIGIBILITY):
        if int(row["horizon_frames"]) == 1:
            eligibility[row["plant_id"]] = row

    grouped = defaultdict(list)

    for (genotype, plant_id), timestamps in plants.items():
        timestamps = sorted(timestamps)

        gaps = [
            (b - a).total_seconds() / 3600
            for a, b in zip(timestamps, timestamps[1:])
        ]

        run = longest_regular_run(timestamps)

        grouped[genotype].append({
            "plant_id": plant_id,
            "frame_count": len(timestamps),
            "gaps": gaps,
            "longest_regular_run": run,
            "e0_3_eligible": int(
                eligibility.get(
                    plant_id, {}
                ).get("strict_plant_eligible", 0)
            ),
        })

    rows = []

    for genotype, plant_records in sorted(grouped.items()):
        all_gaps = [
            gap
            for record in plant_records
            for gap in record["gaps"]
        ]

        categories = Counter(
            gap_category(gap) for gap in all_gaps
        )

        runs = [
            record["longest_regular_run"]
            for record in plant_records
        ]

        counts = [
            record["frame_count"]
            for record in plant_records
        ]

        eligible = sum(
            record["e0_3_eligible"]
            for record in plant_records
        )

        row = {
            "genotype": genotype,
            "plants": len(plant_records),
            "e0_3_eligible_plants": eligible,
            "median_frames": median(counts),
            "median_gap_hours": (
                round(median(all_gaps), 3)
                if all_gaps else ""
            ),
            "min_gap_hours": (
                round(min(all_gaps), 3)
                if all_gaps else ""
            ),
            "max_gap_hours": (
                round(max(all_gaps), 3)
                if all_gaps else ""
            ),
            "fraction_gaps_3_to_9h": (
                round(
                    categories["3_to_9h"] / len(all_gaps),
                    4
                ) if all_gaps else 0
            ),
            "gaps_under_3h": categories["under_3h"],
            "gaps_3_to_9h": categories["3_to_9h"],
            "gaps_9_to_15h": categories["over_9_to_15h"],
            "gaps_15_to_30h": categories["over_15_to_30h"],
            "gaps_over_30h": categories["over_30h"],
            "median_longest_regular_run": median(runs),
            "plants_with_7plus_regular_frames": sum(
                run >= HISTORY_FRAMES + 1
                for run in runs
            ),
            "plants_with_14plus_regular_frames": sum(
                run >= HISTORY_FRAMES + 8
                for run in runs
            ),
        }

        rows.append(row)

    write_csv(OUTPUT, rows)

    header = (
        f"{'Genotype':13s} "
        f"{'Plants':>6s} "
        f"{'Eligible':>8s} "
        f"{'Gap(h)':>9s} "
        f"{'3-9h %':>8s} "
        f"{'Run':>7s}"
    )

    print("=" * 70)
    print("E0.4 — TEMPORAL SAMPLING DIAGNOSIS")
    print("=" * 70)
    print(header)
    print("-" * 70)

    for r in rows:
        gap = r["median_gap_hours"]

        print(
            f"{r['genotype']:13s} "
            f"{r['plants']:6d} "
            f"{r['e0_3_eligible_plants']:8d} "
            f"{str(gap):>9s} "
            f"{100*r['fraction_gaps_3_to_9h']:7.1f}% "
            f"{r['median_longest_regular_run']:7.1f}"
        )

    missing = [
        r for r in rows
        if r["e0_3_eligible_plants"] == 0
    ]

    report = [
        "# E0.4 — Temporal Sampling Diagnosis",
        "",
        "## Objective",
        "Explain the exclusion of 13 genotypes in E0.3.",
        "",
        "## Results",
        "",
        "| Genotype | Plants | E0.3 eligible | "
        "Median gap (h) | 3–9h gaps (%) | "
        "Median longest regular run |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    for r in rows:
        report.append(
            f"| {r['genotype']} "
            f"| {r['plants']} "
            f"| {r['e0_3_eligible_plants']} "
            f"| {r['median_gap_hours']} "
            f"| {100*r['fraction_gaps_3_to_9h']:.1f} "
            f"| {r['median_longest_regular_run']} |"
        )

    report.extend([
        "",
        "## Genotypes with zero E0.3 eligibility",
        "",
    ])

    report.extend(
        f"- {r['genotype']}: median gap "
        f"{r['median_gap_hours']} hours"
        for r in missing
    )

    report.extend([
        "",
        "## Scientific caution",
        "",
        "This audit describes sampling patterns, "
        "not leaf-level biological validity.",
        "",
        "The E0.3 eligibility also depends on leaf "
        "visibility and annotation continuity.",
        "",
        "Do not finalize forecasting horizons or "
        "genotype splits before reviewing these results.",
    ])

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("\nGenotypes without E0.3 windows:")
    for r in missing:
        print(
            f"  {r['genotype']}: "
            f"median gap={r['median_gap_hours']}h"
        )

    print("\nSaved:", OUTPUT)
    print("Saved:", REPORT)


if __name__ == "__main__":
    main()
