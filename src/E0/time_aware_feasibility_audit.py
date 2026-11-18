
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from bisect import bisect_left
from statistics import median
import csv

# ============================================================
# E0.5 — TIME-AWARE FORECASTING FEASIBILITY
#
# Uses actual elapsed time instead of fixed frame steps.
# Reads existing E0.1 and E0.2 CSV tables.
# No model training. Raw dataset is not modified.
#
# Forecast target: observed VISIBLE leaf area in pixels.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TABLES = ROOT / "experiments/E0/tables"
REPORTS = ROOT / "experiments/E0/reports"
FIGURES = ROOT / "experiments/E0/figures"

for folder in (TABLES, REPORTS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

PAIRING = TABLES / "E0_1_image_pairing.csv"
AREAS = TABLES / "E0_2_leaf_area_trajectories.csv"

HISTORY_LENGTH = 6
MAX_HISTORY_HOURS = 72.0

HORIZONS_HOURS = (24, 48)
TARGET_TOLERANCE_HOURS = 3.0

MIN_AREA_PIXELS = 20

FLAGGED_PLANT = "Is-1/rep_24"


def read_rows(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(
        newline="", encoding="utf-8"
    ) as f:
        return list(csv.DictReader(f))


def save_csv(path, rows, columns):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def elapsed_hours(t1, t2):
    return (t2 - t1).total_seconds() / 3600.0


def load_data():
    """
    Build ordered image timestamps for each plant
    and attach leaf observations from E0.2.
    """

    plants = defaultdict(dict)

    for row in read_rows(PAIRING):
        if (
            row["paired"] != "1"
            or not row["timestamp"]
        ):
            continue

        pid = row["plant_id"]
        filename = row["filename"]

        plants[pid][filename] = {
            "genotype": row["genotype"],
            "timestamp": datetime.fromisoformat(
                row["timestamp"]
            ),
            "leaf_areas": {},
        }

    linked = 0

    for row in read_rows(AREAS):
        pid = row["plant_id"]
        filename = row["filename"]

        frame = plants.get(pid, {}).get(filename)

        if frame is None:
            raise RuntimeError(
                f"Unmatched area record: {pid}/{filename}"
            )

        pixels = int(row["area_pixels"])

        if pixels >= MIN_AREA_PIXELS:
            frame["leaf_areas"][
                int(row["leaf_id"])
            ] = pixels
            linked += 1

    sequences = {}

    for pid, frames in plants.items():
        seq = sorted(
            frames.values(),
            key=lambda x: x["timestamp"]
        )

        if len({
            x["timestamp"] for x in seq
        }) != len(seq):
            raise ValueError(
                f"Duplicate timestamps in {pid}"
            )

        sequences[pid] = seq

    print("Plants:", len(sequences))
    print("Linked leaf observations:", linked)

    return sequences


def eligible_windows_for_plant(plant_id, frames):
    """
    For each leaf, select six successive OBSERVED
    measurements in a history span <=72 hours.

    At the last historical measurement (cutoff),
    find an observed target near +24h or +48h.

    IMPORTANT:
    - Only observations <= cutoff enter the history.
    - No future information is used to select inputs.
    - Missing observations are not replaced with zero.
    - A leaf track can contribute overlapping windows.
    """

    genotype = frames[0]["genotype"]

    leaf_series = defaultdict(list)

    for frame in frames:
        for leaf_id, pixels in frame["leaf_areas"].items():
            leaf_series[leaf_id].append(
                (frame["timestamp"], pixels)
            )

    results = {
        h: {
            "windows": 0,
            "leaf_ids": set(),
            "cutoff_times": set(),
            "history_spans": [],
            "target_offsets": [],
            "examples": [],
        }
        for h in HORIZONS_HOURS
    }

    for leaf_id, series in leaf_series.items():
        if len(series) <= HISTORY_LENGTH:
            continue

        times = [x[0] for x in series]

        for cutoff_index in range(
            HISTORY_LENGTH - 1,
            len(series) - 1
        ):
            start_index = (
                cutoff_index - HISTORY_LENGTH + 1
            )

            history = series[
                start_index:cutoff_index + 1
            ]

            first_time = history[0][0]
            cutoff_time = history[-1][0]

            history_span = elapsed_hours(
                first_time, cutoff_time
            )

            if history_span > MAX_HISTORY_HOURS:
                continue

            for horizon in HORIZONS_HOURS:
                desired_seconds = (
                    cutoff_time.timestamp()
                    + horizon * 3600
                )

                # Find the nearest observed target in
                # the SAME individual leaf trajectory.
                # This uses timestamps only to locate
                # a held-out label, never as an input.
                numeric_times = [
                    t.timestamp() for t in times
                ]

                position = bisect_left(
                    numeric_times, desired_seconds,
                    lo=cutoff_index + 1
                )

                candidates = []

                for idx in (position - 1, position):
                    if cutoff_index < idx < len(series):
                        candidates.append(idx)

                if not candidates:
                    continue

                target_index = min(
                    candidates,
                    key=lambda idx: abs(
                        numeric_times[idx]
                        - desired_seconds
                    )
                )

                target_time, target_pixels = (
                    series[target_index]
                )

                actual_offset = elapsed_hours(
                    cutoff_time, target_time
                )

                if abs(
                    actual_offset - horizon
                ) > TARGET_TOLERANCE_HOURS:
                    continue

                result = results[horizon]

                result["windows"] += 1
                result["leaf_ids"].add(leaf_id)
                result["cutoff_times"].add(cutoff_time)
                result["history_spans"].append(
                    history_span
                )
                result["target_offsets"].append(
                    actual_offset
                )

                if len(result["examples"]) < 2:
                    result["examples"].append({
                        "genotype": genotype,
                        "plant_id": plant_id,
                        "leaf_id": leaf_id,
                        "horizon_hours": horizon,
                        "history_start": (
                            first_time.isoformat()
                        ),
                        "input_cutoff": (
                            cutoff_time.isoformat()
                        ),
                        "target_timestamp": (
                            target_time.isoformat()
                        ),
                        "actual_offset_hours": round(
                            actual_offset, 3
                        ),
                        "history_span_hours": round(
                            history_span, 3
                        ),
                        "observed_target_pixels": (
                            target_pixels
                        ),
                    })

    rows = []
    examples = []

    for horizon, result in results.items():
        examples.extend(result["examples"])

        rows.append({
            "genotype": genotype,
            "plant_id": plant_id,
            "horizon_hours": horizon,
            "eligible": int(
                result["windows"] > 0
            ),
            "windows": result["windows"],
            "leaf_tracks": len(
                result["leaf_ids"]
            ),
            "distinct_cutoffs": len(
                result["cutoff_times"]
            ),
            "median_history_span_hours": (
                round(
                    median(result["history_spans"]), 3
                )
                if result["history_spans"] else ""
            ),
            "median_actual_offset_hours": (
                round(
                    median(result["target_offsets"]), 3
                )
                if result["target_offsets"] else ""
            ),
            "flagged_annotation_plant": int(
                plant_id == FLAGGED_PLANT
            ),
        })

    return rows, examples


def summarize(rows, group):
    return {
        "genotype": group,
        "horizon_hours": rows[0]["horizon_hours"],
        "total_plants": len(rows),
        "eligible_plants": sum(
            r["eligible"] for r in rows
        ),
        "windows": sum(
            r["windows"] for r in rows
        ),
        "leaf_tracks": sum(
            r["leaf_tracks"] for r in rows
        ),
        "distinct_plant_cutoffs": sum(
            r["distinct_cutoffs"] for r in rows
        ),
    }


def main():
    print("=" * 70)
    print("E0.5 — TIME-AWARE FORECASTING FEASIBILITY")
    print("=" * 70)

    sequences = load_data()

    plant_rows = []
    example_rows = []

    for pid, frames in sorted(sequences.items()):
        rows, examples = eligible_windows_for_plant(
            pid, frames
        )

        plant_rows.extend(rows)
        example_rows.extend(examples)

    genotype_rows = []
    overall_rows = []

    for horizon in HORIZONS_HOURS:
        subset = [
            r for r in plant_rows
            if r["horizon_hours"] == horizon
        ]

        genotypes = sorted({
            r["genotype"] for r in subset
        })

        for genotype in genotypes:
            group_rows = [
                r for r in subset
                if r["genotype"] == genotype
            ]

            genotype_rows.append(
                summarize(group_rows, genotype)
            )

        overall_rows.append(
            summarize(subset, "ALL")
        )

        without_flagged = [
            r for r in subset
            if not r["flagged_annotation_plant"]
        ]

        overall_rows.append(
            summarize(
                without_flagged,
                "EXCLUDING_FLAGGED_PLANT"
            )
        )

    plant_columns = [
        "genotype", "plant_id",
        "horizon_hours", "eligible",
        "windows", "leaf_tracks",
        "distinct_cutoffs",
        "median_history_span_hours",
        "median_actual_offset_hours",
        "flagged_annotation_plant",
    ]

    summary_columns = [
        "genotype", "horizon_hours",
        "total_plants", "eligible_plants",
        "windows", "leaf_tracks",
        "distinct_plant_cutoffs",
    ]

    save_csv(
        TABLES / "E0_5_plant_eligibility.csv",
        plant_rows,
        plant_columns
    )

    save_csv(
        TABLES / "E0_5_genotype_summary.csv",
        genotype_rows,
        summary_columns
    )

    save_csv(
        TABLES / "E0_5_overall_summary.csv",
        overall_rows,
        summary_columns
    )

    save_csv(
        TABLES / "E0_5_example_windows.csv",
        example_rows,
        [
            "genotype", "plant_id", "leaf_id",
            "horizon_hours", "history_start",
            "input_cutoff", "target_timestamp",
            "actual_offset_hours",
            "history_span_hours",
            "observed_target_pixels",
        ]
    )

    report = [
        "# E0.5 — Time-Aware Forecasting Feasibility",
        "",
        "## Preliminary configuration",
        "",
        f"- Historical observations: {HISTORY_LENGTH}",
        f"- Maximum history span: {MAX_HISTORY_HOURS} h",
        f"- Horizons: {HORIZONS_HOURS} h",
        f"- Target tolerance: ±{TARGET_TOLERANCE_HOURS} h",
        f"- Annotation warning: `{FLAGGED_PLANT}`",
        "",
        "## Overall results",
        "",
        "| Group | Horizon (h) | Eligible plants | Windows |",
        "|---|---:|---:|---:|",
    ]

    for r in overall_rows:
        report.append(
            f"| {r['genotype']} "
            f"| {r['horizon_hours']} "
            f"| {r['eligible_plants']}/{r['total_plants']} "
            f"| {r['windows']} |"
        )

    report.extend([
        "",
        "## Genotype summary",
        "",
        "| Genotype | Horizon (h) | Eligible plants | Windows |",
        "|---|---:|---:|---:|",
    ])

    for r in genotype_rows:
        report.append(
            f"| {r['genotype']} "
            f"| {r['horizon_hours']} "
            f"| {r['eligible_plants']}/{r['total_plants']} "
            f"| {r['windows']} |"
        )

    report.extend([
        "",
        "## Interpretation",
        "",
        "Eligible forecasting windows are not independent "
        "biological replicates.",
        "",
        "This audit permits irregular historical observation "
        "intervals while requiring an observed future target.",
        "",
        "The history duration can vary by genotype. "
        "Future modeling must use elapsed-time information "
        "or explicitly account for this variation.",
        "",
        "Unknown-color annotations in the flagged plant "
        "remain unresolved.",
        "",
        "The forecast target is visible projected area, "
        "not true physical leaf area.",
        "",
        "No model or split protocol has been finalized.",
    ])

    (
        REPORTS / "E0_5_time_aware_feasibility.md"
    ).write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("\nOVERALL")
    print(
        f"{'Group':28s} {'Hours':>5s} "
        f"{'Plants':>10s} {'Windows':>12s}"
    )

    for row in overall_rows:
        print(
            f"{row['genotype']:28s} "
            f"{row['horizon_hours']:5d} "
            f"{row['eligible_plants']:4d}/"
            f"{row['total_plants']:<5d} "
            f"{row['windows']:12d}"
        )

    print("\nGENOTYPE ELIGIBILITY")

    for horizon in HORIZONS_HOURS:
        group = [
            r for r in genotype_rows
            if r["horizon_hours"] == horizon
        ]

        have_windows = sum(
            r["eligible_plants"] > 0
            for r in group
        )

        have_three = sum(
            r["eligible_plants"] >= 3
            for r in group
        )

        print(
            f"{horizon}h: "
            f"{have_windows}/{len(group)} "
            f"genotypes eligible; "
            f"{have_three} with >=3 eligible plants"
        )

    print("\nGENOTYPES WITH NO ELIGIBLE PLANTS")

    for horizon in HORIZONS_HOURS:
        missing = [
            r["genotype"]
            for r in genotype_rows
            if (
                r["horizon_hours"] == horizon
                and r["eligible_plants"] == 0
            )
        ]

        print(f"{horizon}h:", missing)

    print("\nOutput directory:", TABLES)
    print("E0.5 complete.")


if __name__ == "__main__":
    main()
