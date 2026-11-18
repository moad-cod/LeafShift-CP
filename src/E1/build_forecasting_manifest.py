from pathlib import Path
from collections import defaultdict, Counter
from bisect import bisect_left
from datetime import datetime
import csv
import json
import hashlib
import statistics

# ============================================================
# E1.0 — LEAKAGE-SAFE FORECASTING WINDOW MANIFEST
#
# Study:
# Reliable Individual-Leaf Growth Forecasting
# Under Genotype Shift
#
# Inputs: E0.1 and E0.2 CSV tables
# Outputs: forecasting manifest, summaries, figure, report
#
# No model training. Raw images are not modified.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

E0 = ROOT / "experiments/E0/tables"
OUT = ROOT / "data/processed/forecasting"
EXP = ROOT / "experiments/E1"

TABLES = EXP / "tables"
FIGURES = EXP / "figures"
REPORTS = EXP / "reports"

for folder in (OUT, TABLES, FIGURES, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

PAIRING = E0 / "E0_1_image_pairing.csv"
TRAJECTORIES = E0 / "E0_2_leaf_area_trajectories.csv"

MANIFEST = OUT / "forecast_windows.csv"
METADATA = OUT / "manifest_metadata.json"
SUMMARY = TABLES / "E1_0_window_integrity_summary.csv"
REPORT = REPORTS / "E1_0_manifest_validation.md"

HISTORY_LENGTH = 6
MAX_HISTORY_HOURS = 72.0
HORIZONS = (24, 48)
TARGET_TOLERANCE_HOURS = 3.0
MIN_AREA_PIXELS = 20

FLAGGED_PLANT = "Is-1/rep_24"

REFERENCE_COUNTS = {
    24: 121431,
    48: 107214,
}

FIELDS = [
    "window_id",
    "genotype",
    "plant_id",
    "leaf_id",
    "horizon_hours",
    "history_start",
    "cutoff_timestamp",
    "target_timestamp",
    "actual_target_offset_hours",
    "history_span_hours",
    "history_timestamps",
    "history_areas_pixels",
    "history_relative_hours",
    "target_area_pixels",
    "last_observed_area_pixels",
    "max_history_observation_gap_hours",
    "history_skipped_image_frames",
    "flagged_palette_plant",
]


def read_csv(path):
    if not path.exists():
        raise FileNotFoundError(path)

    with path.open(newline="", encoding="utf-8") as f:
        yield from csv.DictReader(f)


def write_csv(path, rows, columns):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def elapsed_hours(first, second):
    return (second - first).total_seconds() / 3600.0


def load_source():
    frames = defaultdict(dict)

    for row in read_csv(PAIRING):
        if row["paired"] != "1" or not row["timestamp"]:
            continue

        plant = row["plant_id"]
        name = row["filename"]

        frames[plant][name] = {
            "time": datetime.fromisoformat(
                row["timestamp"]
            ),
            "genotype": row["genotype"],
            "filename": name,
        }

    # Track the chronological image-frame positions.
    indices = {}

    for plant, images in frames.items():
        ordered = sorted(
            images.values(),
            key=lambda x: (
                x["time"], x["filename"]
            )
        )

        if len({x["time"] for x in ordered}) != len(ordered):
            raise ValueError(
                f"Duplicate timestamps in {plant}"
            )

        for index, frame in enumerate(ordered):
            indices[(plant, frame["filename"])] = index

    tracks = defaultdict(list)
    linked = 0

    for row in read_csv(TRAJECTORIES):
        plant = row["plant_id"]
        name = row["filename"]
        leaf_id = int(row["leaf_id"])
        area = int(row["area_pixels"])

        if area < MIN_AREA_PIXELS:
            continue

        frame = frames.get(plant, {}).get(name)

        if frame is None:
            raise ValueError(
                f"Unmatched leaf observation: {plant}/{name}"
            )

        if row["genotype"] != frame["genotype"]:
            raise ValueError(
                f"Genotype mismatch for {plant}/{name}"
            )

        tracks[
            (frame["genotype"], plant, leaf_id)
        ].append({
            "time": frame["time"],
            "area": area,
            "frame_index": indices[(plant, name)],
            "filename": name,
        })

        linked += 1

    for key, series in tracks.items():
        series.sort(key=lambda x: x["time"])

        if len({x["time"] for x in series}) != len(series):
            raise ValueError(
                f"Duplicate leaf observation: {key}"
            )

    return tracks, frames, linked


def make_window_id(
    genotype, plant, leaf_id, cutoff, horizon
):
    text = "|".join([
        genotype,
        plant,
        str(leaf_id),
        cutoff.isoformat(),
        str(horizon),
    ])

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:20]


def generate_windows(tracks):
    """
    Reproduce E0.5's selection policy exactly:
    - six successive observed leaf areas
    - history span <=72 hours
    - nearest observed future target
    - target within +/-3 hours

    Note: a missing leaf observation can be skipped.
    Its extent is recorded as a diagnostic.
    """

    for (genotype, plant, leaf_id), series in sorted(
        tracks.items()
    ):
        if len(series) <= HISTORY_LENGTH:
            continue

        times = [x["time"] for x in series]
        numeric_times = [
            x.timestamp() for x in times
        ]

        for cutoff_idx in range(
            HISTORY_LENGTH - 1,
            len(series) - 1
        ):
            start_idx = (
                cutoff_idx - HISTORY_LENGTH + 1
            )
            history = series[start_idx:cutoff_idx + 1]

            start = history[0]["time"]
            cutoff = history[-1]["time"]

            span = elapsed_hours(start, cutoff)

            if span > MAX_HISTORY_HOURS:
                continue

            history_times = [
                obs["time"] for obs in history
            ]
            history_areas = [
                obs["area"] for obs in history
            ]
            relative_times = [
                round(elapsed_hours(cutoff, t), 5)
                for t in history_times
            ]

            skipped_frames = sum(
                max(
                    0,
                    b["frame_index"]
                    - a["frame_index"] - 1
                )
                for a, b in zip(
                    history, history[1:]
                )
            )

            largest_gap = max(
                elapsed_hours(a["time"], b["time"])
                for a, b in zip(
                    history, history[1:]
                )
            )

            for horizon in HORIZONS:
                desired = (
                    cutoff.timestamp()
                    + horizon * 3600
                )

                position = bisect_left(
                    numeric_times,
                    desired,
                    lo=cutoff_idx + 1,
                )

                candidates = [
                    j for j in (position - 1, position)
                    if cutoff_idx < j < len(series)
                ]

                if not candidates:
                    continue

                target_idx = min(
                    candidates,
                    key=lambda j: abs(
                        numeric_times[j] - desired
                    ),
                )

                target = series[target_idx]
                offset = elapsed_hours(
                    cutoff, target["time"]
                )

                if abs(offset - horizon) > (
                    TARGET_TOLERANCE_HOURS
                ):
                    continue

                yield {
                    "window_id": make_window_id(
                        genotype,
                        plant,
                        leaf_id,
                        cutoff,
                        horizon,
                    ),
                    "genotype": genotype,
                    "plant_id": plant,
                    "leaf_id": leaf_id,
                    "horizon_hours": horizon,
                    "history_start": start.isoformat(),
                    "cutoff_timestamp": cutoff.isoformat(),
                    "target_timestamp": (
                        target["time"].isoformat()
                    ),
                    "actual_target_offset_hours": round(
                        offset, 5
                    ),
                    "history_span_hours": round(
                        span, 5
                    ),
                    "history_timestamps": json.dumps([
                        t.isoformat()
                        for t in history_times
                    ]),
                    "history_areas_pixels": json.dumps(
                        history_areas
                    ),
                    "history_relative_hours": json.dumps(
                        relative_times
                    ),
                    "target_area_pixels": target["area"],
                    "last_observed_area_pixels": (
                        history_areas[-1]
                    ),
                    "max_history_observation_gap_hours": round(
                        largest_gap, 5
                    ),
                    "history_skipped_image_frames": (
                        skipped_frames
                    ),
                    "flagged_palette_plant": int(
                        plant == FLAGGED_PLANT
                    ),
                }


def validate_row(row):
    times = [
        datetime.fromisoformat(x)
        for x in json.loads(row["history_timestamps"])
    ]
    areas = json.loads(row["history_areas_pixels"])
    relative = json.loads(row["history_relative_hours"])

    cutoff = datetime.fromisoformat(
        row["cutoff_timestamp"]
    )
    target = datetime.fromisoformat(
        row["target_timestamp"]
    )

    assert len(times) == HISTORY_LENGTH
    assert len(areas) == HISTORY_LENGTH
    assert len(relative) == HISTORY_LENGTH

    assert all(
        a < b
        for a, b in zip(times, times[1:])
    )

    assert times[-1] == cutoff
    assert target > cutoff

    assert all(a >= MIN_AREA_PIXELS for a in areas)
    assert row["target_area_pixels"] >= MIN_AREA_PIXELS
    assert areas[-1] == row["last_observed_area_pixels"]

    for t, r in zip(times, relative):
        assert abs(
            elapsed_hours(cutoff, t) - r
        ) < 0.0001

    assert elapsed_hours(
        times[0], cutoff
    ) <= MAX_HISTORY_HOURS + 1e-5

    assert abs(
        elapsed_hours(cutoff, target)
        - row["horizon_hours"]
    ) <= TARGET_TOLERANCE_HOURS + 1e-5

    assert row["plant_id"].startswith(
        row["genotype"] + "/"
    )


def plot_example(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable: example figure skipped.")
        return

    # Choose one example with a substantial positive
    # increase, to show a meaningful forecasting target.
    best = max(
        rows[:min(len(rows), 3000)],
        key=lambda x: (
            x["target_area_pixels"]
            - x["last_observed_area_pixels"]
        ),
    )

    times = json.loads(best["history_relative_hours"])
    areas = json.loads(best["history_areas_pixels"])

    x = times + [best["actual_target_offset_hours"]]
    y = areas + [best["target_area_pixels"]]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        x[:HISTORY_LENGTH],
        y[:HISTORY_LENGTH],
        marker="o",
        label="Historical observations",
    )
    ax.scatter(
        x[-1], y[-1],
        s=85, marker="*",
        label="Future target",
    )

    ax.axvline(
        0, linestyle="--",
        label="Prediction cutoff",
    )

    ax.set_xlabel("Hours relative to cutoff")
    ax.set_ylabel("Visible projected leaf area (pixels)")
    ax.set_title(
        f"E1.0 Example | {best['plant_id']} "
        f"| Leaf {best['leaf_id']} "
        f"| {best['horizon_hours']}h"
    )
    ax.legend()
    fig.tight_layout()

    fig.savefig(
        FIGURES / "E1_0_forecasting_example.png",
        dpi=200,
    )
    plt.close(fig)


def main():
    print("=" * 68)
    print("E1.0 — FORECASTING WINDOW MANIFEST")
    print("=" * 68)

    tracks, frames, linked = load_source()

    print("Plant sequences:", len(frames))
    print("Color-indexed leaf tracks:", len(tracks))
    print("Linked observations:", linked)

    counts = Counter()
    genotypes_by_horizon = defaultdict(set)
    plants_by_horizon = defaultdict(set)
    skipped_history_count = Counter()
    flagged_count = Counter()

    ids = set()
    sample_for_plot = []

    with MANIFEST.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=FIELDS
        )
        writer.writeheader()

        for row in generate_windows(tracks):
            validate_row(row)

            wid = row["window_id"]

            if wid in ids:
                raise ValueError(
                    f"Duplicate forecast window ID: {wid}"
                )

            ids.add(wid)
            writer.writerow(row)

            h = row["horizon_hours"]
            counts[h] += 1

            genotypes_by_horizon[h].add(
                row["genotype"]
            )
            plants_by_horizon[h].add(
                row["plant_id"]
            )

            if row["history_skipped_image_frames"] > 0:
                skipped_history_count[h] += 1

            if row["flagged_palette_plant"]:
                flagged_count[h] += 1

            if len(sample_for_plot) < 3000:
                sample_for_plot.append(row)

    results = []

    for h in HORIZONS:
        results.append({
            "horizon_hours": h,
            "windows": counts[h],
            "eligible_genotypes": len(
                genotypes_by_horizon[h]
            ),
            "eligible_plants": len(
                plants_by_horizon[h]
            ),
            "windows_with_skipped_history_frames": (
                skipped_history_count[h]
            ),
            "windows_in_flagged_palette_plant": (
                flagged_count[h]
            ),
            "matches_E0_5_reference": int(
                counts[h] == REFERENCE_COUNTS[h]
            ),
        })

    write_csv(
        SUMMARY,
        results,
        list(results[0].keys())
    )

    metadata = {
        "experiment": "E1.0",
        "manifest": str(
            MANIFEST.relative_to(ROOT)
        ),
        "history_length": HISTORY_LENGTH,
        "max_history_hours": MAX_HISTORY_HOURS,
        "horizons_hours": list(HORIZONS),
        "target_tolerance_hours": (
            TARGET_TOLERANCE_HOURS
        ),
        "min_area_pixels": MIN_AREA_PIXELS,
        "unit": "visible_projected_area_pixels",
        "genotype_split_policy": (
            "Leave-one-genotype-out; folds not yet trained"
        ),
        "flagged_plant": FLAGGED_PLANT,
        "summary": results,
        "quality_warning": (
            "Color identities are intended to persist, "
            "but some annotation consistency checks "
            "remain unresolved. Future-target quality "
            "must not enter input features."
        ),
    }

    METADATA.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    report = [
        "# E1.0 — Forecasting Manifest Integrity",
        "",
        "## Configuration",
        "",
        f"- Input observations: {HISTORY_LENGTH}",
        f"- Maximum history span: {MAX_HISTORY_HOURS}h",
        f"- Horizons: {HORIZONS}",
        f"- Target tolerance: +/-{TARGET_TOLERANCE_HOURS}h",
        "",
        "## Results",
        "",
        "| Horizon | Windows | Genotypes | Plants | "
        "Skipped-history windows | E0.5 matched |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for row in results:
        report.append(
            f"| {row['horizon_hours']}h "
            f"| {row['windows']:,} "
            f"| {row['eligible_genotypes']} "
            f"| {row['eligible_plants']} "
            f"| {row['windows_with_skipped_history_frames']:,} "
            f"| {'YES' if row['matches_E0_5_reference'] else 'NO'} |"
        )

    report.extend([
        "",
        "## Integrity checks",
        "",
        "- Six historical areas per window.",
        "- Six strictly ordered timestamps.",
        "- Cutoff equals the last historical timestamp.",
        "- Target strictly follows cutoff.",
        "- Target time within horizon tolerance.",
        "- Unique deterministic window identifiers.",
        "- Genotype and plant identifiers consistent.",
        "- All areas above the extraction threshold.",
        "",
        "## Limitations",
        "",
        "- Candidate targets require visible future leaves.",
        "- Some history observations may skip image frames.",
        "- No imputation or identity correction performed.",
        "- Overlapping windows are not independent samples.",
        "- The manifest contains target labels and must "
        "not be supplied directly as unrestricted model input.",
        "- Genotype-disjoint training folds are not yet built.",
        "- Annotation and occlusion QC remain provisional.",
        "",
        "## Next",
        "",
        "E1.1 — Validate leave-one-genotype-out fold "
        "assignments and prepare causal baseline features.",
    ])

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    if sample_for_plot:
        plot_example(sample_for_plot)

    print("\n" + "=" * 68)
    print("E1.0 — RESULTS")
    print("=" * 68)

    for row in results:
        print(
            f"{row['horizon_hours']}h: "
            f"{row['windows']:,} windows | "
            f"{row['eligible_genotypes']}/21 genotypes | "
            f"{row['eligible_plants']}/318 plants"
        )

        print(
            "  E0.5 count reproduced:",
            bool(row["matches_E0_5_reference"])
        )

        print(
            "  Windows with skipped history frames:",
            row["windows_with_skipped_history_frames"]
        )

        print(
            "  Windows from flagged plant:",
            row["windows_in_flagged_palette_plant"]
        )

    print("\nTotal windows:", sum(counts.values()))
    print("Saved manifest:", MANIFEST)
    print("Saved summary:", SUMMARY)
    print("Saved report:", REPORT)

    if not all(
        r["matches_E0_5_reference"]
        for r in results
    ):
        raise RuntimeError(
            "E1.0 counts differ from E0.5. "
            "Investigate before starting E1.1."
        )

    print("\nE1.0 PASSED — ready for fold validation.")


if __name__ == "__main__":
    main()