
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import csv
from statistics import median

# ============================================================
# E0.3 — GENOTYPE-HELD-OUT FORECASTING FEASIBILITY
#
# Inputs:
#   E0_1_image_pairing.csv
#   E0_2_leaf_area_trajectories.csv
#
# Forecasting target:
#   Visible projected area of a numbered individual leaf.
#
# This is an audit, NOT model training.
# No train/test split is created yet.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

TABLES = ROOT / "experiments/E0/tables"
REPORTS = ROOT / "experiments/E0/reports"
FIGURES = ROOT / "experiments/E0/figures"

for folder in (TABLES, REPORTS, FIGURES):
    folder.mkdir(parents=True, exist_ok=True)

PAIRING_FILE = TABLES / "E0_1_image_pairing.csv"
TRAJECTORY_FILE = TABLES / "E0_2_leaf_area_trajectories.csv"

HISTORY = 6
HORIZONS = (1, 4, 8)

# Use actual timestamps. Individual adjacent acquisitions
# must fall within this provisional range.
MIN_GAP_HOURS = 3.0
MAX_GAP_HOURS = 9.0

# A horizon should also approximately represent
# H * 6 hours, not merely H arbitrary images.
EXPECTED_STEP_HOURS = 6.0
TOLERANCE_PER_STEP_HOURS = 1.5

# E0.2 only recorded leaves with >=20 pixels.
# This is a visibility threshold, not biological QC.
MIN_AREA_PIXELS = 20

# All seven unmapped colors occur in this plant.
# Results will be shown both including and excluding it.
FLAGGED_PLANT = "Is-1/rep_24"

# Small examples table; full counts remain in summaries.
MAX_EXAMPLES_PER_GENOTYPE_HORIZON = 3


def read_csv(path):
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing required E0 table: {path}"
        )

    with path.open(
        newline="", encoding="utf-8"
    ) as file:
        return list(csv.DictReader(file))


def write_csv(path, rows, columns):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=columns
        )
        writer.writeheader()
        writer.writerows(rows)


def hours_between(first, second):
    return (
        second - first
    ).total_seconds() / 3600.0


def load_frames():
    """
    Build one chronological frame sequence per plant.
    Only exact raw/label pairs are included.
    """
    plants = defaultdict(list)

    for row in read_csv(PAIRING_FILE):
        if (
            row["raw_exists"] != "1"
            or row["label_exists"] != "1"
            or not row["timestamp"]
        ):
            continue

        plants[row["plant_id"]].append({
            "genotype": row["genotype"],
            "filename": row["filename"],
            "timestamp": datetime.fromisoformat(
                row["timestamp"]
            ),
            "leaf_areas": {},
        })

    for plant_id in plants:
        plants[plant_id].sort(
            key=lambda frame: (
                frame["timestamp"],
                frame["filename"]
            )
        )

    return plants


def attach_leaf_areas(plants):
    """
    Attach decoded E0.2 area observations to frames.

    Important: absence from the trajectory CSV means
    'not observed above threshold', NOT biological
    disappearance or zero true leaf area.
    """
    lookup = {}

    for plant_id, frames in plants.items():
        for frame in frames:
            key = (plant_id, frame["filename"])

            if key in lookup:
                raise ValueError(
                    f"Duplicate image key: {key}"
                )

            lookup[key] = frame

    linked = 0
    missing = 0

    for row in read_csv(TRAJECTORY_FILE):
        key = (
            row["plant_id"],
            row["filename"]
        )

        frame = lookup.get(key)

        if frame is None:
            missing += 1
            continue

        leaf_id = int(row["leaf_id"])
        pixels = int(row["area_pixels"])

        if pixels < MIN_AREA_PIXELS:
            continue

        frame["leaf_areas"][leaf_id] = pixels
        linked += 1

    if missing:
        raise RuntimeError(
            f"{missing} E0.2 observations do not "
            "match E0.1 image pairs. "
            "Check that both audits used all plants."
        )

    print("Linked leaf observations:", linked)
    return plants


def temporal_window_valid(frames, start, end, horizon):
    """
    Test acquisition continuity using actual timestamps.

    start = beginning of six historical observations.
    end = future target frame.
    """
    for j in range(start, end):
        gap = hours_between(
            frames[j]["timestamp"],
            frames[j + 1]["timestamp"]
        )

        if not MIN_GAP_HOURS <= gap <= MAX_GAP_HOURS:
            return False

    anchor = start + HISTORY - 1

    forecast_hours = hours_between(
        frames[anchor]["timestamp"],
        frames[end]["timestamp"]
    )

    expected_hours = horizon * EXPECTED_STEP_HOURS

    tolerance = max(
        1.5,
        horizon * TOLERANCE_PER_STEP_HOURS
    )

    return (
        abs(forecast_hours - expected_hours)
        <= tolerance
    )


def evaluate_plant(plant_id, frames, horizon):
    """
    Two eligibility definitions:

    endpoint:
       Leaf observed at all six historical timestamps
       and at the target timestamp.

    strict:
       Endpoint condition PLUS leaf observed in
       every intervening frame.

    Both require acceptable acquisition timing.
    """
    strict_windows = 0
    endpoint_windows = 0

    strict_leaf_ids = set()
    endpoint_leaf_ids = set()

    strict_anchor_indices = set()
    endpoint_anchor_indices = set()

    temporal_anchors = 0
    examples = []

    n = len(frames)

    for anchor in range(HISTORY - 1, n - horizon):
        start = anchor - HISTORY + 1
        end = anchor + horizon

        if not temporal_window_valid(
            frames, start, end, horizon
        ):
            continue

        temporal_anchors += 1

        history_frames = frames[start:anchor + 1]
        future_frame = frames[end]

        # Candidate leaves are those visible at every
        # historical input timestamp.
        history_ids = set(
            history_frames[0]["leaf_areas"]
        )

        for frame in history_frames[1:]:
            history_ids.intersection_update(
                frame["leaf_areas"]
            )

        # The target must be observed.
        endpoint_ids = history_ids.intersection(
            future_frame["leaf_areas"]
        )

        for leaf_id in endpoint_ids:
            endpoint_windows += 1
            endpoint_leaf_ids.add(leaf_id)
            endpoint_anchor_indices.add(anchor)

            # Strictly require visibility between
            # input cutoff and target.
            intermediate = frames[anchor + 1:end]

            complete_future = all(
                leaf_id in frame["leaf_areas"]
                for frame in intermediate
            )

            if not complete_future:
                continue

            strict_windows += 1
            strict_leaf_ids.add(leaf_id)
            strict_anchor_indices.add(anchor)

            if len(examples) < 3:
                examples.append({
                    "genotype": frames[0]["genotype"],
                    "plant_id": plant_id,
                    "leaf_id": leaf_id,
                    "horizon_frames": horizon,
                    "history_start": (
                        frames[start]["timestamp"].isoformat()
                    ),
                    "input_cutoff": (
                        frames[anchor]["timestamp"].isoformat()
                    ),
                    "target_timestamp": (
                        future_frame["timestamp"].isoformat()
                    ),
                    "observed_target_pixels": (
                        future_frame["leaf_areas"][leaf_id]
                    ),
                })

    return {
        "genotype": frames[0]["genotype"],
        "plant_id": plant_id,
        "horizon_frames": horizon,
        "frames": n,
        "temporal_anchors": temporal_anchors,
        "endpoint_windows": endpoint_windows,
        "strict_windows": strict_windows,
        "endpoint_leaf_tracks": len(endpoint_leaf_ids),
        "strict_leaf_tracks": len(strict_leaf_ids),
        "endpoint_plant_eligible": int(
            endpoint_windows > 0
        ),
        "strict_plant_eligible": int(
            strict_windows > 0
        ),
        "endpoint_anchor_count": len(
            endpoint_anchor_indices
        ),
        "strict_anchor_count": len(
            strict_anchor_indices
        ),
        "flagged_annotation_plant": int(
            plant_id == FLAGGED_PLANT
        ),
        "examples": examples,
    }


def aggregate(rows, group_name):
    """Summarize plant-level eligibility without
    pretending correlated windows are independent.
    """
    return {
        "group": group_name,
        "horizon_frames": rows[0]["horizon_frames"],
        "plants_total": len(rows),
        "plants_strict_eligible": sum(
            x["strict_plant_eligible"] for x in rows
        ),
        "plants_endpoint_eligible": sum(
            x["endpoint_plant_eligible"] for x in rows
        ),
        "strict_windows": sum(
            x["strict_windows"] for x in rows
        ),
        "endpoint_windows": sum(
            x["endpoint_windows"] for x in rows
        ),
        "strict_leaf_tracks": sum(
            x["strict_leaf_tracks"] for x in rows
        ),
        "endpoint_leaf_tracks": sum(
            x["endpoint_leaf_tracks"] for x in rows
        ),
        "temporal_anchors": sum(
            x["temporal_anchors"] for x in rows
        ),
    }


def main():
    print("=" * 68)
    print("E0.3 — FORECASTING FEASIBILITY UNDER GENOTYPE SHIFT")
    print("=" * 68)

    plants = load_frames()
    attach_leaf_areas(plants)

    print("Candidate plants:", len(plants))
    print(
        "Genotypes:",
        len({
            frames[0]["genotype"]
            for frames in plants.values()
            if frames
        })
    )

    plant_results = []
    example_rows = []

    for plant_id, frames in sorted(plants.items()):
        if not frames:
            continue

        for horizon in HORIZONS:
            result = evaluate_plant(
                plant_id, frames, horizon
            )

            example_rows.extend(result.pop("examples"))
            plant_results.append(result)

    genotype_results = []
    overall_results = []

    for horizon in HORIZONS:
        relevant = [
            r for r in plant_results
            if r["horizon_frames"] == horizon
        ]

        genotype_names = sorted({
            r["genotype"] for r in relevant
        })

        for genotype in genotype_names:
            subset = [
                r for r in relevant
                if r["genotype"] == genotype
            ]

            genotype_results.append(
                aggregate(subset, genotype)
            )

        overall_results.append(
            aggregate(relevant, "ALL")
        )

        without_flagged = [
            r for r in relevant
            if not r["flagged_annotation_plant"]
        ]

        overall_results.append(
            aggregate(
                without_flagged,
                "EXCLUDING_FLAGGED_PLANT"
            )
        )

    plant_columns = [
        "genotype",
        "plant_id",
        "horizon_frames",
        "frames",
        "temporal_anchors",
        "endpoint_windows",
        "strict_windows",
        "endpoint_leaf_tracks",
        "strict_leaf_tracks",
        "endpoint_plant_eligible",
        "strict_plant_eligible",
        "endpoint_anchor_count",
        "strict_anchor_count",
        "flagged_annotation_plant",
    ]

    summary_columns = [
        "group",
        "horizon_frames",
        "plants_total",
        "plants_strict_eligible",
        "plants_endpoint_eligible",
        "strict_windows",
        "endpoint_windows",
        "strict_leaf_tracks",
        "endpoint_leaf_tracks",
        "temporal_anchors",
    ]

    write_csv(
        TABLES / "E0_3_plant_forecast_eligibility.csv",
        plant_results,
        plant_columns
    )

    write_csv(
        TABLES / "E0_3_genotype_forecast_summary.csv",
        genotype_results,
        summary_columns
    )

    write_csv(
        TABLES / "E0_3_overall_forecast_summary.csv",
        overall_results,
        summary_columns
    )

    write_csv(
        TABLES / "E0_3_example_windows.csv",
        example_rows,
        [
            "genotype",
            "plant_id",
            "leaf_id",
            "horizon_frames",
            "history_start",
            "input_cutoff",
            "target_timestamp",
            "observed_target_pixels",
        ]
    )

    # Optional figure: 48-hour strict eligibility
    try:
        import matplotlib.pyplot as plt

        figure_rows = [
            row for row in genotype_results
            if row["horizon_frames"] == 8
        ]

        names = [r["group"] for r in figure_rows]
        values = [
            r["plants_strict_eligible"]
            for r in figure_rows
        ]

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.barh(names, values)
        ax.invert_yaxis()

        ax.set_xlabel(
            "Plants with at least one strict 48-hour window"
        )
        ax.set_ylabel("Genotype")
        ax.set_title(
            "E0.3 — Genotype-specific 48-hour "
            "forecasting eligibility"
        )

        fig.tight_layout()
        fig.savefig(
            FIGURES / "E0_3_48h_genotype_eligibility.png",
            dpi=200
        )
        plt.close(fig)

    except ImportError:
        print("Matplotlib unavailable; figure skipped.")

    # Build report
    report = [
        "# E0.3 — Genotype-Held-Out Forecasting Feasibility",
        "",
        "## Configuration",
        "",
        f"- Historical input frames: {HISTORY}",
        f"- Forecast horizons: {HORIZONS}",
        f"- Minimum visible leaf area: {MIN_AREA_PIXELS} pixels",
        f"- Allowed adjacent gap: {MIN_GAP_HOURS}–{MAX_GAP_HOURS} h",
        f"- Expected interval: {EXPECTED_STEP_HOURS} h",
        f"- Flagged plant: `{FLAGGED_PLANT}`",
        "",
        "## Overall results",
        "",
        "| Group | Horizon | Eligible plants | "
        "Strict windows | Endpoint windows |",
        "|---|---:|---:|---:|---:|",
    ]

    for row in overall_results:
        report.append(
            f"| {row['group']} "
            f"| {row['horizon_frames']} "
            f"| {row['plants_strict_eligible']}"
            f"/{row['plants_total']} "
            f"| {row['strict_windows']} "
            f"| {row['endpoint_windows']} |"
        )

    report.extend([
        "",
        "## Genotype-level strict eligibility",
        "",
        "| Genotype | Horizon | Eligible plants | "
        "Strict windows |",
        "|---|---:|---:|---:|",
    ])

    for row in genotype_results:
        report.append(
            f"| {row['group']} "
            f"| {row['horizon_frames']} "
            f"| {row['plants_strict_eligible']}"
            f"/{row['plants_total']} "
            f"| {row['strict_windows']} |"
        )

    report.extend([
        "",
        "## Scientific interpretation",
        "",
        "The analysis checks whether prediction targets are "
        "observable. It does not demonstrate prediction accuracy.",
        "",
        "Forecasting windows sharing the same plant or leaf "
        "are correlated and are not independent replicates.",
        "",
        "A strict window requires observations at each "
        "historical and intermediate timestamp.",
        "",
        "An endpoint window allows intermediate leaf "
        "observations to be missing, but requires the "
        "historical inputs and final target.",
        "",
        "Missing color-coded leaf pixels are not interpreted "
        "as zero biological area.",
        "",
        "The target is **visible projected area in pixels**, "
        "not corrected physical leaf area or biomass.",
        "",
        "The first 31 documented color identities are used. "
        "The unmapped colors in Is-1/rep_24 remain unresolved.",
        "",
        "## Outstanding questions",
        "",
        "1. Are replicate folders independent plants?",
        "2. Do color identities remain biologically consistent?",
        "3. What fraction of complete windows contain "
        "large area decreases from occlusion?",
        "4. Can held-out genotypes provide sufficient "
        "independent test plants?",
        "5. Are the proposed prediction intervals calibrated "
        "under unseen-genotype shift?",
        "",
        "No research protocol is finalized by this audit.",
    ])

    report_path = (
        REPORTS / "E0_3_forecasting_feasibility.md"
    )

    report_path.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8"
    )

    print("\n" + "=" * 68)
    print("E0.3 — OVERALL RESULTS")
    print("=" * 68)

    print(
        f"{'Group':28s} {'H':>3s} "
        f"{'Eligible':>10s} "
        f"{'Strict':>12s} "
        f"{'Endpoint':>12s}"
    )

    for row in overall_results:
        eligible = (
            f"{row['plants_strict_eligible']}"
            f"/{row['plants_total']}"
        )

        print(
            f"{row['group']:28s} "
            f"{row['horizon_frames']:>3d} "
            f"{eligible:>10s} "
            f"{row['strict_windows']:>12d} "
            f"{row['endpoint_windows']:>12d}"
        )

    print("\nStrict-eligible genotypes by horizon:")

    for horizon in HORIZONS:
        rows = [
            r for r in genotype_results
            if r["horizon_frames"] == horizon
        ]

        eligible_genotypes = sum(
            r["plants_strict_eligible"] > 0
            for r in rows
        )

        genotypes_with_3_plants = sum(
            r["plants_strict_eligible"] >= 3
            for r in rows
        )

        print(
            f"  H={horizon}: "
            f"{eligible_genotypes}/{len(rows)} "
            f"have windows; "
            f"{genotypes_with_3_plants} "
            "have >=3 eligible plants"
        )

    print("\nOutputs:")
    print(TABLES / "E0_3_genotype_forecast_summary.csv")
    print(TABLES / "E0_3_plant_forecast_eligibility.csv")
    print(REPORTS / "E0_3_forecasting_feasibility.md")
    print("\nE0.3 complete.")


if __name__ == "__main__":
    main()