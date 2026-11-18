
from pathlib import Path
from datetime import date
import csv

# ============================================================
# E0 FINALIZATION — DATASET VALIDATION + RESEARCH PROTOCOL
#
# Creates two versioned Markdown research documents.
# Reads previously generated E0 tables.
# Does not modify dataset images, masks or E0 CSVs.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "experiments/E0/tables"
DEST = ROOT / "Research/protocol"

DEST.mkdir(parents=True, exist_ok=True)

VALIDATION = DEST / "dataset_validation_report_v1.md"
PROTOCOL = DEST / "research_protocol_v1.md"


def load_csv(name):
    path = TABLES / name

    if not path.is_file():
        raise FileNotFoundError(
            f"Missing E0 audit table: {path}"
        )

    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def num(value):
    return int(value)


def percent(part, total):
    return 100 * part / total if total else 0.0


def validation_data():
    e05_overall = load_csv(
        "E0_5_overall_summary.csv"
    )
    e05_genotypes = load_csv(
        "E0_5_genotype_summary.csv"
    )
    e06_tracks = load_csv(
        "E0_6_leaf_quality.csv"
    )
    e07_events = load_csv(
        "E0_7_suspicious_leaf_events.csv"
    )
    e10_transitions = load_csv(
        "E0_10_transition_summary.csv"
    )
    e10_matches = load_csv(
        "E0_10_bidirectional_matches.csv"
    )
    e11_sampled = load_csv(
        "E0_11_sampled_transitions.csv"
    )
    e11_matches = load_csv(
        "E0_11_candidate_matches.csv"
    )

    horizon_stats = {}

    for h in (24, 48):
        row = next(
            r for r in e05_overall
            if r["genotype"] == "ALL"
            and num(r["horizon_hours"]) == h
        )

        rows = [
            r for r in e05_genotypes
            if num(r["horizon_hours"]) == h
        ]

        horizon_stats[h] = {
            "plants": num(row["total_plants"]),
            "eligible": num(row["eligible_plants"]),
            "windows": num(row["windows"]),
            "genotypes": len(rows),
            "genotypes_eligible": sum(
                num(r["eligible_plants"]) > 0
                for r in rows
            ),
            "genotypes_3plus": sum(
                num(r["eligible_plants"]) >= 3
                for r in rows
            ),
        }

    drops = sum(
        r["event"] == "SHARP_DECREASE"
        for r in e07_events
    )
    increases = sum(
        r["event"] == "SHARP_INCREASE"
        for r in e07_events
    )

    return {
        "horizons": horizon_stats,
        "tracks": len(e06_tracks),
        "missing_tracks": sum(
            num(r["missing_internal_frames"]) > 0
            for r in e06_tracks
        ),
        "drop_tracks": sum(
            num(r["sharp_drops"]) > 0
            for r in e06_tracks
        ),
        "increase_tracks": sum(
            num(r["sharp_increases"]) > 0
            for r in e06_tracks
        ),
        "drop_events": drops,
        "increase_events": increases,
        "e10_transitions": len(e10_transitions),
        "e10_flagged_transitions": sum(
            num(r["has_strong_candidate"])
            for r in e10_transitions
        ),
        "e10_matches": len(e10_matches),
        "e11_sample": len(e11_sampled),
        "e11_flagged": sum(
            num(r["has_strong_match"])
            for r in e11_sampled
        ),
        "e11_matches": len(e11_matches),
    }


def make_validation_report(d):
    h24 = d["horizons"][24]
    h48 = d["horizons"][48]

    return f"""# PhenoLeaf-TS — Dataset Validation Report v1

**Study:** Reliable Individual-Leaf Growth Forecasting
Under Genotype Shift

**Status:** Preliminary feasibility validated;
annotation identity QC remains incomplete.

**Generated:** {date.today().isoformat()}

## 1. Dataset provenance

Source: PhenoLeaf-TS, Sarić et al., ECCV 2026.

Official project:
https://pisyntor.github.io/PhenoLeaf-TS/

Official repository:
https://github.com/pisyntor/PhenoLeaf-TS

The source documentation specifies 21 Arabidopsis
genotypes, 318 plant replicates, 17,082 images,
and persistent color-coded individual-leaf identities.

Dataset use is subject to the original
non-commercial academic-use terms.

## 2. Structural audit — E0.1

- Genotypes: 21
- Candidate plant sequences: 318
- Raw RGB images: 17,082
- Leaf-label images: 17,082
- Successfully matched pairs: 17,082
- Missing raw or annotation pairs: 0
- Invalid or duplicate raw timestamps: 0

Conclusion: Structural pairing and timestamp
extraction passed the tested checks.

## 3. Leaf extraction — E0.2

- Documented palette colors: 31
- Candidate tracks with four or more detections: 5,208
- Total area observations linked to E0.5: 165,796

Seven non-palette nonblack colors were found,
all in Is-1/rep_24.

Their biological semantics remain unresolved.
This plant must be flagged in sensitivity analysis.

## 4. Fixed-frame forecasting — E0.3/E0.4

With six historical frames and nominal six-hour
sampling, only 8/21 genotypes had eligible windows.

Diagnosis: Acquisition schedules vary across genotypes.

Decision: Do not use a uniform frame-index
forecasting horizon across the entire dataset.

## 5. Time-aware forecasting — E0.5

Configuration:
- Historical observations: six
- Maximum historical span: 72 hours
- Target horizons: 24 and 48 hours
- Target-time tolerance: +/-3 hours
- Area threshold: 20 pixels

| Horizon | Eligible genotypes | Eligible plants | Candidate windows |
|---|---:|---:|---:|
| 24 h | {h24["genotypes_eligible"]}/{h24["genotypes"]} | {h24["eligible"]}/{h24["plants"]} | {h24["windows"]:,} |
| 48 h | {h48["genotypes_eligible"]}/{h48["genotypes"]} | {h48["eligible"]}/{h48["plants"]} | {h48["windows"]:,} |

At 24 hours, {h24["genotypes_3plus"]} genotypes
have at least three eligible candidate plants.

At 48 hours, {h48["genotypes_3plus"]} genotypes
have at least three eligible candidate plants.

These counts are candidate time-aware windows,
not validated biologically continuous windows.

## 6. Trajectory QC — E0.6/E0.7

- Tracks with >=4 detections: {d["tracks"]:,}
- Tracks with internal missing observations: {d["missing_tracks"]:,}
- Tracks containing sharp decreases: {d["drop_tracks"]:,}
- Tracks containing sharp increases: {d["increase_tracks"]:,}

Adjacent-frame suspicious events:
- Sharp decreases: {d["drop_events"]:,}
- Sharp increases: {d["increase_events"]:,}
- Total: {d["drop_events"] + d["increase_events"]:,}

Threshold definitions:
- Sharp decrease: current area below 50%
  of preceding adjacent-frame area.
- Sharp increase: current area over 200%
  of preceding adjacent-frame area.

These are diagnostic criteria, NOT biological
error labels or automatic exclusion rules.

## 7. Identity consistency — E0.8 to E0.12

E0.10 anomaly-selected audit:
- Transitions examined: {d["e10_transitions"]}
- Transitions with strong matches:
  {d["e10_flagged_transitions"]}
- Strong cross-color matches: {d["e10_matches"]}

E0.11 stratified random audit:
- Transitions sampled: {d["e11_sample"]}
- Detector-positive transitions: {d["e11_flagged"]}
- Strong matches: {d["e11_matches"]}
- Observed positive fraction:
  {percent(d["e11_flagged"], d["e11_sample"]):.2f}%

The random sample had a capped number of
transitions per genotype and plant. Its flagged
fraction is NOT a validated population error rate.

E0.12 visual inspection:
- Three randomly sampled detector-positive
  examples were manually examined.
- All three remain UNCERTAIN.
- Detector-negative controls await manual review.

Earlier anomaly-selected cases provided stronger
visual indications of color reassignment, but
those cases are not prevalence estimates.

## 8. Scientific target definition

We forecast future visible projected leaf area
derived from color-coded segmentation masks.

This is not:
- True physical leaf surface area
- Leaf biomass
- Occlusion-corrected area
- Guaranteed biologically validated identity

The intended color semantics support tracking,
but individual annotation deviations remain possible.

## 9. Threats to validity

1. Annotation color/identity changes.
2. Occlusion and leaf orientation changes.
3. Missing observations and target-selection bias.
4. Genotype-specific acquisition schedules.
5. Correlation among overlapping leaf windows.
6. Possible environmental or experimental-batch
   confounding with genotype identity.
7. Unequal plant counts across genotypes.
8. Potential uncertainty under unseen-genotype shift.

## 10. Validation conclusion

**Preliminary dataset feasibility: PASS.**

**Full biological identity validation: PENDING.**

**Publication-ready experimental protocol:
PROVISIONAL, not frozen.**

The next stage should build leakage-safe
leave-one-genotype-out windows and evaluate simple
forecasting baselines before complex architectures.

No raw data or annotations were modified.

## Citation

Sarić, R., Azam, B., Khan, S., & Čustović, E.
(2026). PhenoLeaf-TS: A Time-Series Benchmark
for Leaf Instance Segmentation, Tracking,
and Growth Stage Classification.
ECCV 2026.
"""


def make_protocol():
    return """# Research Protocol v1 — DRAFT

## Title

Reliable Individual-Leaf Growth Forecasting
Under Genotype Shift

## Research question

How accurately can models forecast future visible
projected area of individual Arabidopsis leaves
when evaluated on genotypes not observed during
training, and how reliable are their uncertainty
estimates under that genotype shift?

## 1. Study design

Dataset: PhenoLeaf-TS.
Species: Arabidopsis thaliana.
Unit of biological replication: plant sequence,
subject to original acquisition documentation.

Primary task: oracle-segmentation area forecasting.

Prediction horizons:
- 24 hours
- 48 hours

Historical input:
- Six observed area values per individual leaf.
- Relative elapsed hours for each input.
- Target forecast horizon in hours.
- Optional leaf age since first observed detection,
  used only if available at prediction cutoff.

Target:
- Observed color-coded visible projected
  leaf area at target time.

Target tolerance:
- +/-3 hours, reported explicitly.
- Analyze sensitivity to stricter tolerance.

History maximum span:
- 72 hours, provisional.

## 2. Outer evaluation

Leave-one-genotype-out (LOGO).

For each of 21 outer evaluations:
1. Hold out one entire genotype as test.
2. Use the other 20 for development.
3. Do not inspect held-out target labels to
   select models or tuning hyperparameters.
4. Fit preprocessing on development data only.
5. Train the selected method on development data.
6. Evaluate once on the held-out genotype.

All plants, leaves, and windows from a held-out
genotype must remain outside model fitting.

## 3. Inner validation

Use genotype-disjoint validation within the
20 development genotypes.

Recommended: four folds, five genotypes held out
per inner fold, subject to balanced grouping.

Hyperparameters and any QC thresholds learned
from data must be selected without outer-test labels.

Model selection must use macro genotype-level
performance, not pooled window-level performance.

## 4. Forecast window construction

At a prediction cutoff:
- Historical inputs must precede or equal cutoff.
- Use six actually observed leaf-area values.
- Keep their actual timestamps.
- Do not interpolate using observations after cutoff.
- Require a future observed target within tolerance.
- Record observation gaps and target-time offset.
- Keep missingness separate from true zero area.

Construct windows only after genotype assignment,
or enforce explicit genotype membership in every
generated window manifest.

Do not allow a leaf's future target values to
enter historical features.

## 5. QC protocol

Primary benchmark:
- Retain documented color IDs 1–31.
- Preserve original measured visible areas.
- Do not silently correct identity labels.
- Flag suspect changes and uncertain observations.

Sensitivity benchmark:
- Exclude or censor windows crossing manually
  validated identity inconsistencies.
- Report the number of affected plants,
  leaves, windows, and genotypes.
- Use a transparent, predeclared QC rule.

Specific exception:
- Is-1/rep_24 contains undocumented colors.
- Compare main results with and without this plant.

Do not define exclusion using test prediction errors.

## 6. Initial baseline models

B0 — Persistence:
    Predict last observed area.

B1 — Time-aware linear extrapolation:
    Fit area as a function of elapsed time over
    the six historical observations, then forecast.

B2 — Ridge regression:
    Use historical area, changes, timestamps,
    current area, historical span, and horizon.

B3 — Gradient-boosted trees:
    Use the same causal historical features.

Advanced architectures are not justified
until these baselines are established.

## 7. Input normalization

Area is measured in pixels.

Potential transformations:
- log1p(area)
- ratios relative to current area
- elapsed time in hours

Fit any learned normalization parameters
on development/training folds only.

Do not compute normalization statistics from
the held-out genotype.

## 8. Evaluation

Evaluate 24h and 48h separately.

Primary point metrics:
- MAE in pixels
- RMSE in pixels

Additional:
- Normalized MAE with denominator explicitly defined
- Error by genotype
- Error by plant
- Error by leaf age / visible area size
- Error by acquisition schedule
- Error by quality flag

Aggregate window errors within plants first
to avoid treating overlapping windows as
independent replicates.

Report macro genotype performance.

For uncertainty:
- Empirical interval coverage
- Average prediction-interval width
- Coverage error relative to nominal coverage
- Genotype-specific calibration differences

Calibration must use only development genotypes.

No valid unseen-genotype coverage guarantee
is assumed merely from calibration on other
genotypes.

## 9. Statistical reporting

Report all 21 held-out genotype results.
Avoid selecting only favorable genotypes.

Use plant-clustered bootstrap intervals when
estimating within-genotype metric uncertainty.

Summarize genotype heterogeneity explicitly.

Compare baselines using paired genotype-level
differences with uncertainty intervals.

## 10. Threats to validity

- Ground-truth mask errors or identity shifts.
- Occlusion versus actual leaf expansion.
- Irregular acquisition intervals.
- Different developmental windows.
- Genotype and acquisition-batch confounding.
- Variable numbers of independent plants.
- Selection bias requiring visible future targets.

## 11. Publication contribution hypothesis

Potential contribution:
A systematic genotype-held-out benchmark for
time-aware future individual-leaf visible-area
prediction, with uncertainty evaluation and
annotation-quality sensitivity analysis.

Novelty is a hypothesis requiring confirmation
from a comprehensive literature review.

## 12. Next experiment

E1.0 — Leakage-safe forecasting window manifest.

Deliverables:
- One immutable row per forecast candidate.
- Explicit genotype, plant, and leaf identifiers.
- Six historical timestamps and areas.
- Prediction cutoff and future target timestamp.
- QC flags, target-time offset, and split metadata.
- One descriptive figure per experimental stage.

Do not train models before E1.0 integrity checks.
"""


def main():
    d = validation_data()

    VALIDATION.write_text(
        make_validation_report(d),
        encoding="utf-8",
    )

    PROTOCOL.write_text(
        make_protocol(),
        encoding="utf-8",
    )

    print("=" * 65)
    print("E0 — DATASET VALIDATION FINALIZATION")
    print("=" * 65)
    print("24h windows:", d["horizons"][24]["windows"])
    print("48h windows:", d["horizons"][48]["windows"])
    print("Leaf tracks:", d["tracks"])
    print("Strong E0.11 signals:", d["e11_flagged"])
    print()
    print("Created:", VALIDATION)
    print("Created:", PROTOCOL)
    print()
    print("Status: PROTOCOL DRAFT — NOT YET FROZEN")


if __name__ == "__main__":
    main()
