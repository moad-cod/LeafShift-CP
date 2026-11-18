
from pathlib import Path
from collections import Counter, defaultdict
import csv
import json

# ============================================================
# E2.0 — LEAKAGE-SAFE UNCERTAINTY PROTOCOL
#
# Project:
# Reliable Individual-Leaf Growth Forecasting
# Under Genotype Shift
#
# Purpose:
# 1. Audit E1.1 genotype-disjoint folds.
# 2. Define fitting / calibration / test roles.
# 3. Count plants and forecast windows per role.
# 4. Save reproducible E2 protocol and figures.
#
# NO MODEL TRAINING.
# NO TEST-GENOTYPE CALIBRATION.
# NO ORIGINAL DATA MODIFICATION.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "data/processed/forecasting/"
    "forecast_windows.csv"
)

FOLDS_PATH = (
    ROOT / "configs/E1_1_genotype_folds.json"
)

OUT = ROOT / "experiments/E2"
TABLES = OUT / "tables"
FIGURES = OUT / "figures"
REPORTS = OUT / "reports"

CONFIGS = ROOT / "configs"

for folder in (TABLES, FIGURES, REPORTS, CONFIGS):
    folder.mkdir(parents=True, exist_ok=True)

SPLITS_CSV = TABLES / "E2_0_split_audit.csv"
GENOTYPE_CSV = TABLES / "E2_0_genotype_distribution.csv"
CONFIG_JSON = CONFIGS / "E2_0_uncertainty_protocol.json"
REPORT_MD = REPORTS / "E2_0_uncertainty_protocol.md"

HORIZONS = (24, 48)
EXPECTED_GENOTYPES = 21
EXPECTED_PLANTS = 318
EXPECTED_WINDOWS = {24: 121431, 48: 107214}

CALIBRATION_INNER_FOLD = 1
TARGET_COVERAGE = 0.90


def save_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to save: {path}")

    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0])
        )
        writer.writeheader()
        writer.writerows(rows)


def load_manifest_counts():
    windows = Counter()
    plants = defaultdict(set)
    all_plant_genotypes = {}
    window_ids = set()

    with MANIFEST.open(
        "r", newline="", encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            genotype = row["genotype"]
            plant = row["plant_id"]
            horizon = int(row["horizon_hours"])
            window_id = row["window_id"]

            if horizon not in HORIZONS:
                raise ValueError(
                    f"Unexpected horizon: {horizon}"
                )

            if not plant.startswith(genotype + "/"):
                raise ValueError(
                    f"Plant/genotype mismatch: {plant}"
                )

            if plant in all_plant_genotypes:
                if all_plant_genotypes[plant] != genotype:
                    raise ValueError(
                        f"Plant spans genotypes: {plant}"
                    )
            else:
                all_plant_genotypes[plant] = genotype

            if window_id in window_ids:
                raise ValueError(
                    f"Duplicate window ID: {window_id}"
                )

            window_ids.add(window_id)
            plants[genotype].add(plant)
            windows[(genotype, horizon)] += 1

    genotypes = sorted(plants)

    if len(genotypes) != EXPECTED_GENOTYPES:
        raise AssertionError(
            f"Expected {EXPECTED_GENOTYPES} genotypes, "
            f"found {len(genotypes)}"
        )

    if len(all_plant_genotypes) != EXPECTED_PLANTS:
        raise AssertionError(
            f"Expected {EXPECTED_PLANTS} plants, "
            f"found {len(all_plant_genotypes)}"
        )

    for horizon in HORIZONS:
        actual = sum(
            windows[(g, horizon)]
            for g in genotypes
        )

        if actual != EXPECTED_WINDOWS[horizon]:
            raise AssertionError(
                f"{horizon}h count mismatch: {actual}"
            )

    distribution = []

    for genotype in genotypes:
        distribution.append({
            "genotype": genotype,
            "plants": len(plants[genotype]),
            "windows_24h": windows[(genotype, 24)],
            "windows_48h": windows[(genotype, 48)],
        })

    return genotypes, windows, plants, distribution


def count_roles(genotype_names, windows, plants, horizon):
    return {
        "genotypes": len(genotype_names),
        "plants": sum(
            len(plants[g])
            for g in genotype_names
        ),
        "windows": sum(
            windows[(g, horizon)]
            for g in genotype_names
        ),
    }


def validate_folds(
    folds, all_genotypes, windows, plants
):
    expected = set(all_genotypes)
    rows = []
    chosen_splits = []

    seen_outer_tests = set()

    if len(folds) != EXPECTED_GENOTYPES:
        raise AssertionError(
            "Expected 21 outer folds"
        )

    for outer in folds:
        test = outer["test_genotype"]

        if test in seen_outer_tests:
            raise AssertionError(
                f"Repeated outer test genotype: {test}"
            )

        seen_outer_tests.add(test)

        development = set(
            outer["development_genotypes"]
        )

        if len(development) != 20:
            raise AssertionError(
                f"Expected 20 development genotypes: {test}"
            )

        if test in development:
            raise AssertionError(
                f"Test genotype leaked into development: {test}"
            )

        if development | {test} != expected:
            raise AssertionError(
                f"Development coverage mismatch: {test}"
            )

        inner_folds = outer["inner_folds"]

        if len(inner_folds) != 4:
            raise AssertionError(
                "Expected four inner folds"
            )

        validation_occurrences = Counter()

        for inner in inner_folds:
            fit = set(inner["train_genotypes"])
            calibration = set(
                inner["validation_genotypes"]
            )

            if len(fit) != 15:
                raise AssertionError(
                    f"Expected 15 fitting genotypes: {test}"
                )

            if len(calibration) != 5:
                raise AssertionError(
                    f"Expected five calibration genotypes: {test}"
                )

            if fit & calibration:
                raise AssertionError(
                    "Fitting/calibration genotype overlap"
                )

            if test in fit or test in calibration:
                raise AssertionError(
                    f"Test leakage in fold for {test}"
                )

            if fit | calibration != development:
                raise AssertionError(
                    f"Development partition mismatch: {test}"
                )

            validation_occurrences.update(
                calibration
            )

            for horizon in HORIZONS:
                fit_n = count_roles(
                    fit, windows, plants, horizon
                )
                cal_n = count_roles(
                    calibration, windows, plants, horizon
                )
                test_n = count_roles(
                    {test}, windows, plants, horizon
                )

                if min(
                    fit_n["windows"],
                    cal_n["windows"],
                    test_n["windows"]
                ) <= 0:
                    raise AssertionError(
                        "A split contains no forecast windows"
                    )

                if (
                    fit_n["windows"]
                    + cal_n["windows"]
                    + test_n["windows"]
                    != EXPECTED_WINDOWS[horizon]
                ):
                    raise AssertionError(
                        "Split window totals do not match"
                    )

                rows.append({
                    "outer_fold": outer["outer_fold"],
                    "inner_fold": inner["inner_fold"],
                    "test_genotype": test,
                    "horizon_hours": horizon,
                    "fit_genotypes": 15,
                    "calibration_genotypes": 5,
                    "test_genotypes": 1,
                    "fit_plants": fit_n["plants"],
                    "calibration_plants": cal_n["plants"],
                    "test_plants": test_n["plants"],
                    "fit_windows": fit_n["windows"],
                    "calibration_windows": cal_n["windows"],
                    "test_windows": test_n["windows"],
                    "primary_protocol": int(
                        inner["inner_fold"]
                        == CALIBRATION_INNER_FOLD
                    ),
                    "disjoint": 1,
                })

            if (
                inner["inner_fold"]
                == CALIBRATION_INNER_FOLD
            ):
                chosen_splits.append({
                    "outer_fold": outer["outer_fold"],
                    "test_genotype": test,
                    "fit_genotypes": sorted(fit),
                    "calibration_genotypes": sorted(
                        calibration
                    ),
                    "test_genotypes": [test],
                    "inner_fold": inner["inner_fold"],
                })

        if set(validation_occurrences) != development:
            raise AssertionError(
                f"Incomplete inner validation rotation: {test}"
            )

        if any(
            validation_occurrences[g] != 1
            for g in development
        ):
            raise AssertionError(
                f"Repeated inner validation genotype: {test}"
            )

    if seen_outer_tests != expected:
        raise AssertionError(
            "Not every genotype was held out once"
        )

    if len(chosen_splits) != 21:
        raise AssertionError(
            "Expected 21 primary calibration splits"
        )

    return rows, chosen_splits


def make_plot(rows):
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("Matplotlib unavailable; figure skipped.")
        return

    primary = [
        row for row in rows
        if row["primary_protocol"] == 1
        and row["horizon_hours"] == 24
    ]

    primary.sort(key=lambda r: r["outer_fold"])

    labels = [r["test_genotype"] for r in primary]
    fitting = [r["fit_plants"] for r in primary]
    calibration = [
        r["calibration_plants"] for r in primary
    ]
    testing = [r["test_plants"] for r in primary]

    x = range(len(primary))

    fig, ax = plt.subplots(figsize=(12, 7))

    ax.barh(x, fitting, label="Model fitting")
    ax.barh(
        x, calibration,
        left=fitting,
        label="Calibration",
    )
    ax.barh(
        x, testing,
        left=[
            a + b for a, b in zip(
                fitting, calibration
            )
        ],
        label="Held-out test",
    )

    ax.set_yticks(list(x))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()

    ax.set_xlabel("Number of plant sequences")
    ax.set_title(
        "E2.0 — Genotype-Disjoint Uncertainty Splits"
    )
    ax.legend()
    fig.tight_layout()

    output = (
        FIGURES / "E2_0_calibration_split_distribution.png"
    )
    fig.savefig(output, dpi=200)
    plt.close(fig)

    print("Saved figure:", output)


def main():
    print("=" * 70)
    print("E2.0 — UNCERTAINTY CALIBRATION PROTOCOL")
    print("=" * 70)

    with FOLDS_PATH.open(encoding="utf-8") as f:
        config = json.load(f)

    genotypes, windows, plants, distribution = (
        load_manifest_counts()
    )

    audit_rows, primary_splits = validate_folds(
        config["folds"],
        genotypes,
        windows,
        plants,
    )

    save_csv(SPLITS_CSV, audit_rows)
    save_csv(GENOTYPE_CSV, distribution)

    protocol = {
        "experiment": "E2.0",
        "research_task": (
            "Uncertainty quantification for individual-leaf "
            "visible projected area under genotype shift"
        ),
        "forecast_horizons_hours": [24, 48],
        "nominal_coverage": TARGET_COVERAGE,
        "evaluation": "leave_one_genotype_out",
        "outer_folds": 21,
        "inner_folds_available": 4,
        "primary_calibration_inner_fold": (
            CALIBRATION_INNER_FOLD
        ),
        "model_fitting_genotypes": 15,
        "calibration_genotypes": 5,
        "test_genotypes": 1,
        "baseline_model": {
            "name": "HistGradientBoostingRegressor",
            "configuration": "E1.5 flexible",
            "max_iter": 150,
            "max_leaf_nodes": 31,
            "learning_rate": 0.05,
            "l2_regularization": 1.0,
            "max_bins": 64,
            "min_samples_leaf": 40,
            "early_stopping": False,
            "random_state": 20261108,
        },
        "calibration_method_next_phase": (
            "Absolute-residual split conformal, "
            "with observed empirical coverage evaluated "
            "under held-out genotype shift."
        ),
        "primary_metrics": [
            "empirical_window_coverage",
            "plant_averaged_coverage",
            "macro_genotype_coverage",
            "mean_prediction_interval_width",
            "per_genotype_coverage",
            "point_prediction_MAE",
        ],
        "additional_diagnostics": [
            "leaf_size_stratification",
            "history_gap_stratification",
            "growth_rate_variation",
            "calibration_split_sensitivity",
        ],
        "scientific_constraints": [
            "Never fit on test genotype",
            "Never calibrate using test genotype",
            "Never use future target area in predictors",
            "Do not treat overlapping windows as independent",
            "Do not claim distribution-free coverage under "
            "unverified genotype exchangeability",
            "Use calibration genotype folds without choosing "
            "the best fold from test performance",
            "Retain annotation quality warnings",
        ],
        "primary_splits": primary_splits,
    }

    CONFIG_JSON.write_text(
        json.dumps(protocol, indent=2),
        encoding="utf-8",
    )

    primary_rows = [
        row for row in audit_rows
        if row["primary_protocol"] == 1
    ]

    report_lines = [
        "# E2.0 — Uncertainty Calibration Protocol",
        "",
        "## Research question",
        "",
        "Can prediction intervals for future visible "
        "individual-leaf area remain empirically "
        "reliable on entirely unseen genotypes?",
        "",
        "## Experimental design",
        "",
        "- 21 outer test genotypes.",
        "- 15 fitting genotypes per outer fold.",
        "- 5 genotype-disjoint calibration genotypes.",
        "- Primary calibration assignment: inner fold 1.",
        "- Remaining inner folds reserved for sensitivity.",
        "- Separate evaluation at 24 and 48 hours.",
        "- Nominal interval coverage: 90%.",
        "",
        "## Integrity",
        "",
        f"- Total genotypes: {len(genotypes)}",
        f"- Total plants: {sum(map(len, plants.values()))}",
        f"- Total windows: {sum(EXPECTED_WINDOWS.values()):,}",
        f"- Outer / inner / horizon audit rows: "
        f"{len(audit_rows)}",
        f"- Primary evaluation rows: "
        f"{len(primary_rows)}",
        "- All fitting/calibration/test genotype sets disjoint.",
        "- All primary split roles nonempty.",
        "",
        "## Calibration strategy",
        "",
        "E2.1 will fit the baseline forecaster on the "
        "15 fitting genotypes, calculate absolute "
        "forecast residuals on the five calibration "
        "genotypes, and build 90% prediction intervals. "
        "Quantile selection and plant-block handling "
        "must be specified before evaluating test outcomes.",
        "",
        "## Scientific limitations",
        "",
        "- Calibration and test genotypes may not be "
        "exchangeable at the forecasting-window level.",
        "- Windows from the same plant overlap.",
        "- Standard pooled-window conformal quantiles "
        "do not automatically guarantee 90% held-out "
        "genotype coverage.",
        "- Plant-averaged and pooled-window coverage "
        "answer different questions.",
        "- These E2 models use 15 fitting genotypes, "
        "not E1.5's 20.",
        "- Calibration sample composition differs by fold.",
        "- The prediction target is visible projected "
        "leaf area, not anatomical area or biomass.",
        "- Annotation identity problems remain unresolved.",
        "",
        "## Next experiment",
        "",
        "E2.1 — Fit and calibrate the baseline on "
        "development genotypes only; evaluate intervals "
        "on the outer held-out genotype.",
    ]

    REPORT_MD.write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    make_plot(audit_rows)

    print("\nDATA VALIDATION")
    print("Genotypes:", len(genotypes))
    print(
        "Plants:",
        sum(len(p) for p in plants.values())
    )
    print(
        "Windows:",
        sum(EXPECTED_WINDOWS.values())
    )

    print("\nSPLIT VALIDATION")
    print("Outer folds:", len(primary_splits))
    print("Inner configurations:", len(audit_rows) // 2)
    print("Primary horizon-specific splits:", len(primary_rows))
    print("Fitting genotypes per split: 15")
    print("Calibration genotypes per split: 5")
    print("Test genotypes per split: 1")

    for horizon in HORIZONS:
        subset = [
            row for row in primary_rows
            if row["horizon_hours"] == horizon
        ]

        print(f"\n{horizon}h:")
        print(
            "  Minimum calibration plants:",
            min(r["calibration_plants"] for r in subset)
        )
        print(
            "  Minimum calibration windows:",
            min(r["calibration_windows"] for r in subset)
        )
        print(
            "  Minimum test plants:",
            min(r["test_plants"] for r in subset)
        )

    print("\nSaved:")
    print(SPLITS_CSV)
    print(GENOTYPE_CSV)
    print(CONFIG_JSON)
    print(REPORT_MD)

    print(
        "\nE2.0 PASSED — "
        "uncertainty split protocol validated."
    )


if __name__ == "__main__":
    main()