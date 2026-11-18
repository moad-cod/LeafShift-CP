
from pathlib import Path
from collections import defaultdict, Counter
import csv
import json
import hashlib

# ============================================================
# E1.1 — GENOTYPE-DISJOINT FOLD VALIDATION
#
# 21 leave-one-genotype-out outer evaluations
# 4 genotype-disjoint inner folds per outer evaluation
#
# Does NOT train models.
# Does NOT duplicate the forecasting manifest.
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    ROOT / "data/processed/forecasting"
    / "forecast_windows.csv"
)

OUT = ROOT / "experiments/E1"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"
FIGURES = OUT / "figures"

CONFIGS = ROOT / "configs"

for folder in (TABLES, REPORTS, FIGURES, CONFIGS):
    folder.mkdir(parents=True, exist_ok=True)

OUTER_CSV = TABLES / "E1_1_outer_folds.csv"
INNER_CSV = TABLES / "E1_1_inner_folds.csv"
GENOTYPE_CSV = TABLES / "E1_1_genotype_distribution.csv"
FOLD_JSON = CONFIGS / "E1_1_genotype_folds.json"
REPORT = REPORTS / "E1_1_fold_integrity_report.md"

EXPECTED_GENOTYPES = 21
EXPECTED_PLANTS = 318
EXPECTED_WINDOWS = 228645
HORIZONS = (24, 48)

INNER_FOLDS = 4
VALIDATION_GENOTYPES_PER_FOLD = 5


def save_csv(path, rows, fieldnames):
    with path.open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames
        )
        writer.writeheader()
        writer.writerows(rows)


def load_distribution():
    """
    Stream the manifest.

    Only statistics are stored in memory, not
    the 228,645 forecasting examples.
    """

    window_counts = Counter()
    plant_sets = defaultdict(set)
    all_window_ids = set()

    total = 0

    with MANIFEST.open(
        newline="", encoding="utf-8"
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            genotype = row["genotype"]
            plant_id = row["plant_id"]
            horizon = int(row["horizon_hours"])
            window_id = row["window_id"]

            if not genotype or not plant_id:
                raise ValueError(
                    "Missing genotype or plant identifier"
                )

            if not plant_id.startswith(genotype + "/"):
                raise ValueError(
                    f"Genotype/plant mismatch: {plant_id}"
                )

            if horizon not in HORIZONS:
                raise ValueError(
                    f"Unexpected horizon: {horizon}"
                )

            if window_id in all_window_ids:
                raise ValueError(
                    f"Duplicate window ID: {window_id}"
                )

            all_window_ids.add(window_id)
            window_counts[(genotype, horizon)] += 1
            plant_sets[genotype].add(plant_id)
            total += 1

    genotypes = sorted(plant_sets)

    if len(genotypes) != EXPECTED_GENOTYPES:
        raise AssertionError(
            f"Expected 21 genotypes; found {len(genotypes)}"
        )

    n_plants = sum(
        len(plants) for plants in plant_sets.values()
    )

    if n_plants != EXPECTED_PLANTS:
        raise AssertionError(
            f"Expected 318 plants; found {n_plants}"
        )

    if total != EXPECTED_WINDOWS:
        raise AssertionError(
            f"Expected {EXPECTED_WINDOWS} windows; "
            f"found {total}"
        )

    distribution = []

    for genotype in genotypes:
        n24 = window_counts[(genotype, 24)]
        n48 = window_counts[(genotype, 48)]

        if n24 == 0 or n48 == 0:
            raise AssertionError(
                f"Genotype lacks a horizon: {genotype}"
            )

        distribution.append({
            "genotype": genotype,
            "plants": len(plant_sets[genotype]),
            "windows_24h": n24,
            "windows_48h": n48,
            "windows_total": n24 + n48,
        })

    return distribution, total


def make_inner_groups(development, plant_counts):
    """
    Partition 20 development genotypes into
    four groups of five.

    Greedy balancing uses plant counts only.
    Each genotype belongs to exactly one
    inner validation group.
    """
    ordered = sorted(
        development,
        key=lambda g: (-plant_counts[g], g),
    )

    groups = [[] for _ in range(INNER_FOLDS)]
    loads = [0] * INNER_FOLDS

    for genotype in ordered:
        candidates = [
            i for i in range(INNER_FOLDS)
            if len(groups[i])
            < VALIDATION_GENOTYPES_PER_FOLD
        ]

        chosen = min(
            candidates,
            key=lambda i: (loads[i], i),
        )

        groups[chosen].append(genotype)
        loads[chosen] += plant_counts[genotype]

    assert all(len(group) == 5 for group in groups)

    return [sorted(group) for group in groups]


def sum_metric(names, by_genotype, field):
    return sum(
        by_genotype[g][field] for g in names
    )


def main():
    print("=" * 70)
    print("E1.1 — GENOTYPE-DISJOINT EVALUATION")
    print("=" * 70)

    distribution, total = load_distribution()

    genotypes = [
        row["genotype"] for row in distribution
    ]

    by_genotype = {
        row["genotype"]: row
        for row in distribution
    }

    plants_per_genotype = {
        row["genotype"]: row["plants"]
        for row in distribution
    }

    outer_rows = []
    inner_rows = []
    fold_definitions = []

    for outer_index, test_genotype in enumerate(
        genotypes, start=1
    ):
        development = [
            g for g in genotypes
            if g != test_genotype
        ]

        assert len(development) == 20

        outer_rows.append({
            "outer_fold": outer_index,
            "test_genotype": test_genotype,
            "development_genotypes": len(development),
            "development_plants": sum_metric(
                development, by_genotype, "plants"
            ),
            "test_plants": by_genotype[
                test_genotype
            ]["plants"],
            "development_windows": sum_metric(
                development,
                by_genotype,
                "windows_total",
            ),
            "test_windows": by_genotype[
                test_genotype
            ]["windows_total"],
        })

        groups = make_inner_groups(
            development, plants_per_genotype
        )

        # Every development genotype must appear
        # exactly once as an inner validation genotype.
        flattened = [
            g for group in groups
            for g in group
        ]

        assert len(flattened) == 20
        assert set(flattened) == set(development)
        assert len(set(flattened)) == 20

        inner_definitions = []

        for inner_index, validation in enumerate(
            groups, start=1
        ):
            train = [
                g for g in development
                if g not in set(validation)
            ]

            train_set = set(train)
            validation_set = set(validation)
            test_set = {test_genotype}

            assert len(train) == 15
            assert len(validation) == 5

            assert train_set.isdisjoint(validation_set)
            assert train_set.isdisjoint(test_set)
            assert validation_set.isdisjoint(test_set)

            assert (
                train_set | validation_set | test_set
            ) == set(genotypes)

            inner_rows.append({
                "outer_fold": outer_index,
                "inner_fold": inner_index,
                "test_genotype": test_genotype,
                "train_genotypes": ";".join(train),
                "validation_genotypes": ";".join(
                    validation
                ),
                "n_train_genotypes": len(train),
                "n_validation_genotypes": len(validation),
                "n_test_genotypes": 1,
                "train_plants": sum_metric(
                    train, by_genotype, "plants"
                ),
                "validation_plants": sum_metric(
                    validation, by_genotype, "plants"
                ),
                "test_plants": by_genotype[
                    test_genotype
                ]["plants"],
                "train_windows": sum_metric(
                    train,
                    by_genotype,
                    "windows_total",
                ),
                "validation_windows": sum_metric(
                    validation,
                    by_genotype,
                    "windows_total",
                ),
                "test_windows": by_genotype[
                    test_genotype
                ]["windows_total"],
                "disjoint": 1,
            })

            inner_definitions.append({
                "inner_fold": inner_index,
                "train_genotypes": train,
                "validation_genotypes": validation,
            })

        fold_definitions.append({
            "outer_fold": outer_index,
            "test_genotype": test_genotype,
            "development_genotypes": development,
            "inner_folds": inner_definitions,
        })

    save_csv(
        GENOTYPE_CSV,
        distribution,
        [
            "genotype",
            "plants",
            "windows_24h",
            "windows_48h",
            "windows_total",
        ],
    )

    save_csv(
        OUTER_CSV,
        outer_rows,
        [
            "outer_fold",
            "test_genotype",
            "development_genotypes",
            "development_plants",
            "test_plants",
            "development_windows",
            "test_windows",
        ],
    )

    save_csv(
        INNER_CSV,
        inner_rows,
        [
            "outer_fold",
            "inner_fold",
            "test_genotype",
            "train_genotypes",
            "validation_genotypes",
            "n_train_genotypes",
            "n_validation_genotypes",
            "n_test_genotypes",
            "train_plants",
            "validation_plants",
            "test_plants",
            "train_windows",
            "validation_windows",
            "test_windows",
            "disjoint",
        ],
    )

    config = {
        "experiment": "E1.1",
        "evaluation": "leave_one_genotype_out",
        "outer_folds": len(fold_definitions),
        "inner_folds_per_outer": INNER_FOLDS,
        "inner_assignment": (
            "deterministic balancing by plant count"
        ),
        "manifest": str(MANIFEST.relative_to(ROOT)),
        "note": (
            "Folds specify membership by genotype. "
            "Calibration groups must be defined separately "
            "inside development data before uncertainty "
            "models are evaluated."
        ),
        "folds": fold_definitions,
    }

    FOLD_JSON.write_text(
        json.dumps(config, indent=2),
        encoding="utf-8",
    )

    report = [
        "# E1.1 — Genotype-Disjoint Fold Integrity",
        "",
        f"- Manifest windows: {total:,}",
        f"- Genotypes: {len(genotypes)}",
        f"- Outer folds: {len(outer_rows)}",
        f"- Inner folds total: {len(inner_rows)}",
        "- Outer test: one genotype",
        "- Inner validation: five genotypes",
        "- Inner training: fifteen genotypes",
        "",
        "## Integrity checks",
        "",
        "- Every genotype used once as outer test.",
        "- Test genotype absent from development.",
        "- Each development genotype used once as "
        "inner validation per outer fold.",
        "- Train, validation, and test sets disjoint.",
        "- Every window identified by its genotype.",
        "- No plant appears under multiple genotypes.",
        "",
        "## Methodological qualifications",
        "",
        "The full manifest includes target labels.",
        "Training scripts must explicitly select only "
        "their assigned genotype rows.",
        "",
        "This step validates partition membership, "
        "not learned-feature or preprocessing leakage.",
        "",
        "No learned normalization, feature selection, "
        "model fitting, or calibration is performed here.",
        "",
        "Split balancing by plant count does not "
        "guarantee balanced growth stages or batches.",
        "",
        "Uncertainty calibration requires a separate "
        "development-only strategy in later experiments.",
    ]

    REPORT.write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )

    # Representative plot: candidate windows per genotype
    try:
        import matplotlib.pyplot as plt

        names = [
            row["genotype"] for row in distribution
        ]
        values = [
            row["windows_total"] for row in distribution
        ]

        fig, ax = plt.subplots(figsize=(11, 8))
        ax.barh(names, values)
        ax.invert_yaxis()
        ax.set_xlabel(
            "Candidate forecasting windows (24h + 48h)"
        )
        ax.set_ylabel("Genotype")
        ax.set_title(
            "E1.1 — Forecasting Window Distribution"
        )
        fig.tight_layout()
        fig.savefig(
            FIGURES / "E1_1_windows_per_genotype.png",
            dpi=200,
        )
        plt.close(fig)

    except ImportError:
        print("Matplotlib not installed; plot skipped.")

    print("\nRESULTS")
    print("Genotypes:", len(genotypes))
    print("Plants:", EXPECTED_PLANTS)
    print("Manifest windows:", total)
    print("Outer folds:", len(outer_rows))
    print("Inner folds:", len(inner_rows))

    print("\nFirst three outer folds:")

    for row in outer_rows[:3]:
        print(
            f"  Fold {row['outer_fold']:2d}: "
            f"test={row['test_genotype']:10s} "
            f"test_plants={row['test_plants']:2d} "
            f"test_windows={row['test_windows']:,}"
        )

    print("\nSaved:")
    print(FOLD_JSON)
    print(OUTER_CSV)
    print(INNER_CSV)
    print(REPORT)

    print("\nE1.1 PASSED — genotype-disjoint folds validated.")


if __name__ == "__main__":
    main()
