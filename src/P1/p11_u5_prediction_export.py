#!/usr/bin/env python3
"""P1.1: audited U5 window-level export, faithful to E3.2.

Run from project root: uv run --active python src/P1/run_p11_u5_prediction_export.py
No existing E3 results are modified. Requires the archived E3.2 fold metrics.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "E3"))
import run_cqr_baselines as cqr  # noqa: E402

OUT = ROOT / "experiments" / "P1" / "P1_1"
TABLES = OUT / "tables"
REPORTS = OUT / "reports"

KEY = ["horizon_hours", "outer_fold", "inner_fold", "test_genotype", "method"]
METRICS = ["threshold", "plants", "windows", "coverage", "width", "score"]


def find_reference() -> tuple[Path, pd.DataFrame]:
    """Find one authoritative E3.2 reference, never silently skip verification."""
    candidates = [
        ROOT / "experiments/E3/tables/E3_2_fold_metrics.csv",
        ROOT / "experiments/E3/tables/E3_2_cqr_fold_metrics.csv",
        ROOT / "experiments/E3/tables/E3_2_CQR_fold_metrics.csv",
    ]
    candidates += sorted((ROOT / "experiments/E3/tables").glob("*.csv"))
    valid = []
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        try:
            header = set(pd.read_csv(path, nrows=0).columns)
        except Exception:
            continue
        if not set(KEY + METRICS).issubset(header):
            continue
        df = pd.read_csv(path)
        rows = df.loc[df.method.eq("U5_CQR_balanced")].copy()
        if len(rows) == 168 and not rows.duplicated(KEY).any():
            valid.append((path, rows))
    if not valid:
        raise RuntimeError(
            "No 168-row U5 E3.2 fold reference located under experiments/E3/tables. "
            "Inspect table filenames and supply the original fold metrics before export."
        )
    # Prefer conventional E3_2 naming, while checking alternative copies agree.
    valid.sort(key=lambda item: ("E3_2" not in item[0].name, item[0].name))
    reference_path, reference = valid[0]
    for other_path, other in valid[1:]:
        a = reference.sort_values(KEY).reset_index(drop=True)
        b = other.sort_values(KEY).reset_index(drop=True)
        if not a[KEY].equals(b[KEY]) or not np.allclose(
            a[METRICS].to_numpy(float), b[METRICS].to_numpy(float),
            rtol=1e-8, atol=1e-8,
        ):
            raise RuntimeError(f"Conflicting U5 references: {reference_path} and {other_path}")
    print(f"Original U5 reference: {reference_path}", flush=True)
    return reference_path, reference.set_index(KEY)


def metadata_by_horizon() -> dict[int, dict[str, np.ndarray]]:
    fields = {
        h: {name: [] for name in [
            "window_id", "leaf_id", "history_span_hours", "actual_target_offset_hours",
            "history_growth_abs_px_per_hour", "history_growth_relative_per_hour",
            "cutoff_timestamp", "target_timestamp",
        ]}
        for h in cqr.HORIZONS
    }
    with cqr.MANIFEST.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            h = int(row["horizon_hours"])
            d = fields[h]
            times = np.asarray(json.loads(row["history_relative_hours"]), dtype=float)
            areas = np.asarray(json.loads(row["history_areas_pixels"]), dtype=float)
            span = float(times[-1] - times[0])
            if span <= 0 or not np.isfinite(span):
                raise ValueError(f"Invalid history span for {row['window_id']}")
            abs_growth = float((areas[-1] - areas[0]) / span)
            relative_growth = float((math.log(areas[-1]) - math.log(areas[0])) / span)
            values = {
                "window_id": row["window_id"],
                "leaf_id": row["leaf_id"],
                "history_span_hours": span,
                "actual_target_offset_hours": float(row["actual_target_offset_hours"]),
                "history_growth_abs_px_per_hour": abs_growth,
                "history_growth_relative_per_hour": relative_growth,
                "cutoff_timestamp": row["cutoff_timestamp"],
                "target_timestamp": row["target_timestamp"],
            }
            for name, value in values.items():
                d[name].append(value)
    return {h: {k: np.asarray(v) for k, v in d.items()} for h, d in fields.items()}


def check_reproduction(reference: pd.DataFrame, row: dict) -> None:
    key = tuple(row[k] for k in KEY)
    if key not in reference.index:
        raise AssertionError(f"Missing original U5 reference key {key}")
    baseline = reference.loc[key]
    for metric in METRICS:
        expected = float(baseline[metric])
        observed = float(row[metric])
        if not np.isclose(expected, observed, rtol=1e-8, atol=1e-7):
            raise AssertionError(
                f"E3.2 REPLICATION FAILED: {key}, metric={metric}, "
                f"original={expected:.12g}, new={observed:.12g}. "
                "No exported results may be used until corrected."
            )


def evaluate_export(d: dict, meta: dict, outer: dict, inner: dict, h: int,
                    reference: pd.DataFrame) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    fit_g = set(inner["train_genotypes"])
    cal_g = set(inner["validation_genotypes"])
    test_g = str(outer["test_genotype"])
    if len(fit_g) != 15 or len(cal_g) != 5 or fit_g & cal_g or test_g in fit_g | cal_g:
        raise AssertionError("LOGO split integrity failure")
    fit = np.isin(d["g"], list(fit_g))
    cal = np.isin(d["g"], list(cal_g))
    test = d["g"] == test_g
    if set(d["g"][fit]) != fit_g or set(d["g"][cal]) != cal_g or set(d["g"][test]) != {test_g}:
        raise AssertionError("Incomplete fold membership")

    w_fit = cqr.balanced_weights(d["g"][fit], d["p"][fit])
    lo_model = cqr.fit_quantile_model(d["X"][fit], d["y"][fit], w_fit, 0.05)
    hi_model = cqr.fit_quantile_model(d["X"][fit], d["y"][fit], w_fit, 0.95)
    lo_cal_raw = lo_model.predict(d["X"][cal])
    hi_cal_raw = hi_model.predict(d["X"][cal])
    cal_lo = np.minimum(lo_cal_raw, hi_cal_raw)
    cal_hi = np.maximum(lo_cal_raw, hi_cal_raw)
    scores = np.maximum(cal_lo - d["y"][cal], d["y"][cal] - cal_hi)
    w_cal = cqr.balanced_weights(d["g"][cal], d["p"][cal])
    q = max(0.0, cqr.weighted_quantile(scores, w_cal))
    lo_test_raw = lo_model.predict(d["X"][test])
    hi_test_raw = hi_model.predict(d["X"][test])
    lo_test = np.minimum(lo_test_raw, hi_test_raw)
    hi_test = np.maximum(lo_test_raw, hi_test_raw)
    lower = np.maximum(0.0, lo_test - q)
    upper = np.maximum(lower, hi_test + q)
    y = d["y"][test]
    p = d["p"][test]
    metrics = cqr.plant_metrics(y, lower, upper, p)
    common = {
        "horizon_hours": int(h), "outer_fold": int(outer["outer_fold"]),
        "inner_fold": int(inner["inner_fold"]), "test_genotype": test_g,
        "method": "U5_CQR_balanced",
    }
    row = {
        **common, "threshold": float(q),
        "calibration_plants": len(np.unique(d["p"][cal])),
        "calibration_windows": int(cal.sum()), **metrics,
    }
    check_reproduction(reference, row)

    covered = ((y >= lower) & (y <= upper)).astype(np.int8)
    test_meta = {name: vals[test] for name, vals in meta.items()}
    windows = pd.DataFrame({
        **{key: np.repeat(value, len(y)) for key, value in common.items()},
        "plant_id": p,
        **test_meta,
        "target_area_pixels": y,
        "lower_pixels": lower,
        "upper_pixels": upper,
        "covered": covered,
    })
    if windows["window_id"].duplicated().any():
        raise AssertionError("Duplicate test window ID within assignment")
    # Include per-plant results for bootstrapping without processing ~1 million windows again.
    plants = windows.groupby("plant_id", sort=True).agg(
        windows=("covered", "size"),
        coverage=("covered", "mean"),
    ).reset_index()
    for name, value in common.items():
        plants[name] = value
    return row, windows, plants


def main() -> None:
    print("P1.1 — U5 per-window export, exact E3.2 reproduction required", flush=True)
    reference_path, reference = find_reference()
    with cqr.FOLDS.open(encoding="utf-8") as fh:
        folds = json.load(fh)["folds"]
    if len(folds) != 21:
        raise AssertionError("Expected 21 outer folds")
    data = cqr.load_data()
    metadata = metadata_by_horizon()
    for h in cqr.HORIZONS:
        if len(data[h]["y"]) != len(metadata[h]["window_id"]):
            raise AssertionError("Manifest/E3 feature row count mismatch")
        if len(set(metadata[h]["window_id"])) != len(metadata[h]["window_id"]):
            raise AssertionError("Manifest window IDs not unique")
    TABLES.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    all_rows, all_plant_rows = [], []
    for h in cqr.HORIZONS:
        path = TABLES / f"P1_1_U5_window_predictions_{h}h.csv"
        # Open once; append verified folds only. Previous partial exports are replaced.
        if path.exists():
            path.unlink()
        first = True
        n = 0
        for i, outer in enumerate(folds, 1):
            for inner in outer["inner_folds"]:
                row, windows, plants = evaluate_export(
                    data[h], metadata[h], outer, inner, h, reference
                )
                windows.to_csv(path, mode="a", index=False, header=first)
                first = False
                all_rows.append(row)
                all_plant_rows.append(plants)
                n += len(windows)
            print(f"{h}h | U5 verified {i:02d}/21 | {outer['test_genotype']}", flush=True)
        expected = 4 * len(data[h]["y"])
        if n != expected:
            raise AssertionError(f"Expected {expected} exported test rows, found {n}")
        print(f"{h}h | {n:,} rows, 84/84 fold metrics matched E3.2", flush=True)
    folds_df = pd.DataFrame(all_rows)
    plants_df = pd.concat(all_plant_rows, ignore_index=True)
    if len(folds_df) != 168 or folds_df.duplicated(KEY).any():
        raise AssertionError("Expected 168 unique fold rows")
    if plants_df.duplicated(KEY + ["plant_id"]).any():
        raise AssertionError("Duplicate per-plant assignment key")
    folds_df.to_csv(TABLES / "P1_1_U5_verified_fold_metrics.csv", index=False)
    plants_df.to_csv(TABLES / "P1_1_U5_plant_assignment_coverage.csv", index=False)
    lines = [
        "# P1.1 U5 per-window prediction export", "",
        f"- E3.2 source: `{reference_path.relative_to(ROOT)}`",
        "- All 168 U5 fold thresholds and plant-aggregated metrics reproduced within tolerance (rtol=1e-8, atol=1e-7).",
        "- Training: original weighted 0.05/0.95 quantile HGB models; calibration: original weighted signed CQR scores and nonnegative threshold clipping.",
        "- Per-window outcome keys: horizon, test genotype, outer fold, inner fold, plant ID, leaf ID, window ID.",
        "- Historical absolute growth = (last area - first area) / history span in px/hour; log-relative growth = log(last / first) / history span.",
        "- Outcomes reuse the same held-out windows across four assignments; treat assignment as paired, not independent.",
        "- Metadata row linkage follows the verified E3/manifest positional order. This is not a cryptographic identity-level proof; preserve window IDs for audit.",
        "- No new method was selected or tuned; inference remains secondary exploratory.",
        "",
        "## Export sizes", "",
    ]
    for h in cqr.HORIZONS:
        subset = folds_df[folds_df.horizon_hours.eq(h)]
        grouped = subset.groupby("test_genotype").coverage.mean()
        lines.append(
            f"- {h}h: {int(subset.windows.sum()):,} window-assignment rows; "
            f"macro coverage {grouped.mean():.8f}; worst genotype {grouped.min():.8f}"
        )
    (REPORTS / "P1_1_U5_prediction_export_audit.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\nPASS: All 168 U5 E3.2 fold metrics replicated.")
    print(f"Outputs: {TABLES}")


if __name__ == "__main__":
    main()
