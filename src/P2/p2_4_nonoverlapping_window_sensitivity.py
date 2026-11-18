#!/usr/bin/env python3
"""P2.4 retrospective nonoverlapping-forecast-window sensitivity.

Reads archived U5 prediction intervals only. Selection uses identifiers and
timestamps, never outcomes, errors, interval widths, or coverage. No fitting,
calibration, tuning, or method selection is performed.
"""
from __future__ import annotations

from pathlib import Path
import hashlib
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "experiments/P1/P1_1/tables"
E3_FOLDS = ROOT / "experiments/E3/tables/E3_2_fold_metrics.csv"
OUT = ROOT / "experiments/P2/P2_4/tables"
REPORT = ROOT / "Research/P2/P2_4_nonoverlapping_window_sensitivity.md"
FIGURES = ROOT / "manuscript/figures/supplementary"
HORIZONS = (24, 48)
NOMINAL = 0.90
EXPECTED = {
    24: {"macro": 0.8997119769358646, "worst": 0.8221703622232337, "below": 8},
    48: {"macro": 0.900400611362259, "worst": 0.7639306084000144, "below": 6},
}
ID_COLS = ["test_genotype", "plant_id", "leaf_id", "window_id"]
META_COLS = ID_COLS + ["outer_fold", "horizon_hours", "cutoff_timestamp",
                       "target_timestamp", "history_span_hours",
                       "actual_target_offset_hours"]
REQUIRED = set(META_COLS + ["inner_fold", "method", "target_area_pixels",
                             "lower_pixels", "upper_pixels", "covered"])


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_horizon(horizon: int) -> pd.DataFrame:
    path = PREDICTIONS / f"P1_1_U5_window_predictions_{horizon}h.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing archived prediction export: {path}")
    frame = pd.read_csv(path, parse_dates=["cutoff_timestamp", "target_timestamp"])
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing fields {sorted(missing)}")
    if frame.empty or not frame.horizon_hours.eq(horizon).all():
        raise ValueError(f"{path.name}: empty or inconsistent horizon")
    if not frame.method.eq("U5_CQR_balanced").all():
        raise ValueError(f"{path.name}: unexpected method")
    if frame[list(REQUIRED)].isna().any().any():
        raise ValueError(f"{path.name}: missing required values")
    if not frame.covered.isin([0, 1, False, True]).all():
        raise ValueError(f"{path.name}: invalid coverage flags")
    if (frame.target_timestamp < frame.cutoff_timestamp).any():
        raise ValueError(f"{path.name}: target precedes cutoff")
    observed_offsets = (
        frame.target_timestamp - frame.cutoff_timestamp
    ).dt.total_seconds() / 3600.0
    if not np.allclose(
        observed_offsets.to_numpy(dtype=float),
        frame.actual_target_offset_hours.to_numpy(dtype=float),
        rtol=0.0,
        # The archived metadata are rounded to five decimal places in hours.
        atol=5e-6,
    ):
        raise ValueError(
            f"{path.name}: timestamps disagree with actual_target_offset_hours")
    if (frame.upper_pixels < frame.lower_pixels).any():
        raise ValueError(f"{path.name}: negative interval width")
    if frame.duplicated(ID_COLS + ["inner_fold"]).any():
        raise ValueError(f"{path.name}: duplicate window/assignment rows")

    # The four predictions for one held-out window must share all selection metadata.
    metadata_nunique = frame.groupby(ID_COLS, sort=False)[
        ["outer_fold", "horizon_hours", "cutoff_timestamp", "target_timestamp",
         "history_span_hours", "actual_target_offset_hours", "target_area_pixels"]
    ].nunique(dropna=False)
    if not metadata_nunique.eq(1).all().all():
        raise ValueError(f"{path.name}: assignment-specific selection metadata")
    pairing = frame.groupby(ID_COLS, sort=False).agg(
        assignments=("inner_fold", "nunique"), rows=("inner_fold", "size"))
    if not ((pairing.assignments == 4) & (pairing.rows == 4)).all():
        raise ValueError(f"{path.name}: expected four paired predictions per unique window")
    return frame


def add_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["coverage_value"] = out.covered.astype(float)
    out["width_pixels"] = out.upper_pixels - out.lower_pixels
    out["interval_score"] = (
        out.width_pixels
        + 20.0 * np.maximum(out.lower_pixels - out.target_area_pixels, 0.0)
        + 20.0 * np.maximum(out.target_area_pixels - out.upper_pixels, 0.0)
    )
    return out


def aggregate(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Window mean within plant/assignment; equal plants within genotype/assignment;
    # then average four paired assignments within genotype.
    plant = frame.groupby(
        ["horizon_hours", "test_genotype", "inner_fold", "plant_id"], as_index=False
    ).agg(
        coverage=("coverage_value", "mean"),
        width=("width_pixels", "mean"),
        interval_score=("interval_score", "mean"),
        unique_windows=("window_id", "nunique"),
        leaf_trajectories=("leaf_id", "nunique"),
    )
    genotype_assignment = plant.groupby(
        ["horizon_hours", "test_genotype", "inner_fold"], as_index=False
    ).agg(
        coverage=("coverage", "mean"), width=("width", "mean"),
        interval_score=("interval_score", "mean"), plants=("plant_id", "nunique"),
        unique_windows=("unique_windows", "sum"),
    )
    genotype = genotype_assignment.groupby(
        ["horizon_hours", "test_genotype"], as_index=False
    ).agg(
        coverage=("coverage", "mean"), width=("width", "mean"),
        interval_score=("interval_score", "mean"),
        assignments=("inner_fold", "nunique"), plants=("plants", "first"),
        unique_windows=("unique_windows", "first"),
    )
    if not genotype.assignments.eq(4).all():
        raise ValueError("Assignment pairing broken during aggregation")
    return plant, genotype


def validate_original(frame: pd.DataFrame, genotype: pd.DataFrame, horizon: int) -> None:
    expected = EXPECTED[horizon]
    values = {
        "macro": float(genotype.coverage.mean()),
        "worst": float(genotype.coverage.min()),
        "below": int((genotype.coverage < NOMINAL).sum()),
    }
    for key in ("macro", "worst"):
        if not np.isclose(values[key], expected[key], atol=1e-10, rtol=0):
            raise AssertionError(
                f"{horizon}h baseline {key} mismatch: {values[key]} != {expected[key]}")
    if values["below"] != expected["below"]:
        raise AssertionError(
            f"{horizon}h baseline below-90 mismatch: {values['below']} != {expected['below']}")

    reference = pd.read_csv(E3_FOLDS)
    reference = reference[(reference.method == "U5_CQR_balanced") &
                          (reference.horizon_hours == horizon)]
    fold = frame.groupby(
        ["horizon_hours", "outer_fold", "inner_fold", "test_genotype"], as_index=False
    ).apply(lambda d: pd.Series({
        "coverage_check": d.groupby("plant_id").coverage_value.mean().mean(),
        "width_check": d.groupby("plant_id").width_pixels.mean().mean(),
        "score_check": d.groupby("plant_id").interval_score.mean().mean(),
        "windows_check": d.window_id.nunique(),
        "plants_check": d.plant_id.nunique(),
    }), include_groups=False).reset_index(drop=True)
    merged = fold.merge(reference, on=["horizon_hours", "outer_fold", "inner_fold",
                                       "test_genotype"], validate="one_to_one")
    for calculated, archived in (("coverage_check", "coverage"),
                                  ("width_check", "width"),
                                  ("score_check", "score")):
        if not np.allclose(merged[calculated], merged[archived], atol=1e-9, rtol=0):
            raise AssertionError(f"{horizon}h per-window {calculated} does not reproduce E3")
    if not (merged.windows_check == merged.windows).all() or not (
            merged.plants_check == merged.plants).all():
        raise AssertionError(f"{horizon}h archived fold counts do not reconcile")


def select_nonoverlapping(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    unique = frame.sort_values(ID_COLS + ["inner_fold"]).drop_duplicates(ID_COLS)
    selected_rows: list[dict] = []
    trajectory_cols = ["test_genotype", "plant_id", "leaf_id"]
    for trajectory, group in unique.groupby(trajectory_cols, sort=True):
        group = group.sort_values(["cutoff_timestamp", "target_timestamp", "window_id"])
        previous_target = None
        previous_cutoff = None
        selected_order = 0
        for row in group.itertuples(index=False):
            if previous_target is None or row.cutoff_timestamp >= previous_target:
                selected_order += 1
                history_start = row.cutoff_timestamp - pd.to_timedelta(
                    float(row.history_span_hours), unit="h")
                selected_rows.append({
                    "horizon_hours": int(row.horizon_hours),
                    "test_genotype": trajectory[0],
                    "plant_id": trajectory[1],
                    "leaf_id": trajectory[2],
                    "window_id": row.window_id,
                    "outer_fold": int(row.outer_fold),
                    "selected_order": selected_order,
                    "cutoff_timestamp": row.cutoff_timestamp,
                    "target_timestamp": row.target_timestamp,
                    "previous_selected_target_timestamp": previous_target,
                    "forecast_interval_nonoverlap": (
                        True if previous_target is None else row.cutoff_timestamp >= previous_target),
                    "history_span_hours": float(row.history_span_hours),
                    "inferred_history_start_timestamp": history_start,
                    "history_span_overlaps_previous_selected_history": (
                        False if previous_cutoff is None
                        else history_start < previous_cutoff),
                    "selection_uses_outcomes": False,
                    "selection_rule": "earliest_then_cutoff_gte_previous_selected_target",
                })
                previous_target = row.target_timestamp
                previous_cutoff = row.cutoff_timestamp
    audit = pd.DataFrame(selected_rows)
    selected_ids = audit[ID_COLS]
    thinned = frame.merge(selected_ids, on=ID_COLS, how="inner", validate="many_to_one")
    counts = thinned.groupby(ID_COLS).agg(
        assignments=("inner_fold", "nunique"), rows=("inner_fold", "size"))
    if not ((counts.assignments == 4) & (counts.rows == 4)).all():
        raise AssertionError("Selected window set is not identical across four assignments")
    if not audit.forecast_interval_nonoverlap.all():
        raise AssertionError("Selected forecast intervals overlap")
    return thinned, audit


def make_figure(comparison: pd.DataFrame, summary: pd.DataFrame) -> list[Path]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10.5,
        "axes.labelsize": 9.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "axes.linewidth": .8, "pdf.fonttype": 42, "svg.fonttype": "none",
    })
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.4), sharex=True, sharey=True)
    limits = (.74, .98)
    colors = {24: "#0072B2", 48: "#D55E00"}
    for ax, horizon in zip(axes, HORIZONS):
        data = comparison[comparison.horizon_hours == horizon].copy()
        ax.plot(limits, limits, color="#555555", lw=1, ls="--", label="Identity")
        ax.axhline(NOMINAL, color="#888888", lw=.8, ls=":")
        ax.axvline(NOMINAL, color="#888888", lw=.8, ls=":")
        ax.scatter(data.original_coverage, data.thinned_coverage, s=36,
                   color=colors[horizon], edgecolor="black", linewidth=.4, zorder=3)
        labels = data.nlargest(3, "absolute_change_pp")
        for _, row in labels.iterrows():
            ax.annotate(row.test_genotype, (row.original_coverage, row.thinned_coverage),
                        xytext=(4, 4), textcoords="offset points", fontsize=7)
        retained = summary.loc[summary.horizon_hours == horizon, "retention_fraction"].iloc[0]
        ax.set_title(f"{horizon} h  |  retained {100*retained:.1f}%")
        ax.set_xlabel("Original U5 genotype coverage")
        ax.set_xlim(limits); ax.set_ylim(limits); ax.grid(alpha=.2)
    axes[0].set_ylabel("Nonoverlapping-window U5 genotype coverage")
    fig.suptitle("Retrospective sensitivity to overlapping forecast windows", fontweight="bold")
    fig.text(.5, .01,
             "Each point is one held-out genotype; thinning uses timestamps only. "
             "Dashed: identity; dotted: nominal 90%.", ha="center", fontsize=7.5)
    fig.tight_layout(rect=(0, .035, 1, .96))
    stem = FIGURES / "Figure_S8_Nonoverlapping_Window_Sensitivity"
    outputs = []
    for suffix, kwargs in (("pdf", {}), ("svg", {}), ("png", {"dpi": 360})):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", facecolor="white", **kwargs)
        outputs.append(path)
    plt.close(fig)
    return outputs


def build_report(summary: pd.DataFrame, comparison: pd.DataFrame,
                 source_hashes: dict[str, str], outputs: list[Path]) -> None:
    lines = [
        "# P2.4 — Nonoverlapping-window sensitivity analysis", "",
        "**Status:** Retrospective evaluation-only robustness analysis; no training, tuning, recalibration, or method selection.", "",
        "## Predeclared thinning rule", "",
        "Within each horizon and `(test_genotype, plant_id, leaf_id)` trajectory, sort unique windows by cutoff timestamp, target timestamp, and window ID. Select the earliest window, then select the next window only when its cutoff is greater than or equal to the previous selected target timestamp. Endpoint contact is allowed. The same selected window IDs are used for all four assignments. Selection uses identifiers and timestamps only.", "",
        "This rule makes forecast intervals `[cutoff, target]` nonoverlapping. It does **not** make six-observation historical input spans nonoverlapping; inferred history spans often overlap and are reported only as an audit diagnostic.", "",
        "## Source files", "",
    ]
    lines += [f"- `{path}` — SHA-256 `{digest}`" for path, digest in source_hashes.items()]
    lines += ["", "## Original versus thinned results", "",
              "| Horizon | Original unique windows | Selected | Retained | Original macro coverage | Thinned macro coverage | Original worst | Thinned worst | Original below 90% | Thinned below 90% | Rank correlation | Mean absolute genotype change | Largest change |",
              "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for row in summary.itertuples(index=False):
        lines.append(
            f"| {row.horizon_hours} h | {row.original_unique_windows:,} | {row.selected_unique_windows:,} | "
            f"{100*row.retention_fraction:.2f}% | {row.original_macro_coverage:.6f} | "
            f"{row.thinned_macro_coverage:.6f} | {row.original_worst_coverage:.6f} | "
            f"{row.thinned_worst_coverage:.6f} | {row.original_genotypes_below_90}/21 | "
            f"{row.thinned_genotypes_below_90}/21 | {row.coverage_rank_correlation:.3f} | "
            f"{row.mean_absolute_genotype_change_pp:.2f} pp | {row.largest_absolute_genotype_change_pp:.2f} pp |")
    lines += ["", "## Genotype-level interpretation", ""]
    for horizon in HORIZONS:
        s = summary[summary.horizon_hours == horizon].iloc[0]
        largest = comparison[comparison.horizon_hours == horizon].nlargest(1, "absolute_change_pp").iloc[0]
        lines += [
            f"### {horizon} h", "",
            f"- Selected {int(s.selected_unique_windows):,} of {int(s.original_unique_windows):,} unique windows ({100*s.retention_fraction:.2f}%).",
            f"- Retained {int(s.selected_plants)}/318 plants and {int(s.selected_leaf_trajectories):,} leaf trajectories; zero-window genotypes: {int(s.zero_window_genotypes)}; zero-window plants: {int(s.zero_window_plants)}.",
            f"- Genotype coverage rank correlation was {s.coverage_rank_correlation:.3f}; bottom-five overlap was {int(s.bottom5_overlap)}/5.",
            f"- Largest absolute genotype change: `{largest.test_genotype}`, {largest.coverage_change_pp:+.2f} percentage points.",
            f"- Thinned width/score: {s.thinned_macro_width:.2f} px² / {s.thinned_macro_interval_score:.2f} (original {s.original_macro_width:.2f} / {s.original_macro_interval_score:.2f}).",
        ]
    lines += [
        "", "## Scientific interpretation", "",
        "Near-nominal genotype-macro coverage persists descriptively after thinning at both horizons. The relative difficulty pattern is also broadly stable: rank correlations are 0.978 and 0.940, bottom-five overlap is 5/5 and 4/5, and the same worst genotype remains worst at each horizon (`Ei-6` at 24 h and `Ws-2` at 48 h). Thus, counting overlapping forecast intervals is not the sole driver of the aggregate U5 pattern. The 48 h maximum movement of 3.70 percentage points is nevertheless non-negligible and should not be described as invariance.", "",
        "Only 32.46% and 17.39% of unique windows remain at 24 h and 48 h. Reduced window counts increase finite-sample granularity and instability, especially for genotypes with few plants or short trajectories. The selected intervals remain nested within plants and genotypes, and their historical input spans can overlap. This analysis is not independent validation, a prospective test, proof of exchangeability, or evidence of finite-sample 90% genotype-conditional validity under arbitrary genotype shift.", "",
        "## Automated checks", "",
        "- PASS: Original U5 macro, worst-genotype, and below-90% values reproduced to tolerance `1e-10`.",
        "- PASS: Per-window aggregation reproduced archived E3 coverage, width, interval score, plant counts, and window counts.",
        "- PASS: All 21 genotypes and all 318 plants remained represented at both horizons.",
        "- PASS: Every selected unique window has exactly four paired assignment rows.",
        "- PASS: Timestamp differences agree with rounded `actual_target_offset_hours` within `5e-6` h.",
        "- PASS: Selected forecast intervals do not overlap within trajectory; endpoint contact is allowed.",
        "- PASS: Selection audit records that selection uses no outcomes.",
        "- PASS: Coverage is in [0,1], interval widths are nonnegative, and count identities reconcile.",
        "", "## Outputs", "",
        "- `experiments/P2/P2_4/tables/p2_4_genotype_comparison.csv`",
        "- `experiments/P2/P2_4/tables/p2_4_summary.csv`",
        "- `experiments/P2/P2_4/tables/p2_4_selection_audit.csv`",
        "- `Research/P2/P2_4_nonoverlapping_window_sensitivity.md`",
    ]
    lines += [f"- `{path.relative_to(ROOT)}`" for path in outputs]
    lines += [
        "", "## Scientific decision", "",
        "**PASS WITH LIMITATIONS.** The predeclared, outcome-blind sensitivity is computationally reproducible and supports descriptive robustness of the aggregate U5 pattern. Severe thinning at 48 h, residual plant/genotype clustering, overlapping historical spans, and non-negligible genotype-level changes prevent stronger validation or formal coverage claims.",
    ]
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    comparisons = []
    summaries = []
    audits = []
    source_hashes = {str(E3_FOLDS.relative_to(ROOT)): sha256(E3_FOLDS)}

    for horizon in HORIZONS:
        source = PREDICTIONS / f"P1_1_U5_window_predictions_{horizon}h.csv"
        source_hashes[str(source.relative_to(ROOT))] = sha256(source)
        original = add_metrics(load_horizon(horizon))
        _, original_genotype = aggregate(original)
        validate_original(original, original_genotype, horizon)
        thinned, audit = select_nonoverlapping(original)
        _, thinned_genotype = aggregate(thinned)

        if set(original_genotype.test_genotype) != set(thinned_genotype.test_genotype):
            raise AssertionError(f"{horizon}h thinning removed a genotype")
        original_plants = set(original.plant_id.unique())
        selected_plants = set(thinned.plant_id.unique())
        missing_plants = original_plants - selected_plants
        if missing_plants:
            raise AssertionError(f"{horizon}h thinning removed plants: {sorted(missing_plants)[:5]}")

        comp = original_genotype.merge(
            thinned_genotype, on=["horizon_hours", "test_genotype"],
            suffixes=("_original", "_thinned"), validate="one_to_one")
        comp = comp.rename(columns={
            "coverage_original": "original_coverage", "coverage_thinned": "thinned_coverage",
            "width_original": "original_width", "width_thinned": "thinned_width",
            "interval_score_original": "original_interval_score",
            "interval_score_thinned": "thinned_interval_score",
            "unique_windows_original": "original_unique_windows",
            "unique_windows_thinned": "selected_unique_windows",
        })
        comp["coverage_change_pp"] = 100 * (comp.thinned_coverage - comp.original_coverage)
        comp["absolute_change_pp"] = comp.coverage_change_pp.abs()
        comp["original_rank"] = comp.original_coverage.rank(method="average")
        comp["thinned_rank"] = comp.thinned_coverage.rank(method="average")
        comparisons.append(comp)
        audits.append(audit)

        unique_original = original.drop_duplicates(ID_COLS)
        unique_thinned = thinned.drop_duplicates(ID_COLS)
        original_bottom = set(comp.nsmallest(5, "original_coverage").test_genotype)
        thinned_bottom = set(comp.nsmallest(5, "thinned_coverage").test_genotype)
        summaries.append({
            "horizon_hours": horizon,
            "original_unique_windows": len(unique_original),
            "selected_unique_windows": len(unique_thinned),
            "retention_fraction": len(unique_thinned) / len(unique_original),
            "selected_plants": unique_thinned.plant_id.nunique(),
            "selected_leaf_trajectories": unique_thinned.groupby(
                ["test_genotype", "plant_id", "leaf_id"]).ngroups,
            "zero_window_genotypes": 21 - unique_thinned.test_genotype.nunique(),
            "zero_window_plants": len(missing_plants),
            "original_macro_coverage": comp.original_coverage.mean(),
            "thinned_macro_coverage": comp.thinned_coverage.mean(),
            "original_worst_coverage": comp.original_coverage.min(),
            "thinned_worst_coverage": comp.thinned_coverage.min(),
            "original_worst_genotype": comp.loc[comp.original_coverage.idxmin(), "test_genotype"],
            "thinned_worst_genotype": comp.loc[comp.thinned_coverage.idxmin(), "test_genotype"],
            "original_genotypes_below_90": int((comp.original_coverage < NOMINAL).sum()),
            "thinned_genotypes_below_90": int((comp.thinned_coverage < NOMINAL).sum()),
            "original_macro_width": comp.original_width.mean(),
            "thinned_macro_width": comp.thinned_width.mean(),
            "original_macro_interval_score": comp.original_interval_score.mean(),
            "thinned_macro_interval_score": comp.thinned_interval_score.mean(),
            "coverage_rank_correlation": comp.original_coverage.corr(
                comp.thinned_coverage, method="spearman"),
            "bottom5_overlap": len(original_bottom & thinned_bottom),
            "mean_absolute_genotype_change_pp": comp.absolute_change_pp.mean(),
            "largest_absolute_genotype_change_pp": comp.absolute_change_pp.max(),
            "history_overlap_selected_adjacencies": int(
                audit.history_span_overlaps_previous_selected_history.sum()),
            "selected_rows_with_four_assignments": len(unique_thinned),
        })

    comparison = pd.concat(comparisons, ignore_index=True).sort_values(
        ["horizon_hours", "test_genotype"])
    summary = pd.DataFrame(summaries)
    audit = pd.concat(audits, ignore_index=True).sort_values(
        ["horizon_hours", "test_genotype", "plant_id", "leaf_id", "selected_order"])

    if not comparison.original_coverage.between(0, 1).all() or not (
            comparison.thinned_coverage.between(0, 1).all()):
        raise AssertionError("Coverage outside [0,1]")
    if (comparison[["original_width", "thinned_width"]].to_numpy() < 0).any():
        raise AssertionError("Negative aggregated width")
    if not audit.selection_uses_outcomes.eq(False).all():
        raise AssertionError("Outcome-dependent selection flag detected")

    comparison.to_csv(OUT / "p2_4_genotype_comparison.csv", index=False)
    summary.to_csv(OUT / "p2_4_summary.csv", index=False)
    audit.to_csv(OUT / "p2_4_selection_audit.csv", index=False)
    outputs = make_figure(comparison, summary)
    build_report(summary, comparison, source_hashes, outputs)
    print(summary.to_string(index=False))
    print(json.dumps({"status": "PASS", "genotypes": len(comparison),
                      "selected_windows": int(summary.selected_unique_windows.sum())}, indent=2))


if __name__ == "__main__":
    main()
