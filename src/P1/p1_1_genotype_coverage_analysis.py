#!/usr/bin/env python3
"""P1.1: locked, secondary exploratory U5 genotype coverage analysis.

Reads the verified E3.2 U5 per-plant/per-assignment output. Never retrains.
Run from any directory: uv run --active python src/P1/p1_1_genotype_coverage_analysis.py

IMPORTANT: Random-regrouping reference is descriptive, not a valid permutation
hypothesis test, because plants from different LOGO folds are not exchangeable.
"""
from __future__ import annotations

from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'experiments/P1/P1_1/tables/P1_1_U5_plant_assignment_coverage.csv'
REFERENCE = ROOT / 'experiments/P1/P1_1/tables/P1_1_U5_verified_fold_metrics.csv'
OUT = ROOT / 'experiments/P1/P1_1/statistics'
TABLES = OUT / 'tables'
FIGURES = OUT / 'figures'
REPORTS = OUT / 'reports'
SEED = 20261109
REPS = 10_000
NOMINAL = 0.90
METHOD = 'U5_CQR_balanced'
HORIZONS = (24, 48)


def save_table(frame: pd.DataFrame, filename: str) -> None:
    frame.to_csv(TABLES / filename, index=False, float_format='%.12g')


def load_and_check() -> pd.DataFrame:
    for directory in (TABLES, FIGURES, REPORTS):
        directory.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(INPUT)
    required = {'plant_id', 'windows', 'coverage', 'horizon_hours',
                'outer_fold', 'inner_fold', 'test_genotype', 'method'}
    if not required.issubset(df.columns):
        raise ValueError(f'Missing fields: {sorted(required - set(df.columns))}')
    if not (df['method'] == METHOD).all():
        raise ValueError('Unexpected method in plant export')
    keys = ['horizon_hours', 'test_genotype', 'inner_fold', 'plant_id']
    if df.duplicated(keys).any():
        raise ValueError('Duplicated plant/assignment keys')
    if df[keys].isna().any().any() or df['coverage'].isna().any():
        raise ValueError('Missing identifying fields or coverage')
    if not df['coverage'].between(0, 1).all():
        raise ValueError('Coverage outside [0,1]')
    if not (df['windows'] > 0).all():
        raise ValueError('Nonpositive window count')

    reference = pd.read_csv(REFERENCE)
    reference = reference[reference['method'] == METHOD]
    computed = df.groupby(['horizon_hours', 'outer_fold', 'inner_fold', 'test_genotype'],
                          as_index=False).agg(coverage_check=('coverage', 'mean'),
                                              plants_check=('plant_id', 'nunique'),
                                              windows_check=('windows', 'sum'))
    merged = computed.merge(reference,
                            on=['horizon_hours', 'outer_fold', 'inner_fold', 'test_genotype'],
                            validate='one_to_one', indicator=True)
    if len(merged) != 168 or not merged['_merge'].eq('both').all():
        raise ValueError(f'Expected 168 matching fold metrics, found {len(merged)}')
    if not np.allclose(merged['coverage_check'], merged['coverage'], atol=1e-9, rtol=0):
        diff = np.max(np.abs(merged['coverage_check'] - merged['coverage']))
        raise ValueError(f'Plant mean does not reproduce U5 fold coverage: max difference={diff}')
    if not np.array_equal(merged['plants_check'].to_numpy(), merged['plants'].to_numpy()):
        raise ValueError('Plant counts disagree with verified E3.2')
    if not np.array_equal(merged['windows_check'].to_numpy(), merged['windows'].to_numpy()):
        raise ValueError('Window counts disagree with verified E3.2')
    if set(df['horizon_hours'].unique()) != set(HORIZONS):
        raise ValueError('Unexpected horizons')
    # All four assignments must contain the identical test plants and their window counts.
    a = df.groupby(['horizon_hours', 'test_genotype', 'plant_id']).agg(
        assignments=('inner_fold', 'nunique'),
        rows=('inner_fold', 'size'),
        min_windows=('windows', 'min'), max_windows=('windows', 'max'))
    if not ((a['assignments'] == 4) & (a['rows'] == 4) &
            (a['min_windows'] == a['max_windows'])).all():
        raise ValueError('Test plant coverage is not paired over four assignments')
    sizes = df.groupby(['horizon_hours', 'test_genotype'])['plant_id'].nunique()
    if not all(len(sizes.loc[h]) == 21 and sizes.loc[h].sum() == 318 for h in HORIZONS):
        raise ValueError('Expected 21 test genotypes and 318 plants per horizon')
    print('PASS: 168 fold coverages, plant counts, and window counts agree with verified E3.2')
    print('PASS: Every test plant has exactly four paired assignment outcomes')
    return df


def prepare_plant_data(df: pd.DataFrame) -> pd.DataFrame:
    # Main estimand: window mean within plant, then equal-plant genotype mean.
    # Average the repeated assignments *within* each plant before resampling.
    plant = df.groupby(['horizon_hours', 'test_genotype', 'plant_id'],
                       as_index=False).agg(coverage=('coverage', 'mean'),
                                           assignments=('inner_fold', 'nunique'),
                                           windows=('windows', 'first'))
    if not (plant['assignments'] == 4).all():
        raise ValueError('Assignment pairing broken')
    save_table(plant, 'p1_plant_assignment_averaged_coverage.csv')
    return plant


def clustered_bootstrap(plant: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for h in HORIZONS:
        for g, d in plant[plant['horizon_hours'] == h].groupby('test_genotype', sort=True):
            y = d['coverage'].to_numpy(dtype=float)
            n = len(y)
            sampled = rng.choice(y, size=(REPS, n), replace=True).mean(axis=1)
            lo, hi = np.quantile(sampled, [0.025, 0.975])
            status = ('unstable_fewer_than_8_plants' if n < 8 else
                      'below_90_bootstrap_upper' if hi < NOMINAL else
                      'above_90_bootstrap_lower' if lo > NOMINAL else
                      'inconclusive')
            rows.append(dict(horizon_hours=h, method=METHOD, test_genotype=g,
                             plants=n, coverage=y.mean(), ci_lower=lo,
                             ci_upper=hi, status=status))
    out = pd.DataFrame(rows)
    save_table(out, 'p1_genotype_coverage_ci.csv')
    return out


def stats_of_groups(groups: list[np.ndarray]) -> tuple[float, int, float]:
    means = np.asarray([x.mean() for x in groups], dtype=float)
    return float(means.std(ddof=1)), int((means < NOMINAL).sum()), float(means.min())


def regrouping_reference(plant: pd.DataFrame):
    rows = []
    summary = []
    for h in HORIZONS:
        subset = plant[plant['horizon_hours'] == h]
        groups = [g['coverage'].to_numpy() for _, g in subset.groupby('test_genotype', sort=True)]
        sizes = np.array([len(g) for g in groups], dtype=int)
        obs_s1, obs_s2, obs_s3 = stats_of_groups(groups)
        all_plant = np.concatenate(groups)
        cutpoints = np.cumsum(sizes)[:-1]
        # Separate deterministic random streams per horizon.
        rng = np.random.default_rng(SEED + h)
        null_s1 = np.empty(REPS)
        null_s2 = np.empty(REPS, dtype=int)
        null_s3 = np.empty(REPS)
        for i in range(REPS):
            pseudo = np.split(rng.permutation(all_plant), cutpoints)
            null_s1[i], null_s2[i], null_s3[i] = stats_of_groups(pseudo)
        rows.extend(dict(horizon_hours=h, replicate=i + 1, s1_sd=null_s1[i],
                         s2_below90=int(null_s2[i]), s3_min=null_s3[i]) for i in range(REPS))
        # +1 convention for finite Monte Carlo resolution. Descriptive only.
        tail_fraction = (1 + int((null_s1 >= obs_s1).sum())) / (REPS + 1)
        summary.append(dict(horizon_hours=h, observed_s1_sd=obs_s1,
                            observed_s2_below90=obs_s2, observed_s3_min=obs_s3,
                            null_s1_median=np.median(null_s1),
                            null_s1_p95=np.quantile(null_s1, .95),
                            null_s2_p95=np.quantile(null_s2, .95),
                            null_s3_p05=np.quantile(null_s3, .05),
                            s1_descriptive_tail_fraction=tail_fraction,
                            interpretation='heuristic_not_confirmatory_permutation_p_value'))
    null = pd.DataFrame(rows)
    summary = pd.DataFrame(summary)
    save_table(null, 'p1_permutation_null.csv')
    save_table(summary, 'p1_permutation_summary.csv')
    return null, summary


def assignment_variability(df: pd.DataFrame):
    folds = df.groupby(['horizon_hours','test_genotype','inner_fold'], as_index=False)['coverage'].mean()
    rows = []
    summaries = []
    for h in HORIZONS:
        sub = folds[folds['horizon_hours'] == h]
        per_g = sub.groupby('test_genotype')['coverage'].agg(['mean', 'std']).reset_index()
        between = float(per_g['mean'].std(ddof=1))
        within_rms = float(np.sqrt(np.mean(per_g['std'].to_numpy()**2)))
        ratio = within_rms / between if between > 0 else np.nan
        summaries.append(dict(horizon_hours=h, method=METHOD,
                              between_genotype_sd=between,
                              within_genotype_assignment_sd_rms=within_rms,
                              within_to_between_ratio=ratio,
                              flag_ratio_ge_half=bool(ratio >= 0.5),
                              note='fit_and_calibration_assignments_both_change'))
        for _, r in per_g.iterrows():
            rows.append(dict(horizon_hours=h, method=METHOD,
                             test_genotype=r['test_genotype'],
                             mean_coverage=float(r['mean']),
                             assignment_sd=float(r['std']),
                             between_genotype_sd=between,
                             within_to_between_ratio=ratio))
    out, summary = pd.DataFrame(rows), pd.DataFrame(summaries)
    save_table(out, 'p1_assignment_variance.csv')
    save_table(summary, 'p1_assignment_variance_summary.csv')
    return out, summary


def make_figures(ci: pd.DataFrame, null: pd.DataFrame, permutation: pd.DataFrame) -> None:
    for h in HORIZONS:
        sub = ci[ci.horizon_hours == h].sort_values('coverage')
        x = np.arange(len(sub))
        y = sub.coverage.to_numpy()
        err = np.vstack((y-sub.ci_lower.to_numpy(), sub.ci_upper.to_numpy()-y))
        fig, ax = plt.subplots(figsize=(12, 5.5))
        ax.errorbar(x, y, yerr=err, fmt='o', capsize=2, markersize=4)
        ax.axhline(NOMINAL, linestyle='--', linewidth=1)
        ax.set_xticks(x, sub.test_genotype, rotation=65, ha='right')
        ax.set_ylabel('U5 plant-averaged coverage')
        ax.set_title(f'{h}h: genotype coverage and plant-clustered bootstrap intervals')
        ax.set_ylim(0.5, 1.03)
        ax.grid(alpha=.2, axis='y')
        fig.tight_layout()
        fig.savefig(FIGURES / f'p1_genotype_coverage_ci_{h}h.png', dpi=180)
        plt.close(fig)

        values = null.loc[null.horizon_hours == h, 's1_sd'].to_numpy()
        obs = float(permutation.loc[permutation.horizon_hours == h, 'observed_s1_sd'].iloc[0])
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(values, bins=45, edgecolor='white')
        ax.axvline(obs, linestyle='--', linewidth=2, label='Observed SD')
        ax.set_xlabel('SD of pseudo-genotype mean coverage')
        ax.set_ylabel('Random regroupings')
        ax.set_title(f'{h}h: descriptive plant-regrouping reference (not a valid permutation test)')
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / f'p1_regrouping_s1_{h}h.png', dpi=180)
        plt.close(fig)


def report(ci, null_summary, assignment_summary) -> None:
    lines = ['# P1.1 — U5 Genotype-Level Coverage: Secondary Exploratory Analysis', '',
             f'- Seed: `{SEED}`; bootstrap / random-regrouping replicates: `{REPS:,}`.',
             '- Original estimand retained: window coverage averaged within plant; equal plant weights within genotype; four assignments averaged within plant.',
             '- The 21 test genotypes have been inspected during prior experiment development. This analysis was locked afterward and is **not confirmatory**.',
             '- Plant-clustered bootstrap intervals reflect within-genotype plant sampling, conditional on fitted model/calibration choices and observed genotypes. They do not provide prospective genotype-level guarantees.',
             '- Plant regrouping mixes outcomes from different LOGO model/calibration folds. Its reference-tail fraction is **not a valid confirmatory p-value**.',
             '- Assignment variability changes both fitted and calibration genotypes; it is not an isolated calibration-noise component.', '',
             '## Results', '']
    for h in HORIZONS:
        d = ci[ci.horizon_hours == h]
        p = null_summary[null_summary.horizon_hours == h].iloc[0]
        a = assignment_summary[assignment_summary.horizon_hours == h].iloc[0]
        lines.extend([
            f'### {h}h', '',
            f'- Mean genotype coverage: {d.coverage.mean():.4f}; worst: {d.coverage.min():.4f}; genotypes under 90%: {int((d.coverage < NOMINAL).sum())}/21.',
            f'- Plant-bootstrap statuses: {d.status.value_counts().to_dict()}.',
            f'- Observed genotype SD S1: {p.observed_s1_sd:.5f}; random regrouping 95th percentile: {p.null_s1_p95:.5f}; heuristic reference-tail fraction: {p.s1_descriptive_tail_fraction:.5f}.',
            f'- Observed S2 count: {p.observed_s2_below90}; regrouping 95th percentile: {p.null_s2_p95}.',
            f'- Observed S3 minimum: {p.observed_s3_min:.4f}; regrouping 5th percentile: {p.null_s3_p05:.4f}.',
            f'- Between-genotype SD: {a.between_genotype_sd:.5f}; within-genotype assignment RMS SD: {a.within_genotype_assignment_sd_rms:.5f}; ratio: {a.within_to_between_ratio:.3f}.', '',
        ])
    lines.extend(['## Files', '',
                  '- `tables/p1_genotype_coverage_ci.csv`',
                  '- `tables/p1_permutation_null.csv` and `tables/p1_permutation_summary.csv`',
                  '- `tables/p1_assignment_variance.csv` and `tables/p1_assignment_variance_summary.csv`',
                  '- `figures/p1_genotype_coverage_ci_{24,48}h.png`',
                  '- `figures/p1_regrouping_s1_{24,48}h.png`', '',
                  '## Interpretation boundary', '',
                  'Do not claim that these resampling diagnostics prove biological genotype effects, causal mechanisms, statistically valid subgroup coverage, or independent confirmation of prior selected findings.', ''])
    (REPORTS/'p1_1_genotype_coverage_analysis.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    df = load_and_check()
    plant = prepare_plant_data(df)
    ci = clustered_bootstrap(plant)
    null, p = regrouping_reference(plant)
    assignment, a = assignment_variability(df)
    make_figures(ci, null, p)
    report(ci, p, a)
    print('\nP1.1 U5 EXPLORATORY ANALYSIS COMPLETE')
    for h in HORIZONS:
        c = ci[ci.horizon_hours == h]
        ph = p[p.horizon_hours == h].iloc[0]
        ah = a[a.horizon_hours == h].iloc[0]
        print(f'{h}h | macro={c.coverage.mean():.5f} | worst={c.coverage.min():.5f} '
              f'| below90={(c.coverage < .9).sum()}/21 '
              f'| confident_under={(c.status == "below_90_bootstrap_upper").sum()} '
              f'| unstable={(c.status == "unstable_fewer_than_8_plants").sum()}')
        print(f'  observed S1={ph.observed_s1_sd:.5f}; null95={ph.null_s1_p95:.5f}; '
              f'heuristic tail={ph.s1_descriptive_tail_fraction:.5f}')
        print(f'  assignment RMS SD={ah.within_genotype_assignment_sd_rms:.5f}; '
              f'between SD={ah.between_genotype_sd:.5f}; ratio={ah.within_to_between_ratio:.3f}')
    print(f'Output: {OUT}')

if __name__ == '__main__':
    main()
