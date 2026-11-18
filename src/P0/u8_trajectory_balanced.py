#!/usr/bin/env python3
"""P0.4c: Paired U1 vs U8 offline leaf-trajectory calibration sensitivity.

Run from repository root:
    uv run --active python src/P0/run_u8_trajectory_balanced.py

Fits the ORIGINAL E3 B3 model once per LOGO assignment. Only calibration
weights differ. Neither variant has a finite-sample guarantee under arbitrary
unseen-genotype shift. Outputs go to experiments/P0/P0_4c_U8/.
"""
from __future__ import annotations

import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src' / 'E3'))
import run_normalized_conformal as e3  # noqa: E402

OUT = ROOT / 'experiments' / 'P0' / 'P0_4c_U8'
TABLES = OUT / 'tables'
FIGURES = OUT / 'figures'
REPORTS = OUT / 'reports'
METHODS = ('U1_balanced', 'U8_trajectory_balanced')
METRICS = ('coverage', 'width', 'score', 'mae')


def save_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    target = TABLES / name
    pd.DataFrame(rows).to_csv(target, index=False)


def leaf_ids_aligned(data: dict) -> dict[int, np.ndarray]:
    """Preserve per-horizon source file ordering and verify rows against E3."""
    result = {h: {'g': [], 'p': [], 'y': [], 'leaf': [], 'window_id': []} for h in e3.HORIZONS}
    ids_seen = set()
    with e3.MANIFEST.open(newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            h = int(row['horizon_hours'])
            if h not in result:
                raise ValueError(f'Unexpected horizon {h}')
            wid = row['window_id']
            if not wid or wid in ids_seen:
                raise ValueError(f'Missing/duplicate source window_id: {wid}')
            ids_seen.add(wid)
            d = result[h]
            d['g'].append(row['genotype'])
            d['p'].append(row['plant_id'])
            d['y'].append(float(row['target_area_pixels']))
            d['leaf'].append(row['leaf_id'])
            d['window_id'].append(wid)
    aligned = {}
    for h, s in result.items():
        d = data[h]
        if not (len(s['leaf']) == len(d['y']) and
                np.array_equal(np.asarray(s['g'], dtype=str), d['g'].astype(str)) and
                np.array_equal(np.asarray(s['p'], dtype=str), d['p'].astype(str)) and
                np.allclose(np.asarray(s['y'], dtype=float), d['y'], rtol=1e-8, atol=1e-6)):
            raise ValueError(f'{h}h: positional source/E3 alignment FAILED; do not attach leaf IDs')
        if not all(str(x).strip() for x in s['leaf']):
            raise ValueError(f'{h}h: blank leaf ID')
        aligned[h] = np.asarray(s['leaf'], dtype=str)
        n_tracks = len(set(zip(s['g'], s['p'], s['leaf'])))
        print(f'{h}h linkage PASS: {len(s["leaf"]):,} windows, {n_tracks:,} tracks, '
              f'{len(set(s["window_id"])):,} unique window IDs', flush=True)
    return aligned


def trajectory_weights(g: np.ndarray, p: np.ndarray, leaf: np.ndarray) -> np.ndarray:
    """1/(#genotypes * #plants|g * #leaves|g,p * #windows|g,p,l)."""
    frame = pd.DataFrame({'g': g.astype(str), 'p': p.astype(str), 'l': leaf.astype(str)})
    if frame.empty:
        raise ValueError('Empty calibration set')
    if (frame.groupby('p')['g'].nunique() > 1).any():
        raise ValueError('Plant ID shared across genotypes')
    group_g = frame.groupby('g', sort=False)
    group_p = frame.groupby(['g', 'p'], sort=False)
    group_l = frame.groupby(['g', 'p', 'l'], sort=False)
    n_g = frame['g'].nunique()
    n_p = group_g['p'].transform('nunique').to_numpy(dtype=float)
    n_l = group_p['l'].transform('nunique').to_numpy(dtype=float)
    n_w = group_l['l'].transform('size').to_numpy(dtype=float)
    w = 1.0 / (n_g * n_p * n_l * n_w)
    if not np.isclose(w.sum(), 1.0, rtol=0, atol=1e-10):
        raise ValueError(f'U8 weights do not sum to 1: {w.sum()}')
    by_g = frame.assign(weight=w).groupby('g')['weight'].sum().to_numpy()
    if not np.allclose(by_g, 1. / n_g, rtol=0, atol=1e-10):
        raise ValueError('Unequal genotype mass')
    return w


def evaluate(d, leaf, outer, inner, horizon):
    fit_g = set(inner['train_genotypes'])
    cal_g = set(inner['validation_genotypes'])
    test_g = outer['test_genotype']
    if len(fit_g) != 15 or len(cal_g) != 5 or fit_g & cal_g or test_g in fit_g | cal_g:
        raise ValueError('Invalid LOGO split')
    fit = np.isin(d['g'], list(fit_g))
    cal = np.isin(d['g'], list(cal_g))
    test = d['g'] == test_g
    if not (set(d['g'][fit]) == fit_g and set(d['g'][cal]) == cal_g
            and set(d['g'][test]) == {test_g}):
        raise ValueError('Incomplete genotype membership')
    # EXACT E3 model fitting: do not change fit weights or hyperparameters.
    model = e3.HistGradientBoostingRegressor(**e3.PARAMS)
    fit_weights = e3.balanced_weights(d['g'][fit], d['p'][fit])
    model.fit(d['X'][fit], d['y'][fit], sample_weight=fit_weights * fit_weights.size)
    cal_pred = np.maximum(model.predict(d['X'][cal]), 0.0)
    residuals = np.abs(d['y'][cal] - cal_pred)
    w1 = e3.balanced_weights(d['g'][cal], d['p'][cal])
    w8 = trajectory_weights(d['g'][cal], d['p'][cal], leaf[cal])
    q1 = e3.balanced_quantile(residuals, w1)
    q8 = e3.balanced_quantile(residuals, w8)
    pred = np.maximum(model.predict(d['X'][test]), 0.0)
    target = d['y'][test]
    plants = d['p'][test]
    base = {'horizon_hours': horizon, 'outer_fold': outer['outer_fold'],
            'inner_fold': inner['inner_fold'], 'test_genotype': test_g,
            'calibration_genotypes': len(cal_g),
            'calibration_plants': len(np.unique(d['p'][cal])),
            'calibration_tracks': len(set(zip(d['g'][cal], d['p'][cal], leaf[cal]))),
            'calibration_windows': int(cal.sum()),
            'u1_u8_weight_L1': float(np.abs(w1 - w8).sum())}
    rows = []
    for method, q in ((METHODS[0], q1), (METHODS[1], q8)):
        lower = np.maximum(0., pred - q)
        upper = pred + q
        metrics = e3.plant_aggregated_metrics(target, pred, lower, upper, plants)
        rows.append({**base, 'method': method, 'threshold': float(q), **metrics})
    return rows


def aggregate(rows):
    folds = pd.DataFrame(rows)
    checks = folds.groupby(['horizon_hours', 'test_genotype', 'method']).size()
    if not (checks.eq(4).all() and len(checks) == len(e3.HORIZONS) * 21 * 2):
        raise ValueError('Missing or duplicated inner assignments')
    gen = (folds.groupby(['horizon_hours', 'test_genotype', 'method'], as_index=False)
           [list(METRICS) + ['threshold']].mean())
    summaries = []
    pairs = []
    for h in e3.HORIZONS:
        d = gen[gen.horizon_hours == h]
        for method in METHODS:
            x = d[d.method == method]
            if len(x) != 21:
                raise ValueError(f'{h} {method}: expected 21 genotypes')
            summaries.append({'horizon_hours': h, 'method': method,
                              'macro_coverage': x.coverage.mean(),
                              'macro_width': x.width.mean(),
                              'macro_interval_score': x.score.mean(),
                              'macro_mae': x.mae.mean(),
                              'worst_genotype_coverage': x.coverage.min(),
                              'genotypes_below_90': int((x.coverage < e3.NOMINAL).sum())})
        pivot = d.pivot(index='test_genotype', columns='method', values=list(METRICS)+['threshold'])
        for genotype in pivot.index:
            rec = {'horizon_hours': h, 'test_genotype': genotype}
            for metric in list(METRICS) + ['threshold']:
                v1 = float(pivot.loc[genotype, (metric, METHODS[0])])
                v8 = float(pivot.loc[genotype, (metric, METHODS[1])])
                rec[f'u1_{metric}'] = v1
                rec[f'u8_{metric}'] = v8
                rec[f'delta_{metric}'] = v8-v1
            pairs.append(rec)
    return gen, pd.DataFrame(summaries), pd.DataFrame(pairs)


def verify_saved_u1(folds):
    original = e3.FOLD_OUT
    if not original.exists():
        print(f'Original E3.1 fold CSV missing: {original}; exact baseline check skipped')
        return 'not checked (original file absent)'
    ref = pd.read_csv(original)
    sub = folds[folds.method == METHODS[0]]
    ref = ref[ref.method == METHODS[0]]
    keys = ['horizon_hours', 'outer_fold', 'inner_fold', 'test_genotype', 'method']
    merge = sub.merge(ref, on=keys, suffixes=('_new', '_ref'), how='outer', indicator=True,
                      validate='one_to_one')
    if not merge['_merge'].eq('both').all() or len(merge) != 168:
        raise ValueError('U1 baseline fold set does not match original E3')
    comparisons = ('threshold', 'coverage', 'width', 'score', 'mae')
    for metric in comparisons:
        new = merge[f'{metric}_new'].to_numpy(dtype=float)
        old = merge[f'{metric}_ref'].to_numpy(dtype=float)
        if not np.allclose(new, old, rtol=1e-8, atol=1e-6, equal_nan=True):
            mismatch = np.abs(new-old)
            raise ValueError(f'U1 replication FAILED for {metric}: largest difference={mismatch.max()}')
    print('Original E3 U1 replication PASS: all 168 paired folds, thresholds, metrics', flush=True)
    return 'PASS: 168 assignments matched threshold, coverage, width, score, MAE'


def plots(gen, pairs):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('matplotlib unavailable; skipping plots')
        return
    for h in e3.HORIZONS:
        d = pairs[pairs.horizon_hours == h].sort_values('delta_coverage')
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(d.test_genotype, 100*d.delta_coverage)
        ax.axhline(0, color='black', lw=.8)
        ax.set_title(f'U8 minus U1 genotype coverage — {h}h')
        ax.set_xlabel('Held-out genotype')
        ax.set_ylabel('Coverage difference (percentage points)')
        ax.tick_params(axis='x', rotation=65)
        fig.tight_layout()
        fig.savefig(FIGURES / f'U8_coverage_change_{h}h.png', dpi=170)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(d.test_genotype, d.delta_score)
        ax.axhline(0, color='black', lw=.8)
        ax.set_title(f'U8 minus U1 genotype interval score — {h}h')
        ax.set_xlabel('Held-out genotype')
        ax.set_ylabel('Interval score difference (negative is better)')
        ax.tick_params(axis='x', rotation=65)
        fig.tight_layout()
        fig.savefig(FIGURES / f'U8_interval_score_change_{h}h.png', dpi=170)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharex=True, sharey=True)
    for ax, h in zip(axes, e3.HORIZONS):
        d = gen[gen.horizon_hours == h]
        pivot = d.pivot(index='test_genotype', columns='method', values='coverage')
        ax.scatter(pivot[METHODS[0]], pivot[METHODS[1]])
        ax.plot([0, 1], [0, 1], color='black', linestyle='--', linewidth=.8)
        ax.axhline(.9, linestyle=':', color='gray')
        ax.axvline(.9, linestyle=':', color='gray')
        ax.set_title(f'{h}h')
        ax.set_xlabel('U1 coverage')
    axes[0].set_ylabel('U8 coverage')
    fig.suptitle('Paired genotype-level coverage')
    fig.tight_layout()
    fig.savefig(FIGURES / 'U8_U1_coverage_scatter.png', dpi=170)
    plt.close(fig)


def report(summary, pairs, verification):
    lines = [
        '# P0.4c — U8 trajectory-balanced calibration sensitivity', '',
        '## Scope',
        'U1 and U8 use the same B3 fitted model, point predictions, absolute residuals,',
        'and original E3 empirical weighted 90th percentile. Only calibration weights differ.',
        'Train weights remain the original genotype/plant-balanced weights.',
        'The outer test genotype contributes no fitting or calibration labels.', '',
        '## Original E3 baseline verification', verification, '',
        '## Results',
        '| Horizon | Method | Macro coverage | Width (px) | Interval score | Worst coverage | # below 90% |',
        '|---|---|---:|---:|---:|---:|---:|',
    ]
    for _, r in summary.iterrows():
        lines.append(f'| {int(r.horizon_hours)}h | {r.method} | {r.macro_coverage:.4f} | '
                     f'{r.macro_width:.2f} | {r.macro_interval_score:.2f} | '
                     f'{r.worst_genotype_coverage:.4f} | {int(r.genotypes_below_90)}/21 |')
    lines += ['', '## Paired changes (U8 minus U1)',
              '| Horizon | Mean coverage change (pp) | Mean width change (px) | Mean score change | Improved genotypes by coverage |',
              '|---|---:|---:|---:|---:|']
    for h in e3.HORIZONS:
        d = pairs[pairs.horizon_hours == h]
        lines.append(f'| {h}h | {100*d.delta_coverage.mean():+.3f} | '
                     f'{d.delta_width.mean():+.2f} | {d.delta_score.mean():+.2f} | '
                     f'{int((d.delta_coverage > 0).sum())}/21 |')
    lines += ['', '## Interpretation and limitations',
              '- This is a paired empirical calibration-weight sensitivity, **not** a new conformal algorithm.',
              '- Equal leaf-track weighting does not remove time dependence between overlapping windows.',
              '- Arbitrary unseen-genotype shift violates standard exchangeability conditions; no genotype-level guarantee is claimed.',
              '- The same held-out genotypes are reused across four inner assignments; 84 folds per horizon are not independent test populations.',
              '- Leaf IDs were linked by verified source/E3 positional correspondence; E3 does not directly carry window_id.',
              '- Calibration weights are recomputed within each set of five calibration genotypes.',
              '- Interpretation is exploratory because earlier results from these same genotypes have been inspected.',
              '', '## Files',
              '- `tables/U8_fold_metrics.csv`',
              '- `tables/U8_genotype_metrics.csv`',
              '- `tables/U8_paired_genotype_differences.csv`',
              '- `tables/U8_summary.csv`',
              '- `figures/U8_coverage_change_24h.png` and `48h.png`',
              '- `figures/U8_interval_score_change_24h.png` and `48h.png`',
              '- `figures/U8_U1_coverage_scatter.png`', '']
    (REPORTS / 'P0_4c_U8_report.md').write_text('\n'.join(lines), encoding='utf-8')


def main():
    for folder in (TABLES, FIGURES, REPORTS):
        folder.mkdir(parents=True, exist_ok=True)
    if not e3.FOLDS.exists():
        raise FileNotFoundError(e3.FOLDS)
    with e3.FOLDS.open(encoding='utf-8') as f:
        folds = json.load(f)['folds']
    if len(folds) != 21:
        raise ValueError('Expected 21 outer folds')
    print('P0.4c U8: loading unchanged E3 features...', flush=True)
    data = e3.load_data()
    leaf = leaf_ids_aligned(data)
    all_rows = []
    for h in e3.HORIZONS:
        print(f'\n{h}h | {len(data[h]["y"]):,} windows', flush=True)
        for index, outer in enumerate(folds, 1):
            for inner in outer['inner_folds']:
                all_rows.extend(evaluate(data[h], leaf[h], outer, inner, h))
            save_csv('U8_fold_metrics_checkpoint.csv', all_rows)
            print(f'{h}h | completed {index:02d}/21 | {outer["test_genotype"]}', flush=True)
    folds_df = pd.DataFrame(all_rows)
    # Critical: refuse to interpret U8 if exact original U1 cannot be replicated.
    verification = verify_saved_u1(folds_df)
    gen, summary, pairs = aggregate(all_rows)
    save_csv('U8_fold_metrics.csv', all_rows)
    gen.to_csv(TABLES / 'U8_genotype_metrics.csv', index=False)
    summary.to_csv(TABLES / 'U8_summary.csv', index=False)
    pairs.to_csv(TABLES / 'U8_paired_genotype_differences.csv', index=False)
    plots(gen, pairs)
    report(summary, pairs, verification)
    print('\n' + '='*70 + '\nP0.4c U8 SUMMARY\n' + '='*70)
    print(summary.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    for h in e3.HORIZONS:
        d = pairs[pairs.horizon_hours == h]
        print(f'\n{h}h | U8 - U1: coverage {100*d.delta_coverage.mean():+.3f} pp; '
              f'width {d.delta_width.mean():+.2f} px; '
              f'score {d.delta_score.mean():+.2f}')
        print('Largest absolute coverage changes:')
        print(d.reindex(d.delta_coverage.abs().sort_values(ascending=False).index)
              [['test_genotype', 'delta_coverage', 'delta_width', 'delta_score']]
              .head(5).to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    print(f'\nResults: {OUT}')


if __name__ == '__main__':
    main()
