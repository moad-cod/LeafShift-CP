#!/usr/bin/env python3
"""Figure 5: U5 assignment-specific held-out-genotype coverage heatmaps.

Run from the root of leaf-growth-generalization:
    python figures/rebuilt/F5_assignment_reliability/plot_figure5.py

Reads archived experiment outputs; performs no model fitting and invents no data.
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors, ticker
import numpy as np
import pandas as pd

ROOT = Path.cwd()
FOLD_CSV = ROOT / 'experiments/E3/tables/E3_2_fold_metrics.csv'
CI_CSV = ROOT / 'experiments/P1/P1_1/statistics/tables/p1_genotype_coverage_ci.csv'
OUT = ROOT / 'figures/rebuilt/F5_assignment_reliability'
OUT.mkdir(parents=True, exist_ok=True)
METHOD = 'U5_CQR_balanced'
EXPECTED_MACRO = {24: 0.8997119769358646, 48: 0.9004006113622590}
EXPECTED_WORST = {24: ('Ei-6', 0.8221703622232337),
                  48: ('Ws-2', 0.7639306084000144)}
EXPECTED_BELOW = {24: 8, 48: 6}


def require(condition, message):
    if not condition:
        raise ValueError(message)


for p in (FOLD_CSV, CI_CSV):
    require(p.exists(), f'Missing archived input: {p}')

fold = pd.read_csv(FOLD_CSV)
ci = pd.read_csv(CI_CSV)
needed = {'outer_fold', 'inner_fold', 'horizon_hours', 'test_genotype',
          'method', 'plants', 'coverage'}
require(needed.issubset(fold.columns), f'Fold table missing {needed - set(fold.columns)}')
require({'horizon_hours', 'test_genotype', 'method', 'coverage', 'plants'}.issubset(ci.columns),
        'Bootstrap summary lacks required columns')
fold = fold.loc[fold['method'].eq(METHOD)].copy()
ci = ci.loc[ci['method'].eq(METHOD)].copy()
keys = ['horizon_hours', 'test_genotype']
require(len(fold) == 168, f'Expected 168 U5 fold rows; found {len(fold)}')
require(set(fold.horizon_hours) == {24, 48}, 'Expected exactly two forecast horizons')
require(fold['coverage'].notna().all() and fold['coverage'].between(0, 1).all(),
        'Coverage must be finite within [0,1]')
require(not fold.duplicated(keys + ['inner_fold']).any(),
        'Duplicate genotype/horizon/assignment rows')
require(len(ci) == 42 and not ci.duplicated(keys).any(),
        'Expected exactly one bootstrap summary per genotype and horizon')
require(ci['coverage'].between(0, 1).all(), 'Bootstrap summary has invalid coverage')

# Assignment IDs are kept as recorded; the four assignments are NOT independent
# biological replicates. Confirm all 21 genotypes have identical four IDs in both horizons.
assignment_ids = sorted(fold.inner_fold.unique().tolist())
require(len(assignment_ids) == 4, f'Expected exactly 4 assignments, got {assignment_ids}')
expected_ids = set(assignment_ids)
for (horizon, genotype), group in fold.groupby(keys):
    require(set(group.inner_fold) == expected_ids and len(group) == 4,
            f'Assignment mismatch: {horizon}h, {genotype}')
    require(group.outer_fold.nunique() == 1,
            f'Genotype maps to multiple outer folds: {genotype} {horizon}h')
    require(group.plants.nunique() == 1,
            f'Plant count varies between assignments: {genotype} {horizon}h')

stats = (fold.groupby(keys, as_index=False)
         .agg(coverage=('coverage', 'mean'), plants=('plants', 'first')))
merged = stats.merge(ci[keys + ['coverage', 'plants']], on=keys,
                     suffixes=('_fold', '_bootstrap'), validate='one_to_one')
require(len(merged) == 42, 'Bootstrap and fold tables have mismatched genotype keys')
max_diff = (merged.coverage_fold - merged.coverage_bootstrap).abs().max()
require(max_diff < 1e-9, f'Fold/plant-bootstrap estimates disagree: {max_diff}')
require((merged.plants_fold == merged.plants_bootstrap).all(),
        'Plant-count mismatch in source tables')

names_24 = set(fold.loc[fold.horizon_hours == 24, 'test_genotype'])
names_48 = set(fold.loc[fold.horizon_hours == 48, 'test_genotype'])
require(names_24 == names_48 and len(names_24) == 21,
        'Expected identical sets of 21 genotypes in both horizons')
for horizon in (24, 48):
    s = stats.loc[stats.horizon_hours == horizon]
    macro = s.coverage.mean()
    worst = s.loc[s.coverage.idxmin()]
    below = int((s.coverage < .9).sum())
    expected_name, expected_value = EXPECTED_WORST[horizon]
    require(abs(macro - EXPECTED_MACRO[horizon]) < 1e-9,
            f'Macro mismatch for {horizon}h: {macro}')
    require(worst.test_genotype == expected_name
            and abs(worst.coverage - expected_value) < 1e-9,
            f'Worst-genotype mismatch for {horizon}h')
    require(below == EXPECTED_BELOW[horizon],
            f'Below-90 count mismatch for {horizon}h')
    print(f'{horizon}h: macro={macro:.8f}; worst={worst.test_genotype} '
          f'({worst.coverage:.8f}); below90={below}/21 — PASS')
print(f'Fold versus bootstrap max absolute difference: {max_diff:.3e} — PASS')

# Export exact assignment-level input, preserving the original inner_fold labels.
export = fold[['horizon_hours', 'test_genotype', 'outer_fold', 'inner_fold',
               'plants', 'coverage']].sort_values(
                   ['horizon_hours', 'test_genotype', 'inner_fold'])
source_path = OUT / 'F5_U5_assignment_coverage_source.csv'
export.to_csv(source_path, index=False, float_format='%.16g')

# Shared genotype order: ascending 48-h assignment-mean coverage, as in Figure 4.
order = (stats[stats.horizon_hours == 48]
         .sort_values(['coverage', 'test_genotype'])['test_genotype'].tolist())
nplants = stats[stats.horizon_hours == 48].set_index('test_genotype')['plants']
row_labels = [f'{g}   (n={int(nplants[g])})' for g in order]

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 8.3,
    'pdf.fonttype': 42, 'svg.fonttype': 'none',
    'savefig.facecolor': 'white',
})

# Diverging mapping centered at the nominal 90% threshold. A fixed 0–100% scale
# is used so color meanings are identical between 24h and 48h panels.
cmap = plt.get_cmap('RdYlBu')
norm = colors.TwoSlopeNorm(vmin=0.0, vcenter=0.90, vmax=1.0)
fig, axes = plt.subplots(1, 2, figsize=(10.2, 8.1), sharey=True)
fig.subplots_adjust(left=0.165, right=0.87, top=0.885, bottom=0.125, wspace=0.22)

for ax, horizon, panel in zip(axes, (24, 48), ('a', 'b')):
    sub = fold.loc[fold.horizon_hours == horizon]
    matrix = sub.pivot(index='test_genotype', columns='inner_fold', values='coverage')
    matrix = matrix.loc[order, assignment_ids]
    require(matrix.shape == (21, 4) and matrix.notna().all().all(),
            f'Incomplete heatmap for {horizon}h')
    arr = matrix.to_numpy(dtype=float)
    im = ax.imshow(arr, cmap=cmap, norm=norm, aspect='auto', interpolation='nearest')

    # Subtle cell boundaries; no pseudoreplicate significance indicators.
    ax.set_xticks(np.arange(-0.5, 4, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, 21, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=1.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.set_xticks(range(4), [f'A{i+1}' for i in range(4)])
    ax.set_yticks(range(21), row_labels)
    ax.tick_params(axis='both', length=0)
    ax.set_xlabel('Fitting/calibration assignment')
    if horizon == 24:
        ax.tick_params(axis='y', labelleft=True)
        ax.set_ylabel('Held-out genotype  (plant count)', labelpad=8)
    else:
        ax.tick_params(axis='y', labelleft=False)
    ax.set_title(f'({panel}) {horizon}-hour forecast', loc='left', weight='bold', pad=22)
    macro = stats.loc[stats.horizon_hours == horizon, 'coverage'].mean()
    below = EXPECTED_BELOW[horizon]
    ax.text(0.0, 1.012, f'Macro {macro:.2%}  ·  {below}/21 genotypes below 90%',
            transform=ax.transAxes, ha='left', va='bottom', fontsize=8.6)

    # One decimal percentage point per cell; contrasting foreground for legibility.
    for r in range(21):
        for c in range(4):
            value = arr[r, c]
            rgba = cmap(norm(value))
            luminance = .2126*rgba[0] + .7152*rgba[1] + .0722*rgba[2]
            label_color = 'black' if luminance > .52 else 'white'
            ax.text(c, r, f'{100*value:.1f}', ha='center', va='center',
                    fontsize=7.5, color=label_color)

# A common colorbar makes threshold-centered colors directly comparable.
cbar_ax = fig.add_axes([0.898, 0.19, 0.018, 0.61])
cbar = fig.colorbar(im, cax=cbar_ax, ticks=[0, .5, .7, .8, .9, .95, 1.0])
cbar.ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1, decimals=0))
cbar.set_label('Assignment-specific empirical coverage', labelpad=9)
cbar.ax.axhline(.90, color='#333333', linewidth=1.3)

# Scientific caveat belongs in caption, not as a crowded plot footnote.
stem = OUT / 'F5_U5_assignment_reliability'
fig.savefig(stem.with_suffix('.pdf'), bbox_inches='tight',
            metadata={'Title': 'U5 assignment-level empirical coverage by held-out genotype',
                      'Creator': 'leaf-growth-generalization figure pipeline'})
fig.savefig(stem.with_suffix('.svg'), bbox_inches='tight',
            metadata={'Title': 'U5 assignment-level empirical coverage by held-out genotype'})
fig.savefig(stem.with_suffix('.png'), dpi=500, bbox_inches='tight')
plt.close(fig)
print('VALIDATION PASS: 21 genotypes × 4 paired assignments × 2 horizons')
print(f'Saved source: {source_path}')
print(f'Saved figure: {stem}.{{pdf,svg,png}}')
