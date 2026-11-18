#!/usr/bin/env python3
"""P3.1 plant-replication sensitivity for archived held-out U5 window predictions.

Standard project run (no path arguments required):
    .venv/bin/python src/P3/P3_1_analysis.py

By default, the script reads the archived 24 h and 48 h U5 window-prediction
tables under experiments/P1/P1_1/tables/ and writes to experiments/P3/P3_1/.
Custom inputs may be supplied with --input FILE [FILE ...].
Optional: --genotype-col ..., --plant-col ..., --horizon-col ..., --assignment-col ...,
          --target-col y --lower-col lower --upper-col upper
OR       --coverage-col covered

Does NOT retrain, recalibrate, or fabricate research data.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = (
    ROOT / 'experiments/P1/P1_1/tables/P1_1_U5_window_predictions_24h.csv',
    ROOT / 'experiments/P1/P1_1/tables/P1_1_U5_window_predictions_48h.csv',
)
DEFAULT_OUTDIR = ROOT / 'experiments/P3/P3_1'

EXPECTED = {24: {'macro':0.8997, 'below':8, 'min':0.8222, 'worst':'Ei-6'},
            48: {'macro':0.9004, 'below':6, 'min':0.7639, 'worst':'Ws-2'}}


def repository_path(value):
    """Resolve command-line paths relative to the repository root."""
    path = Path(value).expanduser()
    return path if path.is_absolute() else ROOT / path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        '--input', nargs='+', type=repository_path, default=list(DEFAULT_INPUTS),
        metavar='CSV',
        help='Optional override: one or more U5 prediction CSVs. Omit for the repository archives.',
    )
    ap.add_argument(
        '--outdir', type=repository_path, default=DEFAULT_OUTDIR,
        help='Optional override. Omit to write to experiments/P3/P3_1.',
    )
    for key, default in [('genotype','test_genotype'),('plant','plant_id'),
                         ('horizon','horizon_hours'),('assignment','inner_fold'),
                         ('target','target_area_pixels'),('lower','lower_pixels'),
                         ('upper','upper_pixels'),('coverage','covered')]:
        ap.add_argument('--'+key+'-col', default=default)
    ap.add_argument('--repetitions', type=int, default=10000)
    ap.add_argument('--seed', type=int, default=20261116)
    args = ap.parse_args()
    out = args.outdir
    out.mkdir(parents=True,exist_ok=True)
    missing_inputs = [path for path in args.input if not path.is_file()]
    if missing_inputs:
        raise FileNotFoundError(
            'Missing input file(s): ' + ', '.join(str(path) for path in missing_inputs)
        )
    raw = pd.concat(
        (pd.read_csv(path) for path in args.input),
        ignore_index=True,
        sort=False,
    )
    cols = {key: getattr(args,key+'_col') for key in ('genotype','plant','horizon','assignment','target','lower','upper','coverage')}
    req = [cols[x] for x in ('genotype','plant','horizon','assignment')]
    missing = [c for c in req if c not in raw]
    if missing: raise ValueError(f'Missing required columns: {missing}; found {list(raw)}')
    canonical = {
        'genotype': 'genotype',
        'plant': 'plant',
        'horizon': 'horizon',
        'assignment': 'assignment',
        'target': 'y',
        'lower': 'lower',
        'upper': 'upper',
        'coverage': 'covered',
    }
    rename = {
        source: canonical[key]
        for key, source in cols.items()
        if source in raw.columns
    }
    df = raw.rename(columns=rename).copy()
    if df[['genotype','plant','horizon','assignment']].isna().any().any():
        raise ValueError('Null genotype/plant/horizon/assignment identifiers; inspect input')
    df['genotype'] = df['genotype'].astype(str)
    df['plant'] = df['plant'].astype(str)
    df['horizon'] = pd.to_numeric(df['horizon'],errors='raise').astype(int)
    if set(df.horizon.unique()) != {24,48}:
        raise ValueError(f'Expected horizons 24,48; got {sorted(df.horizon.unique())}')
    if 'covered' in df.columns:
        vals = pd.to_numeric(df['covered'], errors='coerce')
        if vals.isna().any() or not vals.isin([0,1]).all(): raise ValueError('covered must be 0 or 1 with no nulls')
        df['covered'] = vals.astype(int)
    else:
        if not {'y','lower','upper'} <= set(df.columns):
            raise ValueError('Need covered column OR y,lower,upper endpoint columns')
        for c in ('y','lower','upper'):
            df[c] = pd.to_numeric(df[c], errors='coerce')
            if df[c].isna().any(): raise ValueError(f'NaN/invalid {c}')
        if (df.lower > df.upper).any(): raise ValueError('Detected lower > upper')
        df['covered'] = ((df.y >= df.lower) & (df.y <= df.upper)).astype(int)
    # Require exactly four assignments, same labels, for each genotype/horizon/plant.
    distinct = set(df.assignment.astype(str).unique())
    if len(distinct) != 4: raise ValueError(f'Expected 4 assignment labels; found {distinct}')
    groups = df.groupby(['horizon','genotype','plant'],observed=True,sort=True)
    bad = groups['assignment'].nunique().loc[lambda x: x != 4]
    if len(bad): raise ValueError(f'Missing assignment(s) for {len(bad)} plant/horizon cases: {bad.head()}')
    # Check counts of prediction windows match across assignments.
    sizes = df.groupby(['horizon','genotype','plant','assignment'],observed=True).size()
    spread = sizes.groupby(level=[0,1,2]).agg(['min','max'])
    if (spread['min'] != spread['max']).any():
        raise ValueError('Assignment-specific window counts differ within plant; cannot reproduce equal-window estimand')
    if 'window_id' in df:
        wid = df.groupby(['horizon','genotype','plant','assignment'],observed=True)['window_id'].apply(lambda s: frozenset(s.astype(str)))
        if wid.groupby(level=[0,1,2]).nunique().gt(1).any():
            raise ValueError('window_id sets differ across assignments within a plant/horizon')
        if df.duplicated(['horizon','genotype','plant','assignment','window_id']).any():
            raise ValueError('Duplicate window_id within assignment')
    # Mean within plant × assignment, then mean over 4 assignments.
    pa = df.groupby(['horizon','genotype','plant','assignment'],observed=True)['covered'].mean().rename('assignment_coverage').reset_index()
    plants = pa.groupby(['horizon','genotype','plant'],observed=True)['assignment_coverage'].mean().rename('plant_coverage').reset_index()
    gen = plants.groupby(['horizon','genotype'],observed=True).agg(coverage=('plant_coverage','mean'),plants=('plant','nunique'),plant_sd=('plant_coverage','std')).reset_index()
    gen['below_90'] = gen.coverage < 0.9
    gen['replication_group'] = np.where(gen.plants < 8,'<8 plants','>=8 plants')
    gen.to_csv(out/'P3_1_genotype_summary.csv',index=False)
    rng = np.random.default_rng(args.seed)
    rows=[]
    for h in (24,48):
        for g, frame in plants[plants.horizon==h].groupby('genotype',sort=True):
            vals = frame.plant_coverage.to_numpy(dtype=float)
            if len(vals) < 20: continue
            full = float(vals.mean())
            for n in (4,5,6,7):
                draws = np.empty(args.repetitions)
                for j in range(args.repetitions):
                    draws[j] = vals[rng.choice(len(vals), n, replace=False)].mean()
                rows.append(dict(horizon=h,genotype=g,full_plants=len(vals),sample_plants=n,
                                 full_coverage=full,subsample_mean=float(draws.mean()),
                                 subsample_sd=float(draws.std(ddof=1)),p025=float(np.quantile(draws,.025)),
                                 p975=float(np.quantile(draws,.975)),
                                 frac_below_90=float(np.mean(draws < .9)),
                                 frac_at_or_below_original_worst=float(np.mean(draws <= EXPECTED[h]['min']))))
    samples = pd.DataFrame(rows)
    samples.to_csv(out/'P3_1_subsampling_summary.csv',index=False)
    fig, axes = plt.subplots(1,2,figsize=(12,5.5),sharey=True)
    for ax,h in zip(axes,(24,48)):
        sub = samples[samples.horizon==h]
        for g, part in sub.groupby('genotype',sort=True):
            part = part.sort_values('sample_plants')
            ax.plot(part.sample_plants,part.subsample_mean,marker='o',linewidth=1.25,label=g)
            ax.fill_between(part.sample_plants,part.p025,part.p975,alpha=.08)
            ax.axhline(part.full_coverage.iloc[0],alpha=.08,linestyle=':')
        ax.axhline(.9,color='black',linestyle='--',linewidth=1,label='90% target')
        ax.axhline(EXPECTED[h]['min'],color='gray',linestyle=':',linewidth=1.5,label='Observed minimum')
        ax.set_title(f'U5: {h}-hour horizon')
        ax.set_xlabel('Number of sampled plants')
        ax.set_xticks([4,5,6,7])
        ax.set_ylim(.65,1.005)
    axes[0].set_ylabel('Equal-plant empirical coverage')
    handles, labels=axes[-1].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,-.09),ncol=5,fontsize=8)
    fig.suptitle('P3.1: retrospective sensitivity to plant replication')
    fig.tight_layout(rect=[0,.03,1,.95]);fig.savefig(out/'P3_1_figure.png',dpi=200,bbox_inches='tight');plt.close(fig)
    input_lines = [f'- `{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}`'
                   for path in args.input]
    report=['# P3.1 — numerical results','','Inputs:',*input_lines,'',
            f'Monte Carlo draws per cell: {args.repetitions}; seed: {args.seed}','',
            '## Reproduction checks (must review before interpreting sensitivity)','']
    nplant = plants.groupby(['genotype','plant']).ngroups
    report.append(f'- Unique genotype–plant pairs: {nplant}; expected manuscript count 318.')
    report.append(f'- Genotypes: {gen.genotype.nunique()}; expected 21.')
    passed=True
    if nplant != 318 or gen.genotype.nunique()!=21: passed=False
    for h in (24,48):
        s=gen[gen.horizon==h]; ex=EXPECTED[h]
        macro=float(s.coverage.mean()); below=int(s.below_90.sum())
        worst=s.loc[s.coverage.idxmin()]
        ok=(len(s)==21 and abs(macro-ex['macro'])<=.00015 and below==ex['below'] and
            abs(float(worst.coverage)-ex['min'])<=.00015 and str(worst.genotype)==ex['worst'])
        passed=passed and ok
        report.append(f'- {h} h: macro {macro:.5%} (expected {ex["macro"]:.2%}); below 90% {below} (expected {ex["below"]}); minimum {worst.genotype} {worst.coverage:.5%} (expected {ex["worst"]} {ex["min"]:.2%}); {"PASS" if ok else "DISCREPANCY"}.')
    report += ['',f'**Overall reproduction check: {"PASS" if passed else "DISCREPANCY — investigate BEFORE interpreting"}.**','',
               '## Descriptive replication groups','']
    for h in (24,48):
        for group in ('<8 plants','>=8 plants'):
            part=gen[(gen.horizon==h)&(gen.replication_group==group)]
            report.append(f'- {h} h, {group}: {len(part)} genotypes; unweighted subgroup mean {part.coverage.mean():.5%}; below 90% {part.below_90.sum()}.')
    report+=['','## Sensitivity output','',
             f'- High-replication genotypes included per horizon: {samples[samples.horizon==24].genotype.nunique()}.',
             '- Full draw summaries and tail fractions: `P3_1_subsampling_summary.csv`.','',
             '## Interpretation limits','',
             'This is a retrospective conditional sensitivity to resampling existing observed plants, not proof that extreme undercoverage is sampling noise or a genotype effect. Four assignments and overlapping windows are not independent plant replicates. No refitting, calibration, or manuscript edits were performed.']
    (out/'P3_1_results.md').write_text('\n'.join(report)+'\n',encoding='utf-8')
    print('\n'.join(report))
    if not passed: raise SystemExit('Reproduction discrepancy — review exported predictions and input assumptions; files written for diagnosis.')

if __name__=='__main__': main()
