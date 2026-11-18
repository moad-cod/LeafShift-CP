#!/usr/bin/env python3
"""Figure 2: held-out genotype point-forecasting MAE, using current E1 tables.

Run from anywhere: python figures/rebuilt/F2_point_forecasting/plot_figure2.py
Dependencies: numpy pandas matplotlib.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MaxNLocator

ROOT = Path(__file__).resolve().parents[3]
E1 = ROOT / 'experiments/E1/tables'
OUT = ROOT / 'figures/rebuilt/F2_point_forecasting'
OUT.mkdir(parents=True, exist_ok=True)

BASELINE = E1 / 'E1_2_genotype_metrics.csv'
BASELINE_SUM = E1 / 'E1_2_summary.csv'
BOOST = E1 / 'E1_5_genotype_metrics.csv'
BOOST_SUM = E1 / 'E1_5_summary.csv'
RIDGE_SUM = E1 / 'E1_4_summary.csv'


def ensure(condition, msg):
    if not condition:
        raise ValueError(msg)


def close(actual, expected, what, tol=0.00015):
    ensure(abs(float(actual) - float(expected)) <= tol,
           f'{what}: computed {actual:.10f}, archived {expected:.10f}')

base = pd.read_csv(BASELINE)
boost = pd.read_csv(BOOST)
base_sum = pd.read_csv(BASELINE_SUM)
boost_sum = pd.read_csv(BOOST_SUM)
ridge_sum = pd.read_csv(RIDGE_SUM)

# E1.2 has additional QC subsets: use ALL, not a subgroup.
base = base[(base['subset'] == 'ALL') & (base['model'] == 'B0_persistence')].copy()
ensure(len(base) == 42, 'Expected exactly 42 persistence genotype-horizon rows.')
ensure(len(boost) == 42, 'Expected exactly 42 current E1.5 genotype-horizon rows.')

keys_b = ['horizon_hours', 'genotype']
keys_h = ['horizon_hours', 'test_genotype']
ensure(not base.duplicated(keys_b).any(), 'Duplicate E1.2 persistence genotype/horizon.')
ensure(not boost.duplicated(keys_h).any(), 'Duplicate E1.5 genotype/horizon.')

b = base[['horizon_hours', 'genotype', 'plants', 'windows',
          'plant_mean_mae_pixels']].rename(columns={
              'plant_mean_mae_pixels': 'persistence_mae',
              'plants': 'baseline_plants', 'windows': 'baseline_windows'})
h = boost[['horizon_hours', 'test_genotype', 'plants', 'windows',
           'B2_mae', 'B3_mae', 'B1_mae']].rename(columns={
               'test_genotype': 'genotype', 'plants': 'boost_plants',
               'windows': 'boost_windows', 'B2_mae': 'ridge_mae',
               'B3_mae': 'hgb_mae', 'B1_mae': 'linear_mae'})
data = h.merge(b, on=keys_b, how='outer', validate='one_to_one', indicator=True)
ensure((data['_merge'] == 'both').all() and len(data) == 42,
       'E1.2 and E1.5 genotype keys do not agree.')
data = data.drop(columns='_merge')
ensure(set(data.horizon_hours) == {24, 48}, 'Unexpected horizon values.')
ensure(data.groupby('horizon_hours').genotype.nunique().to_dict() == {24:21, 48:21},
       'Expected 21 genotypes at each horizon.')
ensure(set(data.loc[data.horizon_hours == 24, 'genotype']) ==
       set(data.loc[data.horizon_hours == 48, 'genotype']),
       'Genotype identity differs between horizons.')
ensure((data.baseline_plants == data.boost_plants).all(), 'Plant count mismatch.')
ensure((data.baseline_windows == data.boost_windows).all(), 'Window count mismatch.')
ensure(data[['persistence_mae', 'ridge_mae', 'hgb_mae']].notna().all().all(),
       'Missing model MAE.')
ensure((data[['persistence_mae', 'ridge_mae', 'hgb_mae']] >= 0).all().all(),
       'Negative MAE value.')

EXPECTED = {
  24: (710.06528, 327.758182, 309.921529),
  48: (1355.38917, 586.799846, 518.512105),
}
for horizon in (24, 48):
    sub = data[data.horizon_hours == horizon]
    sm_b = base_sum[(base_sum.subset == 'ALL') &
                    (base_sum.horizon_hours == horizon) &
                    (base_sum.model == 'B0_persistence')]
    sm_h = boost_sum[boost_sum.horizon_hours == horizon]
    sm_r = ridge_sum[ridge_sum.horizon_hours == horizon]
    ensure(len(sm_b) == len(sm_h) == len(sm_r) == 1,
           f'Missing or duplicated E1 summaries for {horizon}h.')
    sm_b, sm_h, sm_r = sm_b.iloc[0], sm_h.iloc[0], sm_r.iloc[0]
    p, r, h = [sub[c].mean() for c in ('persistence_mae','ridge_mae','hgb_mae')]
    close(p, sm_b['macro_genotype_plant_mae_pixels'], f'{horizon}h persistence summary')
    close(r, sm_h['B2_macro_mae'], f'{horizon}h Ridge E1.5 summary')
    close(h, sm_h['B3_macro_mae'], f'{horizon}h HGB E1.5 summary')
    close(r, sm_r['B2_ridge_macro_mae_pixels'], f'{horizon}h Ridge E1.4 summary')
    for got, ref, name in zip((p,r,h), EXPECTED[horizon], ('persistence','ridge','HGB')):
        close(got, ref, f'{horizon}h {name} independent expected')
    hgb_wins = int((sub.hgb_mae < sub.ridge_mae).sum())
    ridge_wins = int((sub.hgb_mae > sub.ridge_mae).sum())
    ensure(hgb_wins == int(sm_h.B3_wins_over_B2) and
           ridge_wins == int(sm_h.B2_wins_over_B3),
           f'{horizon}h HGB-vs-Ridge win count mismatch.')
    print(f'{horizon}h: persistence={p:.5f}, Ridge={r:.5f}, HGB={h:.5f} px²; '
          f'HGB wins {hgb_wins}/21 against Ridge — PASS')

# Same row order on both panels, most difficult HGB genotypes at 48h first.
order = (data[data.horizon_hours == 48]
         .sort_values(['hgb_mae','genotype'], ascending=[False,True])
         .genotype.tolist())
idx = {g:i for i,g in enumerate(order)}
data['plot_order'] = data.genotype.map(idx)
data = data.sort_values(['horizon_hours','plot_order'])
source = OUT / 'F2_point_forecasting_source.csv'
data.to_csv(source, index=False, float_format='%.16g')

plt.rcParams.update({
    'font.family':'DejaVu Sans', 'font.size':9.0, 'axes.labelsize':10,
    'axes.titlesize':12, 'xtick.labelsize':8, 'ytick.labelsize':8.5,
    'axes.linewidth':0.65, 'pdf.fonttype':42, 'svg.fonttype':'none',
    'savefig.facecolor':'white',
})
C = {'persistence_mae':'#7F858D','ridge_mae':'#0072B2','hgb_mae':'#D55E00'}
MARK = {'persistence_mae':'o','ridge_mae':'s','hgb_mae':'D'}
LABEL = {'persistence_mae':'Persistence','ridge_mae':'Ridge','hgb_mae':'HGB'}
# horizontal offset per model so overlapping points do not hide one another
YOFFSET = {'persistence_mae':-0.19,'ridge_mae':0.0,'hgb_mae':0.19}
fig, axes = plt.subplots(1,2,figsize=(10.4,7.35),sharey=True,
                         gridspec_kw={'wspace':0.15})
fig.subplots_adjust(left=0.145, right=0.985, top=0.865, bottom=0.14)
for ax,horizon,panel in zip(axes,(24,48),('a','b')):
    sub = data[data.horizon_hours == horizon].set_index('genotype').loc[order].reset_index()
    yy = np.arange(21)
    for i in range(21):
        ax.axhspan(i-.5,i+.5,color=('#F7F8FA' if i%2==0 else '#FFFFFF'),zorder=0)
    for col in ('persistence_mae','ridge_mae','hgb_mae'):
        vals = sub[col].to_numpy()
        ax.scatter(vals, yy+YOFFSET[col], s=27 if col!='hgb_mae' else 33,
                   marker=MARK[col], c=C[col], linewidths=0,
                   zorder=3, label=LABEL[col])
    # Explicit header positions reserve a clear gap between title and subtitle.
    ax.text(0.0, 1.095, f'({panel}) {horizon}-hour forecast',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=12, fontweight='bold', clip_on=False)
    p,r,h = (sub[c].mean() for c in ('persistence_mae','ridge_mae','hgb_mae'))
    ax.text(0.0,1.035,f'Macro MAE: Persistence {p:,.0f}  ·  Ridge {r:,.0f}  ·  HGB {h:,.0f} px²',
            transform=ax.transAxes,ha='left',va='bottom',fontsize=7.9,color='#52565B')
    ax.set_yticks(yy)
    if horizon==24:
        ax.set_yticklabels([f'{g}  (n={int(n)})' for g,n in zip(order,sub.boost_plants)])
        ax.tick_params(axis='y',length=0,labelleft=True)
    else:
        ax.tick_params(axis='y',left=False,labelleft=False)
    ax.set_ylim(20.65,-0.65)
    ax.set_xlim(left=0)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5,integer=True))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x:,.0f}'))
    ax.grid(axis='x',color='#E1E5EA',linewidth=.7)
    ax.set_axisbelow(True)
    ax.set_xlabel('Genotype-specific mean absolute error (px²)',labelpad=9)
    for spine in ('left','right','top'):
        ax.spines[spine].set_visible(False)
    ax.spines['bottom'].set_color('#A5ABB2')
    ax.tick_params(axis='x',length=3,color='#777')

handles = [Line2D([],[],linestyle='none',marker=MARK[k],color=C[k],
                  markersize=6.5,label=LABEL[k])
           for k in ('persistence_mae','ridge_mae','hgb_mae')]
fig.legend(handles=handles,loc='lower center',ncol=3,frameon=False,
           bbox_to_anchor=(.54,.025),columnspacing=2.8,handletextpad=.55,fontsize=9.5)
stem = OUT / 'F2_point_forecasting'
meta={'Title':'Genotype-specific point forecasting error under leave-one-genotype-out evaluation',
      'Creator':'E1 archived results plotting pipeline'}
fig.savefig(stem.with_suffix('.pdf'),bbox_inches='tight',metadata=meta)
fig.savefig(stem.with_suffix('.svg'),bbox_inches='tight',metadata=meta)
fig.savefig(stem.with_suffix('.png'),dpi=500,bbox_inches='tight')
plt.close(fig)
print('VALIDATION PASS: 21 held-out genotypes x 2 horizons x 3 point models.')
print('Saved source:',source)
print('Saved figure:',str(stem)+'.{pdf,svg,png}')
