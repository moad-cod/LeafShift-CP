#!/usr/bin/env python3
"""Rebuild Figure 3 from frozen E2/E3/P0 summary tables. No fitting or recalibration."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter, MultipleLocator

ROOT = Path.cwd()
OUT = ROOT / 'figures/rebuilt/F3_method_comparison'
OUT.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'U0': ('experiments/E2/tables/E2_4_summary.csv', 'U0_pooled', 'macro_coverage', 'macro_width_pixels', 'macro_interval_score', 'genotypes_mean_coverage_below_90'),
    'U1': ('experiments/E2/tables/E2_4_summary.csv', 'U1_balanced', 'macro_coverage', 'macro_width_pixels', 'macro_interval_score', 'genotypes_mean_coverage_below_90'),
    'U2': ('experiments/E3/tables/E3_1_summary.csv', 'U2_normalized_pooled', 'macro_coverage', 'macro_width', 'macro_interval_score', 'genotypes_below_90'),
    'U3': ('experiments/E3/tables/E3_1_summary.csv', 'U3_normalized_balanced', 'macro_coverage', 'macro_width', 'macro_interval_score', 'genotypes_below_90'),
    'U4': ('experiments/E3/tables/E3_2_summary.csv', 'U4_CQR_pooled', 'macro_coverage', 'macro_width', 'macro_interval_score', 'genotypes_below_90'),
    'U5': ('experiments/E3/tables/E3_2_summary.csv', 'U5_CQR_balanced', 'macro_coverage', 'macro_width', 'macro_interval_score', 'genotypes_below_90'),
    'U6': ('experiments/E3/tables/E3_3_summary.csv', 'U6_Mondrian_pooled', 'macro_coverage', 'macro_width', 'macro_interval_score', 'genotypes_below_90'),
    'U7': ('experiments/E3/tables/E3_3_summary.csv', 'U7_Mondrian_balanced', 'macro_coverage', 'macro_width', 'macro_interval_score', 'genotypes_below_90'),
    'U8': ('experiments/P0/P0_4c_U8/tables/U8_summary.csv', 'U8_trajectory_balanced', 'macro_coverage', 'macro_width', 'macro_interval_score', 'genotypes_below_90'),
}
EXPECTED = {
    (24,'U0'):(.836015,1073.799670,2398.232949,19),
    (24,'U1'):(.899218,1394.937802,2287.068890,9),
    (24,'U2'):(.879876,1262.076170,2103.819798,17),
    (24,'U3'):(.900063,1379.671473,2092.640446,12),
    (24,'U4'):(.881455,1403.178490,2140.534431,16),
    (24,'U5'):(.899712,1479.058951,2139.587082,8),
    (24,'U6'):(.868387,1209.986109,2198.608909,19),
    (24,'U7'):(.903582,1389.721561,2179.534406,9),
    (24,'U8'):(.904159,1429.833091,2285.336465,9),
    (48,'U0'):(.837937,1817.200136,3694.733584,19),
    (48,'U1'):(.898680,2305.041386,3528.252020,9),
    (48,'U2'):(.883323,2252.585090,3428.181276,17),
    (48,'U3'):(.899627,2413.212078,3437.774821,11),
    (48,'U4'):(.868710,2164.331740,3395.171520,16),
    (48,'U5'):(.900401,2385.689153,3389.091671,6),
    (48,'U6'):(.865687,2028.597036,3506.301404,19),
    (48,'U7'):(.903799,2348.521897,3468.431669,7),
    (48,'U8'):(.912349,2464.007931,3530.146638,6),
}
FAMILIES = {'U0':'Absolute residual', 'U1':'Absolute residual',
            'U2':'Normalized residual', 'U3':'Normalized residual',
            'U4':'CQR', 'U5':'CQR', 'U6':'Size-Mondrian', 'U7':'Size-Mondrian',
            'U8':'Trajectory sensitivity'}
COLORS = {'Absolute residual':'#0072B2', 'Normalized residual':'#009E73',
          'CQR':'#D55E00', 'Size-Mondrian':'#88419D',
          'Trajectory sensitivity':'#8C6D31'}

# Formatting choices: leave enough room for labels around the near-90% cluster.
# All x positions and y positions are exact archived values; offsets only move labels.
OFFSETS = {
    24: {'U0':(6,9), 'U1':(-23,13), 'U2':(-4,12), 'U3':(-27,-19),
         'U4':(6,-13), 'U5':(-4,-17), 'U6':(-8,12), 'U7':(-2,23), 'U8':(17,11)},
    48: {'U0':(8,10), 'U1':(-25,-14), 'U2':(5,11), 'U3':(17,-18),
         'U4':(7,11), 'U5':(-25,14), 'U6':(-8,-16), 'U7':(-5,26), 'U8':(-23,10)},
}

frames = {}
records = []
for method, (path, source_method, ccol, wcol, scol, bcol) in SOURCES.items():
    if path not in frames:
        p = ROOT / path
        if not p.is_file():
            raise FileNotFoundError(f'Missing source CSV: {p}')
        frames[path] = pd.read_csv(p)
    frame = frames[path]
    for h in (24, 48):
        match = frame.loc[(frame['horizon_hours']==h) & (frame['method']==source_method)]
        if len(match) != 1:
            raise ValueError(f'Expected exactly one {h}h {source_method} row in {path}; found {len(match)}')
        r = match.iloc[0]
        coverage, width, score, below = float(r[ccol]), float(r[wcol]), float(r[scol]), int(r[bcol])
        expected = EXPECTED[(h, method)]
        # Expected rounded audit values are cross-checked to 6 decimals for coverage,
        # and to 5 decimals for width/score (do not substitute these values in plot).
        if not (abs(coverage-expected[0]) < 5.1e-7 and
                abs(width-expected[1]) < 5.1e-6 and
                abs(score-expected[2]) < 5.1e-6 and below == expected[3]):
            raise ValueError(f'Unexpected archived values at {(h,method)}: '
                             f'{(coverage,width,score,below)} != {expected}')
        if not (0 <= coverage <= 1 and width >= 0 and score >= 0 and 0 <= below <= 21):
            raise ValueError(f'Invalid metric values for {(h,method)}')
        records.append({'horizon_hours':h,'method':method,'procedure':source_method,
                        'family':FAMILIES[method], 'type': ('exploratory' if method=='U8' else
                        'pooled' if int(method[1:]) % 2 == 0 else 'balanced'),
                        'coverage':coverage,'width_px2':width,'interval_score':score,
                        'genotypes_below_90':below,'source_csv':path})

combined = pd.DataFrame(records).sort_values(['horizon_hours','method']).reset_index(drop=True)
assert len(combined)==18 and combined[['horizon_hours','method']].duplicated().sum()==0
# Check that the independently stored benchmark summary agrees on the shared four rows.
comparison_path = ROOT / 'experiments/E3/tables/E3_4_method_comparison.csv'
if not comparison_path.is_file():
    raise FileNotFoundError(comparison_path)
comparison = pd.read_csv(comparison_path)
for h in (24,48):
    for m in ('U1','U3','U5','U7'):
        row = combined.query('horizon_hours == @h and method == @m').iloc[0]
        match = comparison[(comparison.horizon_hours==h) & (comparison.method==row.procedure)]
        if len(match)!=1:
            raise ValueError(f'Missing cross-check E3.4 row for {m}/{h}h')
        rr=match.iloc[0]
        for a,b in [('coverage','coverage'),('width_px2','width'),('interval_score','score')]:
            if not np.isclose(row[a],rr[b],atol=1e-9,rtol=0):
                raise ValueError(f'E3.4 cross-check failed: {m}/{h}h {a}')
        if row.genotypes_below_90 != int(rr.genotypes_below_90):
            raise ValueError(f'E3.4 below90 mismatch: {m}/{h}h')

csv_path = OUT / 'F3_method_comparison_source.csv'
combined.to_csv(csv_path,index=False,float_format='%.15g')
print(f'VALIDATION PASS: 18 unique entries; U0–U8 at 24h and 48h; '
      f'E3.4 cross-check passed for U1/U3/U5/U7.')
print(combined[['horizon_hours','method','coverage','width_px2','interval_score','genotypes_below_90']]
      .to_string(index=False,float_format=lambda v: f'{v:.6f}'))
print(f'Saved source data: {csv_path}')

plt.rcParams.update({'font.family':'DejaVu Sans', 'font.size':9,
                     'axes.labelsize':10,'axes.titlesize':11,'xtick.labelsize':8.5,
                     'ytick.labelsize':8.5,'pdf.fonttype':42,'svg.fonttype':'none',
                     'savefig.facecolor':'white','axes.linewidth':0.75})
fig, axs = plt.subplots(1, 2, figsize=(11.8, 6.3), sharey=True)
fig.subplots_adjust(left=.08, right=.98, top=.88, bottom=.30, wspace=.16)
for ax,h,panel in zip(axs,(24,48),('a','b')):
    sub = combined[combined.horizon_hours==h]
    # Dotted reference is an empirical target, not a guarantee.
    ax.axhline(.9,color='#4F5660',lw=1.15,ls=(0,(4,3)),zorder=1)
    for pair in [('U0','U1'),('U2','U3'),('U4','U5'),('U6','U7')]:
        d = sub[sub.method.isin(pair)].set_index('method')
        ax.plot([d.loc[pair[0],'width_px2'],d.loc[pair[1],'width_px2']],
                [d.loc[pair[0],'coverage'],d.loc[pair[1],'coverage']],
                color=COLORS[d.loc[pair[0],'family']],alpha=.4,lw=1.2,zorder=2)
    for row in sub.itertuples():
        color=COLORS[row.family]
        marker = '*' if row.method=='U8' else ('o' if row.type=='pooled' else 's')
        ax.scatter(row.width_px2,row.coverage,s=(135 if row.method=='U8' else 68),
                   marker=marker,edgecolor=color,
                   facecolor=('white' if row.method=='U8' or row.type=='pooled' else color),
                   linewidth=1.55,zorder=4)
        dx,dy=OFFSETS[h][row.method]
        ax.annotate(row.method,(row.width_px2,row.coverage),xytext=(dx,dy),
                    textcoords='offset points',fontsize=9,weight='bold',
                    ha='center',va='center',color=color,zorder=5,
                    bbox=dict(facecolor='white',edgecolor='none',alpha=.86,pad=.65))
    ax.set_title(f'({panel}) {h}-hour forecast',loc='left',fontweight='bold',pad=13)
    ax.set_xlabel('Mean prediction-interval width (px²)',labelpad=8)
    ax.set_ylim(.815,.936)
    ax.yaxis.set_major_locator(MultipleLocator(.02))
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1,decimals=0))
    # Different x ranges prevent the two horizons' widths being visually compressed.
    ax.set_xlim((970,1590) if h==24 else (1710,2590))
    ax.grid(axis='both',color='#E4E7EB',lw=.65,zorder=0)
    ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.spines[['left','bottom']].set_color('#A8ABB0')
axs[0].set_ylabel('Genotype-macro empirical coverage',labelpad=7)

legend_fam = [Line2D([0],[0],marker='s',ls='none',ms=7,color=v,markerfacecolor=v,
                     label=k) for k,v in COLORS.items()]
legend_shape = [Line2D([0],[0],marker='o',ls='none',ms=7,markerfacecolor='white',
                       markeredgecolor='#505050',label='Pooled'),
                Line2D([0],[0],marker='s',ls='none',ms=7,markerfacecolor='#505050',
                       color='#505050',label='Balanced'),
                Line2D([0],[0],marker='*',ls='none',ms=11,markerfacecolor='white',
                       markeredgecolor=COLORS['Trajectory sensitivity'],
                       color=COLORS['Trajectory sensitivity'],label='U8: retrospective'),
                Line2D([0],[0],ls=(0,(4,3)),color='#4F5660',label='90% nominal target')]
fig.legend(handles=legend_fam,ncol=5,frameon=False,loc='lower center',
           bbox_to_anchor=(.5,.13),fontsize=8.1,columnspacing=1.7,handletextpad=.4)
fig.legend(handles=legend_shape,ncol=4,frameon=False,loc='lower center',
           bbox_to_anchor=(.5,.065),fontsize=8.1,columnspacing=2,handletextpad=.4)
# Interval scores deliberately stay in the published results table (not encoded as marker size).
output = OUT / 'F3_coverage_efficiency'
meta={'Title':'Empirical genotype-macro coverage versus interval width',
      'Creator':'Reproducible archived E2/E3/P0 analysis'}
fig.savefig(output.with_suffix('.pdf'),bbox_inches='tight',metadata=meta)
fig.savefig(output.with_suffix('.svg'),bbox_inches='tight',
            metadata={'Title':meta['Title'],'Creator':meta['Creator']})
fig.savefig(output.with_suffix('.png'),bbox_inches='tight',dpi=500)
plt.close(fig)
print('Saved:', *(str(output.with_suffix(ext)) for ext in ('.pdf','.svg','.png')),sep='\n  ')
