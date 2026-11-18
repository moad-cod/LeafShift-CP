#!/usr/bin/env python3
"""Rebuild S1 from P1.2 archived genotype data. Run from any working directory."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter

ROOT = Path(__file__).resolve().parents[3]  # figures/rebuilt/S1_growth_associations/script.py
BASE = ROOT / 'experiments/P1/P1_2/tables'
OUT = ROOT / 'figures/rebuilt/S1_growth_associations'
OUT.mkdir(parents=True, exist_ok=True)
context = pd.read_csv(BASE / 'p1_growth_coverage_genotype_context.csv')
reported = pd.read_csv(BASE / 'p1_growth_analysis_summary.csv')
cols = {'horizon_hours','test_genotype','subset','coverage','growth','plants','windows'}
assert cols <= set(context.columns), f'Missing columns: {cols - set(context.columns)}'
assert {'horizon_hours','subset','spearman_rho_across_genotypes','genotypes_for_correlation'} <= set(reported.columns)
context = context.loc[context['subset'].eq('full')].copy()
assert len(context) == 42, f'Expected 42 full-subset rows; got {len(context)}'
assert not context.duplicated(['horizon_hours','test_genotype']).any()
assert context.groupby('horizon_hours')['test_genotype'].nunique().to_dict() == {24:21,48:21}
assert context[['growth','coverage']].notna().all().all()
assert (context.growth >= 0).all() and context.coverage.between(0,1).all()
assert (context.plants > 0).all()
# Reproduce the archived across-genotype rank correlations.
for h in (24,48):
    d = context[context.horizon_hours == h]
    ref = reported[(reported.horizon_hours == h) & (reported.subset == 'full')]
    assert len(ref) == 1 and int(ref.iloc[0].genotypes_for_correlation) == 21
    rho = d.growth.corr(d.coverage, method='spearman')
    expected = float(ref.iloc[0].spearman_rho_across_genotypes)
    assert abs(rho - expected) < 1e-6, f'{h}h rho mismatch: {rho} vs {expected}'
    print(f'{h}h: 21 genotypes, Spearman rho={rho:.6f} — PASS')

context.sort_values(['horizon_hours','test_genotype']).to_csv(OUT/'Figure_S1_source.csv',index=False,float_format='%.16g')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,
                     'axes.titlesize':11,'xtick.labelsize':9,'ytick.labelsize':9,
                     'pdf.fonttype':42,'svg.fonttype':'none'})
fig, axes = plt.subplots(1,2,figsize=(10.6,4.65),sharey=True)
fig.subplots_adjust(left=.085,right=.985,bottom=.20,top=.82,wspace=.14)
for ax,h,panel in zip(axes,(24,48),('a','b')):
    d = context[context.horizon_hours == h].sort_values('growth')
    rho = d.growth.corr(d.coverage,method='spearman')
    ax.scatter(d.growth,d.coverage,s=47,c='#D55E00',edgecolors='#222222',linewidths=.5,zorder=3)
    # Descriptive linear trend (not Spearman model / inference).
    fit = np.polyfit(d.growth,d.coverage,1)
    xx = np.linspace(d.growth.min(),d.growth.max(),100)
    ax.plot(xx,np.polyval(fit,xx),ls='--',lw=1.2,color='#666666',zorder=2)
    ax.axhline(.90,color='#70777F',ls=':',lw=1.1)
    ax.set_title(f'({panel}) {h}-hour horizon  |  Spearman ρ = {rho:.2f}',loc='left',pad=12,weight='bold')
    ax.set_xlabel('Plant-averaged historical absolute growth (px² h⁻¹)')
    ax.grid(alpha=.18)
    ax.yaxis.set_major_formatter(PercentFormatter(1,decimals=0))
    ax.spines[['top','right']].set_visible(False)
axes[0].set_ylabel('Genotype-specific U5 empirical coverage')
fig.text(.085,.92,'Exploratory growth–coverage association across held-out genotypes',fontsize=13,weight='bold')
fig.text(.50,.055,'Each point is one held-out genotype (n = 21 per horizon). Dashed fits are descriptive; not causal or confirmatory.',
         ha='center',fontsize=8,color='#555555')
stem=OUT/'Figure_S1_Exploratory_Growth_Associations'
for ext,extra in [('pdf',{}),('svg',{}),('png',{'dpi':500})]:
    fig.savefig(stem.with_suffix('.'+ext),bbox_inches='tight',**extra)
plt.close(fig)
print('Saved:',str(stem)+'{.pdf,.svg,.png}', 'and Figure_S1_source.csv')
