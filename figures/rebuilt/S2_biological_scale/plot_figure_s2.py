#!/usr/bin/env python3
"""Rebuild S2 from P1.4 archived plant summaries, verifying window-level medians."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT/'experiments/P1/P1_4/tables'
OUT = ROOT/'figures/rebuilt/S2_biological_scale'
OUT.mkdir(parents=True,exist_ok=True)
plants = pd.read_csv(BASE/'p1_4_plant_scale_summary.csv')
summary = pd.read_csv(BASE/'p1_4_biological_scale_summary.csv')
needed={'horizon_hours','test_genotype','plant_id','windows','mean_width_px2','mean_width_over_target','median_target_area_px2'}
assert needed <= set(plants.columns),f'Missing columns: {needed-set(plants.columns)}'
assert {'horizon_hours','n_windows','n_genotypes','window_width_over_target_median'} <= set(summary.columns)
assert not plants.duplicated(['horizon_hours','plant_id']).any()
assert plants.groupby('horizon_hours')['plant_id'].nunique().to_dict() == {24:318,48:318},'Expected 318 plants per horizon'
assert plants.groupby('horizon_hours')['test_genotype'].nunique().to_dict() == {24:21,48:21}
assert len(summary)==2 and set(summary.horizon_hours)=={24,48}
assert plants[['mean_width_px2','mean_width_over_target','median_target_area_px2']].notna().all().all()
assert (plants[['mean_width_px2','mean_width_over_target','median_target_area_px2']] > 0).all().all()
for h, expected_windows in ((24,121431),(48,107214)):
    d=plants.loc[plants.horizon_hours==h]
    s=summary[summary.horizon_hours==h].iloc[0]
    assert int(d.windows.sum())==expected_windows==int(s.n_windows),f'{h}h window sum mismatch'
    assert int(s.n_genotypes)==21
    print(f'{h}h: 318 plants, {expected_windows:,} windows — PASS; '
          f'archived window-level median width/target={s.window_width_over_target_median:.6f}; '
          f'plant-level median of plant means={d.mean_width_over_target.median():.6f}')
plants.sort_values(['horizon_hours','test_genotype','plant_id']).to_csv(OUT/'Figure_S2_source.csv',index=False,float_format='%.16g')
BLUE='#0072B2'; ORANGE='#D55E00'
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':10,'axes.titlesize':11,
                     'xtick.labelsize':9,'ytick.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(10.6,4.9),gridspec_kw={'width_ratios':[.88,1.12]})
fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.81,wspace=.32)
fig.text(.09,.92,'Biological-scale diagnostics of U5 interval width',fontsize=13,weight='bold')
# This is a distribution of PLANT-level mean ratios, NOT the archive's window-level median ratio.
vals=[plants.loc[plants.horizon_hours==h,'mean_width_over_target'].to_numpy() for h in (24,48)]
bp=ax1.boxplot(vals,positions=[1,2],widths=.30,patch_artist=True,
               showfliers=True,flierprops={'markersize':3,'markerfacecolor':'white','markeredgecolor':'#777'},
               medianprops={'color':'#202020','linewidth':1.5},
               whiskerprops={'color':'#333'},capprops={'color':'#333'})
for patch,color in zip(bp['boxes'],[BLUE,ORANGE]):
    patch.set_facecolor(color);patch.set_alpha(.52)
ax1.set_yscale('log')
ax1.set_xticks([1,2],['24 h','48 h'])
ax1.set_title('(a) Plant-level relative interval widths',loc='left',weight='bold',pad=12)
ax1.set_ylabel('Plant mean of window width / observed target area')
ax1.grid(axis='y',alpha=.2)
ax1.spines[['top','right']].set_visible(False)
# Points are plant summaries; x is median target area and y is mean absolute width.
for h,color in ((24,BLUE),(48,ORANGE)):
    d=plants[plants.horizon_hours==h]
    ax2.scatter(d.median_target_area_px2,d.mean_width_px2,s=20,alpha=.60,
                edgecolors='none',color=color,label=f'{h} h')
ax2.set_xscale('log')
ax2.set_title('(b) Absolute width versus observed area',loc='left',weight='bold',pad=12)
ax2.set_xlabel('Plant median observed target area (px²)')
ax2.set_ylabel('Plant mean U5 interval width (px²)')
ax2.grid(alpha=.18)
ax2.legend(frameon=False,loc='upper left')
ax2.spines[['top','right']].set_visible(False)
reference=', '.join(f'{h} h: {float(summary.loc[summary.horizon_hours==h,"window_width_over_target_median"].iloc[0]):.2f}' for h in (24,48))
fig.text(.50,.082,'Separate archived window-level median ratios (not boxplot medians): '+reference,
         ha='center',fontsize=8,color='#444444')
fig.text(.50,.044,'Ratios use observed future target area and are retrospective—not available at forecast time.',
         ha='center',fontsize=8,color='#555555')
stem=OUT/'Figure_S2_Interval_Width_and_Leaf_Area'
for ext,extra in [('pdf',{}),('svg',{}),('png',{'dpi':500})]:
    fig.savefig(stem.with_suffix('.'+ext),bbox_inches='tight',**extra)
plt.close(fig)
print('Saved:',str(stem)+'{.pdf,.svg,.png}', 'and Figure_S2_source.csv')
