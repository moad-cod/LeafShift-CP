#!/usr/bin/env python3
"""P1.4 non-destructive audit of projected leaf area and spatial calibration.

Run from repository root: uv run --active python src/P1/p1_4_biological_scale_audit.py
No physical conversion is performed without externally validated calibration.
"""
from __future__ import annotations
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'data/processed/forecasting/forecast_windows.csv'
PRED_DIR = ROOT / 'experiments/P1/P1_1/tables'
OUT = ROOT / 'experiments/P1/P1_4'
TABLES, FIGURES, REPORTS = (OUT / x for x in ('tables','figures','reports'))
SEED = 20261109


def save_csv(data, name):
    path = TABLES / name
    pd.DataFrame(data).to_csv(path, index=False)
    return path


def calibration_inventory():
    """List likely calibration artifacts; a hit is NOT proof of a valid factor."""
    candidates=[]
    roots = [ROOT/'configs', ROOT/'Research', ROOT/'data'/'processed', ROOT/'src']
    pattern = re.compile(r'calibrat|mm.?per.?pix|pixel.?size|pixel.?to.?mm|spatial.?scale|scale.?bar|camera.?height|field.?of.?view|ground.?sampling|mm²|mm2|cm²|cm2',re.I)
    allowed = {'.md','.txt','.json','.yaml','.yml','.py','.toml','.csv'}
    for directory in roots:
        if not directory.exists():
            continue
        for path in sorted(directory.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in allowed:
                continue
            if any(part in {'.git','.venv','__pycache__'} for part in path.parts):
                continue
            if path.stat().st_size > 2_000_000:
                continue
            try:
                with path.open(encoding='utf-8',errors='replace') as fh:
                    for ln, line in enumerate(fh,1):
                        if pattern.search(line):
                            candidates.append({'file':str(path.relative_to(ROOT)), 'line':ln,'excerpt':line.strip()[:250]})
                            if len(candidates)>=250:
                                return candidates
            except (OSError,UnicodeError):
                continue
    return candidates


def main():
    for folder in (TABLES,FIGURES,REPORTS): folder.mkdir(parents=True,exist_ok=True)
    if not MANIFEST.exists(): raise FileNotFoundError(MANIFEST)
    usecols=['window_id','horizon_hours','genotype','plant_id','leaf_id','target_area_pixels','last_observed_area_pixels']
    df=pd.read_csv(MANIFEST,usecols=usecols)
    if df.window_id.duplicated().any(): raise ValueError('Nonunique window IDs')
    if df[usecols].isna().any().any(): raise ValueError('Missing manifest values')
    if not df.horizon_hours.isin([24,48]).all(): raise ValueError('Unexpected horizon')
    if (df.target_area_pixels < 0).any() or (df.last_observed_area_pixels <= 0).any():
        raise ValueError('Invalid leaf areas')
    inventory=calibration_inventory()
    save_csv(inventory, 'p1_4_calibration_candidate_inventory.csv')
    records=[]
    summaries=[]
    for horizon in (24,48):
        d=df.loc[df.horizon_hours.eq(horizon)].copy()
        pred_path=PRED_DIR / f'P1_1_U5_window_predictions_{horizon}h.csv'
        if not pred_path.exists(): raise FileNotFoundError(pred_path)
        cols=['window_id','inner_fold','test_genotype','plant_id','leaf_id','target_area_pixels','lower_pixels','upper_pixels','covered']
        pred=pd.read_csv(pred_path,usecols=cols)
        if len(pred)!=4*len(d): raise ValueError(f'{horizon}h prediction count does not match 4 assignments')
        if pred.duplicated(['window_id','inner_fold']).any(): raise ValueError('Duplicate prediction identity')
        if not pred.inner_fold.isin([1,2,3,4]).all(): raise ValueError('Invalid fold ID')
        counts=pred.groupby('window_id').inner_fold.nunique()
        if len(counts)!=len(d) or not counts.eq(4).all(): raise ValueError('Each window must occur in all four assignments')
        matched=pred.merge(d,on='window_id',how='left',validate='many_to_one',suffixes=('','_src'),indicator=True)
        if not matched['_merge'].eq('both').all(): raise ValueError('Unmatched window IDs')
        for field,src in [('test_genotype','genotype'),('plant_id','plant_id_src'),('leaf_id','leaf_id_src'),('target_area_pixels','target_area_pixels_src')]:
            x,y=matched[field].to_numpy(),matched[src].to_numpy()
            ok=np.isclose(x,y,rtol=0,atol=1e-8) if field=='target_area_pixels' else (x==y)
            if not np.all(ok): raise ValueError(f'{horizon}h inconsistent {field}')
        if not matched.horizon_hours.eq(horizon).all(): raise ValueError('Horizon mismatch')
        lo,hi,target=(matched[c].to_numpy(dtype=float) for c in ('lower_pixels','upper_pixels','target_area_pixels'))
        if not(np.isfinite(lo).all() and np.isfinite(hi).all() and (hi>=lo).all()):
            raise ValueError('Invalid interval bounds')
        covered=((target>=lo)&(target<=hi)).astype(int)
        if not np.array_equal(covered,matched.covered.to_numpy(dtype=int)): raise ValueError('Coverage mismatch')
        matched['width_px2']=hi-lo
        matched['width_over_target_area']=np.where(target>0,(hi-lo)/np.maximum(target,1),np.nan)
        # Across folds: average each window across four assignments; then aggregate at plant/genotype level.
        window_mean=matched.groupby(['window_id','test_genotype','plant_id'],as_index=False).agg(
            width_px2=('width_px2','mean'),
            covered=('covered','mean'),
            width_over_target_area=('width_over_target_area','mean'),
            target_area_pixels=('target_area_pixels','first'),
            last_observed_area_pixels=('last_observed_area_pixels','first'),
        )
        window_mean['horizon_hours']=horizon
        if len(window_mean)!=len(d): raise ValueError('Unexpected averaged-window count')
        records.append(window_mean)
        # Genotype summaries use equal plant mass, not window mass.
        plant=window_mean.groupby(['test_genotype','plant_id'],as_index=False).agg(
            windows=('window_id','size'),coverage=('covered','mean'),
            mean_width_px2=('width_px2','mean'),
            mean_width_over_target=('width_over_target_area','mean'),
            median_target_area_px2=('target_area_pixels','median'),
        )
        plant['horizon_hours']=horizon
        summaries.append(plant)
        print(f'{horizon}h | {len(d):,} distinct windows | {len(plant)} plants | {plant.test_genotype.nunique()} genotypes')
        print(f'  Target leaf area median: {d.target_area_pixels.median():.1f} px²; 10–90%: {d.target_area_pixels.quantile(.1):.1f}–{d.target_area_pixels.quantile(.9):.1f} px²')
        print(f'  U5 median window interval width: {window_mean.width_px2.median():.1f} px²')
        print(f'  Typical per-window width / target ratio (median): {window_mean.width_over_target_area.median():.2f}')
    windows=pd.concat(records,ignore_index=True)
    plants=pd.concat(summaries,ignore_index=True)
    save_csv(plants,'p1_4_plant_scale_summary.csv')
    genotype=plants.groupby(['horizon_hours','test_genotype'],as_index=False).agg(
        plants=('plant_id','nunique'),
        coverage=('coverage','mean'),
        mean_interval_width_px2=('mean_width_px2','mean'),
        mean_width_over_target=('mean_width_over_target','mean'),
        median_plant_target_area_px2=('median_target_area_px2','median'),
    )
    save_csv(genotype,'p1_4_genotype_scale_summary.csv')
    summary=[]
    for h in (24,48):
        wd=windows[windows.horizon_hours.eq(h)]
        gp=genotype[genotype.horizon_hours.eq(h)]
        summary.append({'horizon_hours':h,'n_windows':len(wd),'n_genotypes':len(gp),
            'target_area_median_px2':wd.target_area_pixels.median(),
            'target_area_q10_px2':wd.target_area_pixels.quantile(.1),
            'target_area_q90_px2':wd.target_area_pixels.quantile(.9),
            'window_width_median_px2':wd.width_px2.median(),
            'window_width_over_target_median':wd.width_over_target_area.median(),
            'genotype_macro_width_px2':gp.mean_interval_width_px2.mean(),
            'genotype_macro_coverage':gp.coverage.mean()})
    save_csv(summary,'p1_4_biological_scale_summary.csv')
    # Plot genotype widths and empirical coverages, without asserting physical units.
    import matplotlib.pyplot as plt
    for horizon in (24,48):
        s=genotype[genotype.horizon_hours.eq(horizon)].sort_values('mean_interval_width_px2')
        fig,ax=plt.subplots(figsize=(10,6))
        ax.barh(s.test_genotype,s.mean_interval_width_px2)
        ax.set_xlabel('U5 mean interval width (pixel area, px²)')
        ax.set_ylabel('Held-out genotype')
        ax.set_title(f'{horizon}h — Plant-aggregated U5 interval width by genotype')
        fig.tight_layout()
        fig.savefig(FIGURES/f'p1_4_u5_width_by_genotype_{horizon}h.png',dpi=180)
        plt.close(fig)
    lines=['# P1.4 — Biological scale and spatial-calibration audit','','## Result',
        '- No spatial conversion factor was assumed or inferred from image dimensions.',
        '- Calibration keyword matches are candidate evidence only; inspect source provenance before using a physical unit.',
        '- All pixel-area statistics below refer to projected visible, segmentation-derived leaf area, not biomass or true leaf surface area.',
        '- Ratios of width to target area are descriptive and computed on observed outcomes; they are not forecast-time features.',
        '- Repeated assignments are averaged per window; genotype summary is plant-aggregated.','','## Numerical audit','',
        '| Horizon | Median area (px²) | Median U5 width (px²) | Median window width / actual area | Macro coverage |',
        '|---|---:|---:|---:|---:|']
    for row in summary:
        lines.append(f"| {row['horizon_hours']}h | {row['target_area_median_px2']:.2f} | {row['window_width_median_px2']:.2f} | {row['window_width_over_target_median']:.3f} | {row['genotype_macro_coverage']:.4f} |")
    lines.extend(['','## Physical units','',
        '**PENDING CALIBRATION EVIDENCE.** Only convert to mm² after verifying a documented spatial mapping for these exact processed images, including any resizing/cropping. For constant area scale k (mm²/px²): A_mm² = k × A_px². Without k, retain px².',
        '',f'Calibration search returned {len(inventory)} candidate lines (not validated).',
        '','## Important limitations','',
        '- Visible projected leaf area differs from true biological leaf surface area due to overlap and imaging geometry.',
        '- The dataset comes from controlled top-view imaging; cross-experiment physical comparison requires spatial calibration.',
        '- Width relative to true target can be large for very small areas; interpret distributions and representative cases, not just a mean.',
        '- This does not assess practical decision costs or establish environmental generalization.',
        ''])
    (REPORTS/'p1_4_biological_scale_audit.md').write_text('\n'.join(lines),encoding='utf-8')
    print(f'Calibration candidate lines: {len(inventory)} (none automatically validated)')
    print(f'PASS — Outputs: {OUT}')

if __name__=='__main__': main()
