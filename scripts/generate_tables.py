"""Re-aggregate original fold/genotype evidence into paper summary tables."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT=Path(__file__).resolve().parents[1]

def generate():
    sources=[('experiments/E2/tables/E2_4_split_metrics.csv',{'U0','U1'}),
             ('experiments/E3/tables/E3_1_fold_metrics.csv',{'U2','U3'}),
             ('experiments/E3/tables/E3_2_fold_metrics.csv',{'U4','U5'}),
             ('experiments/E3/tables/E3_3_fold_metrics.csv',{'U6','U7'}),
             ('experiments/P0/P0_4c_U8/tables/U8_fold_metrics.csv',{'U8'})]
    frames=[]
    for source,methods in sources:
        frame=pd.read_csv(ROOT/source)
        frame['Method']=frame.method.str.split('_').str[0]
        frame=frame[frame.Method.isin(methods)]
        assert not frame.duplicated(['Method','horizon_hours','test_genotype','inner_fold']).any()
        assert frame.groupby(['Method','horizon_hours','test_genotype']).inner_fold.nunique().eq(4).all()
        frames.append(frame)
    combined=pd.concat(frames,ignore_index=True)
    genotype=combined.groupby(['Method','horizon_hours','test_genotype'],as_index=False)[['coverage','width','score']].mean()
    rows=[]
    canonical=pd.read_csv(ROOT/'manuscript/tables/main/Table_03_Conformal_Performance.csv')
    for (method,h),group in genotype.groupby(['Method','horizon_hours']):
        assert group.test_genotype.nunique()==21
        row=dict(Method=method,Horizon_h=h,Genotype_macro_coverage=group.coverage.mean(),Worst_observed_genotype_coverage=group.coverage.min(),Mean_interval_width_px2=group.width.mean(),Mean_interval_score=group.score.mean(),Genotypes_below_90=int((group.coverage<.9).sum()))
        reference=canonical[(canonical.Method==method)&(canonical.Horizon_h==h)].iloc[0]
        for metric in ['Genotype_macro_coverage','Worst_observed_genotype_coverage','Mean_interval_width_px2','Mean_interval_score','Genotypes_below_90']:
            assert np.isclose(row[metric],reference[metric],rtol=0,atol=1e-8),(method,h,metric,row[metric],reference[metric])
        rows.append(row)
    output=ROOT/'results/generated_tables'
    output.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output/'conformal_summary.csv',index=False,float_format='%.17g')
    adaptive=pd.read_csv(ROOT/'experiments/P3/P3_6/P3_6_genotype_metrics.csv')
    adaptive.groupby(['horizon_hours','method'])[['coverage','width','score','lower_miss','upper_miss']].mean().to_csv(output/'adaptive_summary.csv',float_format='%.17g')
    point=pd.read_csv(ROOT/'experiments/E1/tables/E1_5_genotype_metrics.csv')
    point.groupby('horizon_hours')[['B2_mae','B3_mae']].mean().to_csv(output/'point_summary.csv',float_format='%.17g')
    (output/'validation.json').write_text(json.dumps({'status':'pass','source_level':'Original folds already aggregate windows within plants; assignments averaged within genotype, then equal-genotype means','conformal_rows':len(rows),'source_files':[s for s,_ in sources]},indent=2)+'\n')
    print('Validated and regenerated 18 U0–U8 method/horizon rows, point and adaptive summaries.')

if __name__=='__main__':generate()
