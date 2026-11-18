#!/usr/bin/env python3
"""Read-only P0.5 evidence inventory/reconciliation; never modifies source experiments.
Run from repository root: uv run --active python src/P0/run_p05_evidence_audit.py
"""
from pathlib import Path
import csv, json, math
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'experiments/P0/P0_5'
OUT.mkdir(parents=True, exist_ok=True)
TABLES = OUT / 'tables'; TABLES.mkdir(exist_ok=True)
REPORTS = OUT / 'reports'; REPORTS.mkdir(exist_ok=True)

source = ROOT / 'data/processed/forecasting/forecast_windows.csv'
foldfile = ROOT / 'configs/E1_1_genotype_folds.json'
checks = []
def check(name, status, details, path=''):
    checks.append(dict(check=name,status=status,details=str(details),source_path=str(path)))
    print(f'[{status}] {name}: {details}')

if not source.exists():
    check('manifest', 'FAIL', 'Missing forecasting manifest', source)
else:
    s = pd.read_csv(source, usecols=['window_id','genotype','plant_id','leaf_id','horizon_hours','target_area_pixels'])
    check('manifest_row_count', 'PASS' if len(s)==228645 else 'REVIEW', f'{len(s):,} rows', source)
    check('unique_window_ids', 'PASS' if s.window_id.is_unique else 'FAIL', f'duplicates={s.window_id.duplicated().sum()}', source)
    check('nonnull_identifiers', 'PASS' if s[['window_id','genotype','plant_id','leaf_id']].notna().all().all() else 'FAIL','checked',source)
    collision=s.groupby('plant_id')['genotype'].nunique()
    check('plant_genotype_uniqueness','PASS' if (collision <=1).all() else 'FAIL', f'cross-genotype collision plants={(collision>1).sum()}',source)
    for horizon, expected in [(24,121431),(48,107214)]:
        h=s[s.horizon_hours.eq(horizon)]
        tracks=h[['genotype','plant_id','leaf_id']].drop_duplicates()
        check(f'{horizon}h_records','PASS' if len(h)==expected else 'FAIL',f'windows={len(h):,}; genotypes={h.genotype.nunique()}; plants={h.plant_id.nunique()}; tracks={len(tracks):,}', source)

if foldfile.exists():
    folds=json.loads(foldfile.read_text())['folds']
    check('outer_folds','PASS' if len(folds)==21 else 'FAIL',f'n={len(folds)}',foldfile)
    seen=[]; inner_ok=True
    for o in folds:
        test=o['test_genotype']; seen.append(test)
        inners=o['inner_folds']
        if len(inners)!=4:inner_ok=False
        for it in inners:
            train=set(it['train_genotypes']);cal=set(it['validation_genotypes'])
            if len(train)!=15 or len(cal)!=5 or train&cal or test in train|cal: inner_ok=False
    check('split_integrity','PASS' if inner_ok and len(set(seen))==21 else 'FAIL', '21 distinct outer test genotypes × 4 valid assignments' if inner_ok else 'invalid split found',foldfile)
else:check('fold_config','FAIL','Missing fold manifest',foldfile)

# Gather actual empirical artifacts; avoid inventing schemas or relying on unseen files.
patterns=[
 ('E0','experiments/E0/**/*.csv'),('E1','experiments/E1/**/*.csv'),
 ('E2','experiments/E2/**/*.csv'),('E3','experiments/E3/**/*.csv'),
 ('E4','experiments/E4/**/*.csv'),('P0.4a','experiments/P0/P0_4a/**/*.csv'),
 ('P0.4b','experiments/P0/P0_4b/**/*.csv'),('P0.4c','experiments/P0/P0_4c_U8/**/*.csv')]
rows=[]
for label,pat in patterns:
    paths=sorted(ROOT.glob(pat))
    check(f'{label}_csv_inventory','PASS' if paths else 'REVIEW',f'{len(paths)} CSV files found',label)
    for path in paths:
        try:
            with path.open(newline='',encoding='utf-8') as f:
                reader=csv.reader(f); header=next(reader,[])
            rows.append(dict(section=label,path=str(path.relative_to(ROOT)),n_columns=len(header),columns=' | '.join(header),size_bytes=path.stat().st_size))
        except Exception as exc:
            rows.append(dict(section=label,path=str(path.relative_to(ROOT)),n_columns=-1,columns=f'ERROR: {exc}',size_bytes=0))
pd.DataFrame(rows,columns=['section','path','n_columns','columns','size_bytes']).to_csv(TABLES/'P0_5_source_inventory.csv',index=False)

# Compare reported U1/U8 numeric outputs if the summary contains the expected column names.
u8=ROOT/'experiments/P0/P0_4c_U8/tables/U8_summary.csv'
if u8.exists():
    d=pd.read_csv(u8)
    required={'horizon_hours','method','macro_coverage','macro_width','macro_interval_score','worst_genotype_coverage','genotypes_below_90'}
    if not required.issubset(d.columns):
        check('U8_summary_schema','REVIEW',f'columns={list(d.columns)}',u8)
    else:
        expected={
          (24,'U1_balanced'):(.8992,1394.9378,2287.0689,.8216,9),
          (24,'U8_trajectory_balanced'):(.9042,1429.8331,2285.3365,.8292,9),
          (48,'U1_balanced'):(.8987,2305.0414,3528.2520,.7781,9),
          (48,'U8_trajectory_balanced'):(.9123,2464.0079,3530.1466,.8019,6),
        }
        mismatches=[]
        for (h,m), values in expected.items():
            sub=d[(d.horizon_hours==h)&(d.method==m)]
            if len(sub)!=1:
                mismatches.append(f'{h} {m}: expected one row, got {len(sub)}');continue
            row=sub.iloc[0]
            for col,v,tol in zip(['macro_coverage','macro_width','macro_interval_score','worst_genotype_coverage','genotypes_below_90'],values,[.00015,.03,.03,.00015,0]):
                if abs(float(row[col])-v)>tol:mismatches.append(f'{h} {m} {col}: file={row[col]}, expected≈{v}')
        check('U8_terminal_summary_reproduction','PASS' if not mismatches else 'FAIL', 'four method/horizon rows match reported rounded values' if not mismatches else '; '.join(mismatches),u8)
else: check('U8_summary_file','REVIEW','U8 summary not found at expected path; see inventory',u8)

pd.DataFrame(checks).to_csv(TABLES/'P0_5_audit_checks.csv',index=False)
fail=[x for x in checks if x['status']=='FAIL']
review=[x for x in checks if x['status']=='REVIEW']
lines=[
 '# P0.5 — Repository Evidence Reconciliation', '',
 'This report is generated from local files. It checks metadata, fold structure and selected U8 summary values; it is **not** a full review of every model formula or a statistical validation.', '',
 f'- PASS checks: {sum(x["status"]=="PASS" for x in checks)}',
 f'- REVIEW checks: {len(review)}',f'- FAIL checks: {len(fail)}','',
 '| Check | Status | Observation |','|---|---|---|',
]
for x in checks:
    lines.append(f'| {x["check"]} | {x["status"]} | {x["details"].replace("|","/")} |')
lines += ['', '## Follow-up','', 'Inspect all REVIEW and FAIL items; use the inventory CSV to locate the actual E0–E4 files and construct a claim-to-file evidence map. P0.6 novelty review must be performed separately.']
(REPORTS/'P0_5_repository_reconciliation.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(f'\nSaved: {TABLES/"P0_5_audit_checks.csv"}')
print(f'Saved: {TABLES/"P0_5_source_inventory.csv"}')
print(f'Saved: {REPORTS/"P0_5_repository_reconciliation.md"}')
print(f'PASS={sum(x["status"]=="PASS" for x in checks)}, REVIEW={len(review)}, FAIL={len(fail)}')
