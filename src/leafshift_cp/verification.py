"""Verify genuine archived evidence; never fit a model or manufacture outputs."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import ROOT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(root=ROOT):
    root = Path(root)
    checks = []

    def check(name, observed, expected, atol=1e-9, source=""):
        ok = bool(np.isclose(observed, expected, rtol=0, atol=atol))
        checks.append(dict(name=name, observed=float(observed), expected=float(expected),
                           tolerance=atol, status="pass" if ok else "fail", source=source,
                           evidence_level="verified from archived outputs"))

    with (root / "docs/FILE_INVENTORY.csv").open() as handle:
        inventory = list(csv.DictReader(handle))
    for row in inventory:
        if row["decision"] == "copied":
            p = root / row["destination"]
            checks.append(dict(name="copy integrity: " + row["destination"],
                               status="pass" if p.is_file() and digest(p) == row["sha256"] else "fail",
                               source=row["original_path"], evidence_level="byte-verified source copy"))
    folds = json.loads((root / "configs/E1_1_genotype_folds.json").read_text())["folds"]
    check("outer LOGO folds", len(folds), 21)
    check("distinct held-out genotypes", len({f["test_genotype"] for f in folds}), 21)
    for f in folds:
        dev, test = set(f["development_genotypes"]), f["test_genotype"]
        check(f"fold {f['outer_fold']} development count", len(dev), 20)
        check(f"fold {f['outer_fold']} assignments", len(f["inner_folds"]), 4)
        for inner in f["inner_folds"]:
            fit, cal = set(inner["train_genotypes"]), set(inner["validation_genotypes"])
            valid = len(fit) == 15 and len(cal) == 5 and not (fit & cal) and test not in fit | cal and fit | cal == dev
            check(f"fold {f['outer_fold']} assignment {inner['inner_fold']} 15/5/1 disjointness", int(valid), 1)

    metadata = json.loads((root / "data/manifests/manifest_metadata.json").read_text())
    for row in metadata["summary"]:
        h = row["horizon_hours"]
        check(f"{h}h windows", row["windows"], {24: 121431, 48: 107214}[h], source="data/manifests/manifest_metadata.json")
        check(f"{h}h plants", row["eligible_plants"], 318)
        check(f"{h}h genotypes", row["eligible_genotypes"], 21)
    check("total windows", sum(r["windows"] for r in metadata["summary"]), 228645)
    plants = pd.read_csv(root / "data/manifests/retained_observations_by_plant.csv")
    check("retained leaf-area observations", plants.retained_observations.sum(), 165796,
          source="data/manifests/retained_observations_by_plant.csv")
    lineage = json.loads((root / "data/manifests/observation_lineage.json").read_text())
    check("retained observation summary digest", int(digest(root / lineage["derived_file"]) == lineage["derived_sha256"]), 1)
    point = pd.read_csv(root / "experiments/E1/tables/E1_5_genotype_metrics.csv")
    point_summary = pd.read_csv(root / "experiments/E1/tables/E1_5_summary.csv")
    for h, expected in [(24, 309.92), (48, 518.51)]:
        observed = point[point.horizon_hours == h].B3_mae.mean()
        check(f"HGB {h}h macro MAE rounded", observed, expected, .005,
              "experiments/E1/tables/E1_5_genotype_metrics.csv")
        check(f"HGB {h}h archived aggregation", observed, point_summary[point_summary.horizon_hours == h].iloc[0].B3_macro_mae)

    genotype = pd.read_csv(root / "experiments/P3/P3_6/P3_6_genotype_metrics.csv")
    summary = pd.read_csv(root / "experiments/P3/P3_6/P3_6_method_summary.csv")
    for row in summary.itertuples(index=False):
        group = genotype[(genotype.horizon_hours == row.horizon_hours) & (genotype.method == row.method)]
        check(f"{row.method} {row.horizon_hours}h genotype count", group.test_genotype.nunique(), 21)
        for metric in ["coverage", "width", "score", "lower_miss", "upper_miss"]:
            check(f"{row.method} {row.horizon_hours}h {metric} aggregation", group[metric].mean(), getattr(row, metric))
    targets = {24: (89.97120, 1479.059, 2139.587, 82.217, 2130.538),
               48: (90.04006, 2385.689, 3389.092, 76.393, 3364.958)}
    for h, (coverage, width, score, worst, adaptive) in targets.items():
        u5 = genotype[(genotype.horizon_hours == h) & (genotype.method == "U5")]
        sga = genotype[(genotype.horizon_hours == h) & (genotype.method == "SGA_CQR")]
        for label, obs, exp, tol in [("coverage percent",100*u5.coverage.mean(),coverage,.000005),
                                    ("width",u5.width.mean(),width,.0005),
                                    ("score",u5.score.mean(),score,.0005),
                                    ("worst coverage percent",100*u5.coverage.min(),worst,.0005),
                                    ("SGA-CQR score",sga.score.mean(),adaptive,.0005)]:
            check(f"{h}h manuscript {label}",obs,exp,tol,"experiments/P3/P3_6/P3_6_genotype_metrics.csv")

    fold = pd.read_csv(root / "experiments/E3/tables/E3_2_fold_metrics.csv")
    plant = pd.read_csv(root / "experiments/P1/P1_1/tables/P1_1_U5_plant_assignment_coverage.csv")
    for h in [24, 48]:
        p = plant[plant.horizon_hours == h]
        check(f"{h}h plant assignment coverage reconstructs U5",p.groupby(["test_genotype","inner_fold"]).coverage.mean().groupby("test_genotype").mean().mean(),targets[h][0]/100,5e-8)
        for metric in ["coverage", "width", "score"]:
            f = fold[(fold.horizon_hours == h) & (fold.method == "U5_CQR_balanced")]
            g = genotype[(genotype.horizon_hours == h) & (genotype.method == "U5")]
            check(f"{h}h E3 to P3 U5 {metric}",f.groupby("test_genotype")[metric].mean().mean(),g[metric].mean())

    all_methods = pd.read_csv(root / "manuscript/tables/main/Table_03_Conformal_Performance.csv")
    check("U0–U8 procedure coverage",int(set(all_methods.Method)=={f"U{i}" for i in range(9)}),1)
    for r in all_methods.itertuples(index=False):
        if r.Method == "U5":
            check(f"paper table U5 {r.Horizon_h}h coverage",r.Genotype_macro_coverage,targets[r.Horizon_h][0]/100,5e-8)
    directional = pd.read_csv(root / "experiments/P3/P3_2/P3_2_genotype_directional.csv")
    check("directional coverage decomposition",np.max(np.abs(directional.coverage+directional.lower_miss+directional.upper_miss-1)),0)
    return {"scope":"Archived evidence and integrity only; no retraining or raw-data reconstruction", "checks":checks,
            "summary":{"checks":len(checks),"passed":sum(c["status"]=="pass" for c in checks),"failed":sum(c["status"]=="fail" for c in checks)}}
