"""Dispatch unchanged research scripts into a separate working copy."""
from pathlib import Path
import json
import os
import shutil
import subprocess
import sys

from . import ROOT
from .verification import digest

MANIFEST = "data/processed/forecasting/forecast_windows.csv"
PREDICTIONS = [f"experiments/P1/P1_1/tables/P1_1_U5_window_predictions_{h}h.csv" for h in [24,48]]
FIGURES = [
    ("2", "F2_point_forecasting", "plot_figure2.py", "F2_point_forecasting", "F2_point_forecasting"),
    ("3", "F3_method_comparison", "plot_figure3.py", "F3_coverage_efficiency", "F3_coverage_efficiency"),
    ("4", "F4_genotype_reliability", "plot_figure4_layout_v2.py", "F4_U5_genotype_reliability", "F4_genotype_reliability"),
    ("5", "F5_assignment_reliability", "plot_figure5.py", "F5_U5_assignment_reliability", "F5_assignment_reliability"),
    ("6", "F6_time_thinning", "plot_figure6.py", "F6_nonoverlapping_window_sensitivity", "F6_nonoverlapping_window_sensitivity"),
    ("7", "F7_directional_miscoverage", "plot_figure_f7.py", "Figure_F7_Directional_Miscoverage", "Figure_F7_Directional_Miscoverage"),
    ("8", "F8_growth_size_associations", "plot_figure_f8.py", "Figure_F8_Growth_Size_Associations", "Figure_F8_Growth_Size_Associations"),
    ("9", "F9_adaptive_CQR_performance", "plot_figure_f9.py", "Figure_F9_Adaptive_CQR_Performance", "Figure_F9_Adaptive_CQR_Performance"),
    ("S1", "S1_growth_associations", "plot_figure_s1.py", "Figure_S1_Exploratory_Growth_Associations", "Figure_S1_Exploratory_Growth_Associations"),
    ("S2", "S2_biological_scale", "plot_figure_s2.py", "Figure_S2_Interval_Width_and_Leaf_Area", "Figure_S2_Interval_Width_and_Leaf_Area"),
    ("S3", "S3_replication_sensitivity", "plot_figure_s3.py", "Figure_S3_Replication_Sensitivity", "Figure_S3_Replication_Sensitivity"),
    ("S4", "S4_growth_incremental_value", "plot_figure_s4.py", "Figure_S4_Growth_Incremental_Value", "Figure_S4_Growth_Incremental_Value"),
]


def route(scripts, inputs=(), expensive=False):
    return {"scripts":scripts,"required_inputs":list(inputs),"expensive":expensive}


WORKFLOWS = {
    "data-preparation": route(["src/E0/dataset_structure_audit.py","src/E0/leaf_trajectory_audit.py"], ["data/raw/PhenoLeaf-TS_DS/01_raw_dataset", "data/raw/PhenoLeaf-TS_DS/02_leaf_labels", "data/raw/PhenoLeaf-TS_DS/leaves_unique_colours.xlsx"], True),
    "windows": route(["src/E1/build_forecasting_manifest.py", "src/E1/validate_genotype_folds.py"], ["experiments/E0/tables/E0_1_image_pairing.csv","experiments/E0/tables/E0_2_leaf_area_trajectories.csv"], True),
    "point-forecasting": route(["src/E1/run_simple_baselines.py","src/E1/analyze_baseline_generalization.py","src/E1/train_ridge_logo.py","src/E1/train_boosting_logo.py"],[MANIFEST],True),
    "conformal": route(["src/E2/prepare_uncertainty_protocol.py","src/E2/run_conformal_baselines.py","src/E2/validate_calibration_sensitivity.py","src/E3/run_normalized_conformal.py","src/E3/run_cqr_baselines.py","src/E3/run_mondrian_conformal.py","src/P0/u8_trajectory_balanced.py"],[MANIFEST],True),
    "u5-export": route(["src/P1/p11_u5_prediction_export.py"],[MANIFEST],True),
    "reliability-diagnostics": route(["src/P3/P3_1_analysis.py","src/P3/P3_2_directional_miscoverage.py","src/P3/P3_3_growth_directional.py","src/P3/P3_4_growth_robustness.py","src/P3/P3_5_size_controlled_growth.py"], [MANIFEST,*PREDICTIONS],True),
    "adaptive-cqr": route(["src/P3/P3_6_adaptive_cqr.py"],[MANIFEST],True),
    "growth-robustness": route(["src/P3/P3_4_growth_robustness.py"],["experiments/P3/P3_3/P3_3_genotype_contrasts.csv"]),
    "adaptive-comparisons": route(["src/P3/P3_7_paired_comparison.py","src/P3/P3_8_growth_incremental_value.py"],["experiments/P3/P3_6/P3_6_genotype_metrics.csv"]),
    "figures": route([f"figures/rebuilt/{folder}/{script}" for _,folder,script,_,_ in FIGURES]),
    "tables": route(["scripts/generate_tables.py"]),
}


def plan(name, root=ROOT):
    spec = WORKFLOWS[name]
    root = Path(root)
    return {"workflow":name,**spec,"missing_inputs":[p for p in spec["required_inputs"] if not (root/p).exists()],
            "execution_status":"See docs/REPRODUCIBILITY.md; planning does not execute or validate model training"}


def run(name, output, *, allow_expensive=False, inputs_from=None, dataset=None):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output must be a new directory; archived results will not be overwritten")
    if output == ROOT or output in ROOT.parents:
        raise ValueError("Output cannot contain the source repository")
    if ROOT in output.parents and ROOT/"results/runs" not in output.parents:
        raise ValueError("Inside the repository, output must be under results/runs to prevent recursive copying")
    spec = WORKFLOWS[name]
    if spec["expensive"] and not allow_expensive:
        raise ValueError("This unvalidated full workflow requires --allow-expensive and external inputs; inspect plan first")
    output.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(ROOT,output,ignore=shutil.ignore_patterns(".git",".venv","__pycache__","runs"))
    if inputs_from:
        origin = Path(inputs_from).resolve()
        for relative in spec["required_inputs"]:
            source, target = origin/relative, output/relative
            if not target.exists() and source.is_file():
                target.parent.mkdir(parents=True,exist_ok=True)
                shutil.copy2(source,target)
    if dataset:
        target = output/"data/raw/PhenoLeaf-TS_DS"
        target.parent.mkdir(parents=True,exist_ok=True)
        if target.exists():
            raise ValueError("Dataset target already exists")
        target.symlink_to(Path(dataset).resolve(),target_is_directory=True)
    missing=plan(name,output)["missing_inputs"]
    if missing:
        raise FileNotFoundError("Missing genuine inputs: "+", ".join(missing))
    report={"workflow":name,"steps":[],"scope":"Original scripts executed in an isolated copy; no original archived artifact replaced"}
    environment={**os.environ,"MPLBACKEND":"Agg","MPLCONFIGDIR":str(output/".mpl-cache"),"OMP_NUM_THREADS":"1","OPENBLAS_NUM_THREADS":"1"}
    for relative in spec["scripts"]:
        command=[sys.executable,str(output/relative)]
        if name=="adaptive-cqr":command.append("--full")
        result=subprocess.run(command,cwd=output,env=environment,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
        log=relative.replace('/','__')+'.txt'
        (output/log).write_text(result.stdout)
        report["steps"].append({"script":relative,"returncode":result.returncode,"log":log})
        (output/"run_report.json").write_text(json.dumps(report,indent=2)+'\n')
        if result.returncode:
            raise RuntimeError(f"{relative} failed; inspect {output/log}")
    if name=="figures":
        report['figures']=[]
        for label,folder,_,stem,final_stem in FIGURES:
            generated=output/f"figures/rebuilt/{folder}/{stem}.png"
            archived=ROOT/("figures/supplementary" if label.startswith('S') else "figures/main")/(final_stem+'.png')
            report['figures'].append({'label':label,'generated':str(generated.relative_to(output)),
                'png_sha256':digest(generated),'archived_png_sha256':digest(archived),
                'png_byte_identical':digest(generated)==digest(archived)})
    (output/"run_report.json").write_text(json.dumps(report,indent=2)+'\n')
    return report
