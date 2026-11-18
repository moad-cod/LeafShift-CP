# External-data execution recipe — not a completed full reproduction

This describes future authorized execution of the unchanged code-only profile.
No dataset was downloaded, images decoded, plot generated or model fitted to
validate this recipe. Use only an authorized copy of PhenoLeaf-TS. The exact
historical source archive and missing RGB provenance remain unresolved.

The dataset path supplied below must contain `01_raw_dataset/`,
`02_leaf_labels/` and `leaves_unique_colours.xlsx`. The runner links it inside a
new isolated output root. Raw source inputs remain outside the release/Git tree.
Each output path must be new. Subsequent stages copy explicitly needed generated
files from the preceding stage using `--inputs-from`.

First inspect plans, which perform no processing or training:

```sh
.venv/bin/python scripts/reproduce.py plan data-preparation
.venv/bin/python scripts/reproduce.py plan windows
.venv/bin/python scripts/reproduce.py plan point-forecasting
.venv/bin/python scripts/reproduce.py plan conformal
```

Only for a separately authorized future run, after reviewing original scripts:

```sh
.venv/bin/python scripts/reproduce.py run data-preparation --dataset /path/to/authorized/PhenoLeaf-TS_DS --output results/runs/preprocessing --allow-expensive
.venv/bin/python scripts/reproduce.py run windows --inputs-from results/runs/preprocessing --output results/runs/windows --allow-expensive
.venv/bin/python scripts/reproduce.py run point-forecasting --inputs-from results/runs/windows --output results/runs/point-models --allow-expensive
.venv/bin/python scripts/reproduce.py run conformal --inputs-from results/runs/windows --output results/runs/conformal --allow-expensive
.venv/bin/python scripts/reproduce.py run u5-export --inputs-from results/runs/conformal --output results/runs/u5-export --allow-expensive
.venv/bin/python scripts/reproduce.py run reliability-diagnostics --inputs-from results/runs/u5-export --output results/runs/diagnostics --allow-expensive
.venv/bin/python scripts/reproduce.py run adaptive-cqr --inputs-from results/runs/conformal --output results/runs/adaptive --allow-expensive
```

These are unvalidated full-run instructions, not a promise of a successful
reconstruction. Original scripts have reference-count checks, implicit
intermediate-file dependencies and output side effects. Actual source version,
missing inputs and intermediate stage needs must be resolved on an authorized
run. Generated observations, predictions, logs and dataset-dependent figures
remain private until their own distribution rights are established.

## Archived and aggregate-only workflows

`scripts/verify.py` needs the original provenance inventory, manifest metadata,
retained-observation counts, real fold/plant/genotype results and paper tables.
`scripts/generate_tables.py` needs E2/E3/U8 fold metrics plus genuine point and
adaptive summaries. These scripts are real software in the code-only profile,
but executing them requires their specified genuine inputs. No expected-value
substitute or fabricated fixture is provided.

`growth-robustness` needs P3.3 genotype contrasts; `adaptive-comparisons` needs
P3.6 genotype metrics. Plotting implementations need their corresponding numeric
CSVs and sometimes per-plant summaries. They can run without raw RGB/masks when
those genuine inputs are separately available and authorized. They were not run
here. In particular, `plan figures` and `plan tables` currently list no required
inputs even though the implementations read CSVs: inspect the software-dependency
record and implementation, not only this coarse route-level preflight.

Selected derived tables and numerical figures are included under the confirmed permission scope in FINAL_RELEASE_APPROVALS.md. The authorized manuscript Figure 1 is a copied publication asset, not a regenerated result. Original observation exports and separate raw dataset files remain excluded. Verification uses genuine archives; missing inputs are never manufactured to satisfy a test.
