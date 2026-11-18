# LeafShift-CP

**Reliability of Conformal Prediction for Individual-Leaf Area Forecasting Under Held-Out *Arabidopsis* Genotypes**

Research software and evidence for evaluating whether prediction intervals remain
reliable when an entire plant genotype is absent from model fitting and conformal
calibration. The study combines individual-leaf forecasting with genotype-resolved
uncertainty evaluation, failure diagnostics and retrospective adaptive calibration.

[Key findings](#key-experimental-findings) ·
[Research evidence](#where-to-find-the-research-evidence) ·
[Quick start](#quick-start-installation-and-verification) ·
[Figures](#figure-and-table-evidence-map) ·
[Dataset access](#dataset-provenance-and-access)

## Authors

| Author | Manuscript role |
|---|---|
| Mouad el Baz | First author |
| Sahil Khan | Second author and corresponding author |
| Qasim Khan | Third author |

## Research question and motivation

**Can prediction intervals with approximately 90% average coverage still fail for
specific *Arabidopsis* genotypes excluded from both fitting and calibration?**

Longitudinal plant phenotyping follows individual leaves through time. Forecasts
can help characterize subsequent development, but their uncertainty must remain
informative across genetic backgrounds. An interval that looks well calibrated
on average may be unreliable for a particular unseen genotype. This matters for
individual-leaf growth forecasting, uncertainty quantification and reproducible
agricultural machine learning under genotype shift.

The response is **segmentation-derived projected visible leaf area, measured in
px²**. It is an image-derived measurement influenced by visibility and
segmentation, rather than calibrated physical leaf surface area or biomass.

## Principal contributions

1. **Genotype-disjoint reliability evaluation.** Evaluate 24- and 48-hour
   individual-leaf prediction intervals across 21 held-out genotypes using
   plant-balanced, assignment-averaged and genotype-macro metrics, separating
   aggregate calibration from genotype-specific reliability.
2. **Diagnosis of heterogeneous uncertainty failures.** Examine genotype-specific
   coverage, plant replication, development-assignment sensitivity, temporal
   overlap, directional misses and historical leaf-area dynamics. These
   diagnostics characterize failures without establishing biological causation.
3. **Retrospective adaptive conformal calibration.** Compare balanced CQR (U5)
   with size-adaptive CQR (SA-CQR) and size-and-growth-adaptive CQR (SGA-CQR) under
   paired configurations. These are exploratory comparisons informed by earlier
   outer-test diagnostics, without independent prospective validation.

The contribution is an empirical reliability framework. No new finite-sample
coverage theorem or independently validated forecasting architecture is claimed.

## Experimental protocol

![LeafShift-CP experimental workflow](figures/main/F1_workflow.png)

*Figure 1 — Dataset and experimental workflow.* Original PhenoLeaf-TS RGB imagery
and leaf-instance masks illustrate projected visible-area extraction. Six
historical observations support 24- and 48-hour forecasts. Leave-one-genotype-out
evaluation separates 15 fitting, five calibration and one excluded test genotype.
The unchanged manuscript figure is included with permission;
[provider attribution](#dataset-provenance-and-access) and
[permission scope](docs/DATA_AVAILABILITY.md) apply.

## Key experimental findings

| Benchmark component | Scope |
|---|---|
| *Arabidopsis* genotypes | 21 |
| Plants | 318 |
| Derived forecasting windows | 228,645 |
| Forecasting horizons | 24 and 48 hours |
| Original conformal procedures | Nine: U0–U8 |
| Development assignments | Four paired assignments per held-out genotype |

| Metric | 24 h | 48 h |
|---|---:|---:|
| HGB genotype-macro MAE (px²) | 309.92 | 518.51 |
| U5 genotype-macro coverage | 89.97120% | 90.04006% |
| Lowest observed genotype coverage | 82.217% | 76.393% |
| Genotypes below 90% coverage | 8/21 | 6/21 |
| SGA-CQR score reduction vs U5 (px²) | 9.049 | 24.133 |

HGB denotes histogram gradient boosting. Point-forecast MAE comes from the
point-model benchmark; coverage and interval scores follow the conformal
evaluation protocol. Values are preserved archived results, not newly fitted
results from repository preparation.

**Near-nominal average coverage conceals substantial genotype-specific
undercoverage.** The SGA-CQR score reductions are retrospective empirical
improvements; worst-genotype undercoverage remains unresolved.
See the [point-forecast table](results/paper_tables/main/Table_02_Point_Forecasting_Baselines.csv),
[conformal table](results/paper_tables/main/Table_03_Conformal_Performance.csv) and
[paired adaptive comparisons](experiments/P3/P3_7/P3_7_paired_summary.csv).

## Experimental design and conformal methods

### Forecasting and evaluation

PhenoLeaf-TS supplies persistent leaf-instance measurements. Each forecast uses
six preceding retained leaf-area observations. The 29 historical predictors
combine areas, relative observation times, adjacent area changes and rates, and
seven summaries. Identifiers and future targets are excluded from the predictors.

For each held-out genotype, the other 20 genotypes are partitioned into **15
fitting and five calibration genotypes**, using four paired development
assignments. The excluded genotype supplies test outcomes at 24 and 48 hours.
The same held-out test plants and windows are reused across those assignments.

Evaluation averages windows within plants, gives plants equal influence within
genotype, averages assignments within genotype, then weights the 21 genotypes
equally. The **228,645 windows are repeated prediction opportunities, not 228,645
independent biological observations**. Overlapping histories and shared plants
create dependence, and assignments are not independent biological replicates.

### Original procedures U0–U8

| Procedure | Score or construction | Calibration aggregation |
|---|---|---|
| U0 | Absolute residuals | Pooled windows |
| U1 | Absolute residuals | Genotype–plant balanced |
| U2 | Residuals normalized by historical size | Pooled windows |
| U3 | Residuals normalized by historical size | Genotype–plant balanced |
| U4 | Conformalized quantile regression (CQR) | Pooled windows |
| U5 | Conformalized quantile regression (CQR) | Genotype–plant balanced |
| U6 | Absolute residuals within historical-size strata | Pooled within each stratum |
| U7 | Absolute residuals within historical-size strata | Balanced within each stratum |
| U8 | Retrospective trajectory-balanced residual sensitivity | Genotype–plant–trajectory balanced |

**Pooled calibration** gives each calibration window equal mass.
**Genotype–plant-balanced calibration** distributes mass equally across genotypes
and then plants, with windows sharing their plant's mass. Its weighted quantiles
are empirical; the weighting does not establish coverage under arbitrary
genotype shift.

Absolute-residual methods calibrate errors around mean predictions.
Historical-size normalization rescales errors using area observed before the
forecast. CQR calibrates lower and upper quantile predictions, using the
implemented nonnegative-expansion convention. Size-Mondrian methods estimate
separate thresholds within historical-size strata. U8 adds equal weighting of
leaf trajectories within plants as a retrospective calibration sensitivity.
Comparisons across residual and CQR families compare complete pipelines with
different fitted objectives.

### Retrospective adaptive extensions

**SA-CQR** scales the CQR correction using historical leaf size.
**SGA-CQR** additionally uses historical area-change dynamics. Both retain the
base quantile-model comparison under matched fitting, calibration and test
assignments. They differ from size-Mondrian residual calibration and are not
additional members of U0–U8. Their exploratory, post-diagnostic evaluation
requires a subsequent independent assessment before claims of prospective
superiority.

## About this public research release

LeafShift-CP is the curated public research release accompanying this study on
conformal prediction reliability under genotype shift. The research was developed
within a larger private working repository containing intermediate experiments,
development records, original dataset-processing materials and internal
validation artifacts.

This public distribution brings together the original research implementations,
finalized experimental configurations, authorized derived evidence, publication
figures, verification tools and documentation relevant to the reported findings.
Scientific implementations and archived numerical results have been preserved.

Internal development materials, duplicate intermediate artifacts, restricted
source data and private records are excluded to keep the release focused.
The public repository does not contain every intermediate experiment or the
complete development history. Curation did not change the scientific methods or
results; the private workspace is not a separate published research source.

## Repository navigation

### Where to find the research evidence

| Location | What reviewers can inspect |
|---|---|
| [src/](src/) | Original scientific implementations and verification/workflow modules |
| [configs/](configs/) | Genotype assignments and uncertainty protocol |
| [experiments/](experiments/) | Authorized archived experiment metrics and statistical summaries |
| [results/](results/) | Paper tables, numerical evidence and validation receipt |
| [figures/main/](figures/main/) | Original manuscript Figures 1–9 |
| [figures/supplementary/](figures/supplementary/) | Publication Figures S1–S4 |
| [figures/rebuilt/](figures/rebuilt/) | Original plotting implementations and numerical inputs |
| [scripts/](scripts/) | Verification, workflow planning and table-generation entry points |
| [tests/](tests/) | Integrity, import and safe workflow-dispatch tests |
| [Figure/table provenance](docs/FIGURE_TABLE_MAP.csv) | Manuscript numbering, assets, implementations and archived inputs |
| [Reproducibility scope](docs/REPRODUCIBILITY.md) | Verification, historical replay and full-training boundaries |
| [External-data recipe](docs/EXTERNAL_DATA_RECIPE.md) | Authorized dataset requirements and future full-run dependencies |

Follow **manuscript result → figure/table → archived numerical evidence →
implementation → verification or reproduction instructions**. The
[source inventory](docs/FILE_INVENTORY.csv) records copied-file hashes; the
[public manifest](docs/PUBLIC_UPLOAD_MANIFEST.csv) records the release's paths,
sizes and hashes.

## Quick start: installation and verification

Use Python 3.12 and `uv`. Run from the repository root:

```bash
uv venv .venv --python python3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/verify.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/reproduce.py list
.venv/bin/python scripts/reproduce.py plan conformal
```

`uv venv` creates the environment; `uv pip install` installs the recorded
dependency versions in [requirements.txt](requirements.txt).
`verify.py` checks archived arithmetic, protocol rules and source-copy hashes;
`unittest` checks integrity, scientific imports, source syntax and safe dispatch.
`list` prints available workflows; `plan conformal` reports inputs and execution
requirements without fitting models.

Lightweight validation passes **9 tests and 474 scientific/integrity checks**.
These checks do not constitute complete experimental reproduction.

## How to inspect or reproduce the findings

| Level | Available work | Evidence and boundary |
|---|---|---|
| **1 — Repository verification** | Run the quick-start checks and inspect compact archived tables. | Current validation checks 275 original copies and 199 scientific/protocol quantities; imports and safe dispatch are tested. |
| **2 — Archived evidence inspection and replay** | Follow table/figure provenance and supported `tables`, `growth-robustness`, `adaptive-comparisons` or `figures` workflows. | Historical records cover table re-aggregation, selected P3 comparisons and 12 numerical figure replays from genuine archived inputs. These replays were not rerun for this README update. |
| **3 — Full experimental reproduction** | Acquire authorized source data, reconstruct historical processing/windows, fit models and calibrate/evaluate intervals. | Requires external files, generated intermediates and substantially more computation than archive checks. Full end-to-end extraction and model training were not independently reproduced in public-release validation. |

Replay requires the real inputs specified by each implementation; coarse workflow
plans do not enumerate every CSV dependency. Some detailed historical stdout
records are not bundled. Figure 1 is an unchanged manuscript asset, not an output
of the numerical plotting workflows.

Use [REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) for replay commands and evidence
levels, and [EXTERNAL_DATA_RECIPE.md](docs/EXTERNAL_DATA_RECIPE.md) for full-run
inputs, explicit execution gates and unresolved reconstruction requirements.
Historical training cost and complete environment portability have not been
validated here. Missing inputs are not replaced with fabricated data.

## Figure and table evidence map

| Manuscript figure | Publication asset |
|---|---|
| Figure 1 — Dataset and experimental workflow | [Figure 1](figures/main/F1_workflow.png) |
| Figure 2 — Point forecasting performance | [Figure 2](figures/main/F2_point_forecasting.png) |
| Figure 3 — Coverage–efficiency comparison | [Figure 3](figures/main/F3_coverage_efficiency.png) |
| Figure 4 — Genotype-specific reliability | [Figure 4](figures/main/F4_genotype_reliability.png) |
| Figure 5 — Development-assignment sensitivity | [Figure 5](figures/main/F5_assignment_reliability.png) |
| Figure 6 — Temporal thinning | [Figure 6](figures/main/F6_nonoverlapping_window_sensitivity.png) |
| Figure 7 — Directional miscoverage | [Figure 7](figures/main/Figure_F7_Directional_Miscoverage.png) |
| Figure 8 — Historical growth and size associations | [Figure 8](figures/main/Figure_F8_Growth_Size_Associations.png) |
| Figure 9 — Adaptive CQR comparison | [Figure 9](figures/main/Figure_F9_Adaptive_CQR_Performance.png) |
| Figure S1 — Exploratory growth associations | [Figure S1](figures/supplementary/Figure_S1_Exploratory_Growth_Associations.png) |
| Figure S2 — Interval width and biological scale | [Figure S2](figures/supplementary/Figure_S2_Interval_Width_and_Leaf_Area.png) |
| Figure S3 — Replication sensitivity | [Figure S3](figures/supplementary/Figure_S3_Replication_Sensitivity.png) |
| Figure S4 — Incremental value of growth adaptation | [Figure S4](figures/supplementary/Figure_S4_Growth_Incremental_Value.png) |

| Manuscript table | Archived numerical evidence |
|---|---|
| Table 1 — Point forecasting | [Point-model metrics](results/paper_tables/main/Table_02_Point_Forecasting_Baselines.csv) |
| Table 2 — Conformal procedures | [U0–U8 performance](results/paper_tables/main/Table_03_Conformal_Performance.csv) |
| Table 3 — Adaptive CQR | [Adaptive method summary](experiments/P3/P3_6/P3_6_method_summary.csv) |

Export filenames retain their original research numbering. The
[figure/table map](docs/FIGURE_TABLE_MAP.csv) uses final manuscript numbering and
identifies implementations and input dependencies.

## Dataset provenance and access

Experiments use **PhenoLeaf-TS**. Obtain original data through the
[official dataset provider](https://huggingface.co/datasets/rick77a/PhenoLeaf_TS)
under its governing terms; see [dataset acquisition](data/README.md).
The [additional provider resource](https://arabidopsis-rgb.latrobe.edu.au) is
context/access information, not another established experimental input.

Both provider-requested publications must be cited:

1. Sarić, Azam, Khan and Čustović (2026). *PhenoLeaf-TS: A Time-Series Benchmark
   for Leaf Instance Segmentation, Tracking, and Growth Stage Classification*.
   ECCV 2026. [DOI: 10.1007/978-3-032-37359-5_23](https://doi.org/10.1007/978-3-032-37359-5_23).
2. Sarić, Kashkan, Lewsey, Panzarová, Trtílek, Čustović and Whelan (2026).
   *ALeRT: An RGB-based deep learning pipeline and benchmark for automated
   phenotyping of rosette and leaf traits in Arabidopsis thaliana*.
   Computers and Electronics in Agriculture.
   [DOI: 10.1016/j.compag.2026.112620](https://doi.org/10.1016/j.compag.2026.112620).

ALeRT provides RGB phenotyping context. This study does not assert experimental
use of its dataset or implementation.

## Reproducibility boundaries and limitations

- **Validity:** empirical weighted quantiles do not guarantee coverage under
  arbitrary genotype shift.
- **Adaptive-method selection:** SA-CQR and SGA-CQR were evaluated retrospectively
  after earlier outer-test diagnostics.
- **Biological replication:** eleven genotypes had fewer than eight plants.
- **Dependence:** forecast histories/windows overlap; leaves share plants and
  development assignments reuse held-out test outcomes.
- **Measurement:** visible leaf area is segmentation-derived and can reflect
  orientation, occlusion and annotation effects as well as growth.
- **Dataset identity:** the exact original source archive version/checksum was
  not retained.
- **Historical inventory:** the audited source lacked 260 Ba4-1 RGB files.
- **Full reconstruction:** end-to-end extraction and model training have not
  been independently reproduced as part of final public-release validation.

Detailed lineage, optional dependency and bootstrap/replay caveats remain in
the [reproducibility documentation](docs/REPRODUCIBILITY.md).

## Permissions, licensing and citation

Mouad el Baz, the first author, obtained permission from the PhenoLeaf-TS dataset
corresponding author, Dr. Rijad Sarić, for the use of the specified research
materials and derived outputs. The original dataset remains available from its
official provider.

This release includes the authorized original manuscript Figure 1 and numerical
research outputs, including aggregate statistics, plant-level count/QC summaries
and conformal evaluation results. This specific permission does not grant
unrestricted redistribution of PhenoLeaf-TS. Original source RGB archives,
segmentation masks and detailed observation/window/leaf exports remain excluded.
Permission correspondence is private and is not included. See
[data availability](docs/DATA_AVAILABILITY.md),
[authorization scope](docs/FINAL_RELEASE_APPROVALS.md) and
[public exclusions](docs/PUBLIC_EXCLUSIONS.md).

Original software is licensed under [MIT](LICENSE):
`Copyright (c) 2026 Mouad el Baz, Sahil Khan`.
The software license does not automatically cover the dataset, source imagery,
masks, derived-data files, rendered figures or third-party dependencies.
Scientific authorship is separate from software copyright attribution.

Use [CITATION.cff](CITATION.cff), which preserves all three scientific authors in
order, and identify the code version used. The companion manuscript awaits
submission; no paper acceptance, journal publication or publication DOI is
claimed.
