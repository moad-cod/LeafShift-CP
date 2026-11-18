# LeafShift-CP

**Reliability of Conformal Prediction for Individual-Leaf Area Forecasting Under
Held-Out Arabidopsis Genotypes**

**Authors:** Mouad el Baz · Sahil Khan · Qasim Khan.
**Corresponding author:** Sahil Khan.

Research software and authorized derived evidence for individual-leaf forecast
uncertainty under held-out genotypes, published at
[moad-cod/LeafShift-CP](https://github.com/moad-cod/LeafShift-CP).
The repository provides the methodology, archived numerical evidence and
publication figures, including the original manuscript's
[Figure 1 workflow](figures/main/F1_workflow.png).

## Scientific question and design

Near-nominal average interval coverage can conceal substantial undercoverage for
individual genotypes. The benchmark forecasts segmentation-derived projected
visible leaf area (px²) at 24 and 48 hours from six preceding observations.
The data represent 21 Arabidopsis genotypes, 318 plants and 228,645 forecasting
windows. An outer genotype is excluded from fitting and calibration; four paired
assignments divide the remaining genotypes into 15 fitting and five calibration
groups. Assignments share test plants/windows.

Evaluation averages windows within plants, plants within genotype, and assignments
and genotypes explicitly. Retrospective analyses examine replication, overlapping
windows, directional misses, historical dynamics and adaptive CQR. Weighted
empirical calibration is not a proof of coverage under arbitrary genotype shift,
and retrospective comparisons are not independent prospective validation.

## Dataset provenance and required citations

Experiments use **PhenoLeaf-TS**. Cite both provider-requested publications:

1. Sarić, Azam, Khan and Čustović (2026). *PhenoLeaf-TS: A Time-Series Benchmark
   for Leaf Instance Segmentation, Tracking, and Growth Stage Classification*.
   ECCV 2026. [DOI: 10.1007/978-3-032-37359-5_23](https://doi.org/10.1007/978-3-032-37359-5_23).
2. Sarić, Kashkan, Lewsey, Panzarová, Trtílek, Čustović and Whelan (2026).
   *ALeRT: An RGB-based deep learning pipeline and benchmark for automated
   phenotyping of rosette and leaf traits in Arabidopsis thaliana*.
   Computers and Electronics in Agriculture.
   [DOI: 10.1016/j.compag.2026.112620](https://doi.org/10.1016/j.compag.2026.112620).

ALeRT supplies RGB phenotyping context; no use of its dataset or implementation
in these experiments is asserted. Obtain source data from the
[PhenoLeaf-TS provider](https://huggingface.co/datasets/rick77a/PhenoLeaf_TS).
The [additional provider resource](https://arabidopsis-rgb.latrobe.edu.au) is
context/access information, not another established experimental input.
See [dataset access](data/README.md) and [availability](docs/DATA_AVAILABILITY.md).

## Conformal procedures

| Procedure | Construction | Calibration aggregation |
|---|---|---|
| U0 | Absolute residuals | Pooled windows |
| U1 | Absolute residuals | Genotype–plant balanced |
| U2 | Normalized residuals | Pooled windows |
| U3 | Normalized residuals | Genotype–plant balanced |
| U4 | Conformalized quantile regression | Pooled windows |
| U5 | Conformalized quantile regression | Genotype–plant balanced |
| U6 | Leaf-size Mondrian residuals | Pooled within strata |
| U7 | Leaf-size Mondrian residuals | Balanced within strata |
| U8 | Retrospective trajectory-balanced residual sensitivity | Genotype–plant–trajectory balanced |

Archived U5 macro coverages are 89.97120% and 90.04006%; worst observed genotype
coverages are 82.217% and 76.393%. These are genuine archived values, not results
from new fitting during preparation. Point-model selection/refitting and
conformal fitting are distinct protocols.

## Directory guide

```text
configs/                  Frozen genotype membership and uncertainty protocol
src/E0/ ... src/P3/        Original research implementations
src/leafshift_cp/          Archive verification and isolated workflow dispatch
scripts/                  Verification, plans and table-generation entry points
tests/                    Lightweight integrity, import and dispatch tests
data/                     Access instructions and permitted derived count metadata
experiments/              Authorized compact metrics/QC at original working paths
results/                  Paper tables, archived summaries and current check receipt
figures/main/             Manuscript Figure 1 and numerical Figures 2–9
figures/supplementary/     Numerical Figures S1–S4
figures/rebuilt/           Original plotting code and approved numeric inputs
manuscript/tables/        Canonical table dependencies; no manuscript PDF
docs/                     Reproducibility, permission scope and figure/table map
```

## Installation and verified checks

Python 3.12, using the pinned requirements:

```sh
uv venv .venv --python python3.12
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python scripts/verify.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/reproduce.py list
.venv/bin/python scripts/reproduce.py plan conformal
```

The verifier recomputes documented aggregates in memory and checks copy hashes,
split independence, cohort counts and reference metrics. It does not retrain
models, generate replacement scientific data or regenerate figures. Current
exact-candidate outcomes are in [the release report](docs/FINAL_GITHUB_RELEASE_REPORT.md).

## Reproduction and evidence mapping

[FIGURE_TABLE_MAP.csv](docs/FIGURE_TABLE_MAP.csv) identifies final manuscript
items, implementations and genuine archived inputs. The
[source inventory](docs/FILE_INVENTORY.csv) records selected byte-identical copies;
the [upload manifest](docs/PUBLIC_UPLOAD_MANIFEST.csv) identifies actual release files.

Previous records document downstream table, diagnostic and figure replay.
Those runs were not repeated in this finalization.
[REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) separates current archive arithmetic,
historical replay and untested full training. Full extraction, window construction,
fitting and calibration require authorized external/generated inputs; see
[the external-data recipe](docs/EXTERNAL_DATA_RECIPE.md).

## Availability and limitations

Mouad el Baz, the first author, obtained permission from the PhenoLeaf-TS dataset
corresponding author, Dr. Rijad Sarić, for the use of the specified research
materials and derived outputs. The original dataset remains available from its
official provider.

The specified outputs in this repository include aggregate metrics/statistics,
plant-level derived counts/QC, conformal results, numerical figures and the
unchanged manuscript Figure 1, whose public inclusion is authorized by the first
author's confirmation. Figure 1 combines PhenoLeaf-TS imagery with our experimental
workflow; both required dataset publications are cited above. This permission
does not grant unrestricted redistribution of original RGB datasets, segmentation
masks, raw archives or observation/window/leaf exports. Obtain original source
images and masks from the dataset provider. Private correspondence is not
published; see [permission scope](docs/DATA_AVAILABILITY.md).

The original dataset archive identity/checksum is unresolved; 260 Ba4-1 RGB files
were missing from the audited source. Full end-to-end training has not been
independently reproduced. Installation/archive checks do not establish numerical
portability on every platform. Optional PyArrow inventory branches are untested.
An intermediate observation summary differs from the final retained count; the
unchanged lineage record explains it. Figure 9 and the paired analysis use
different original bootstrap draw orderings. Results and limitations are preserved.

## License and citation

Original software is licensed under [MIT](LICENSE), with
`Copyright (c) 2026 Mouad el Baz, Sahil Khan`, following the holders' confirmed
agreement and licensing authority. Copyright attribution does not change the
scientific authorship/contribution statement.

MIT does not automatically cover PhenoLeaf-TS, imagery/masks, derived-data files,
rendered figures or third-party dependencies. Their rights and scoped permissions
remain separate; see [approval scope](docs/FINAL_RELEASE_APPROVALS.md).
No unrestricted data license is inferred from permission to host derived outputs.

Use [CITATION.cff](CITATION.cff), retaining all three authors. The paper is awaiting
submission; no paper DOI, acceptance date or journal publication is claimed.
