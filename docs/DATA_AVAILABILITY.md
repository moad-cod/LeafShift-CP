# Data and code availability

Research software, authorized research materials and selected derived evidence
are publicly available at https://github.com/moad-cod/LeafShift-CP.

## Source and citations

Experiments use PhenoLeaf-TS. Cite both provider-requested works:

- Sarić, Azam, Khan and Čustović (2026), *PhenoLeaf-TS: A Time-Series Benchmark
  for Leaf Instance Segmentation, Tracking, and Growth Stage Classification*.
  [ECCV DOI](https://doi.org/10.1007/978-3-032-37359-5_23).
- Sarić, Kashkan, Lewsey, Panzarová, Trtílek, Čustović and Whelan (2026),
  *ALeRT: An RGB-based deep learning pipeline and benchmark for automated
  phenotyping of rosette and leaf traits in Arabidopsis thaliana*.
  [ALeRT DOI](https://doi.org/10.1016/j.compag.2026.112620).

Obtain source data from the [provider](https://huggingface.co/datasets/rick77a/PhenoLeaf_TS).
The [additional resource](https://arabidopsis-rgb.latrobe.edu.au) is context/access
information; it is not established as an experimental input.

## Permission scope

Mouad el Baz, the first author, obtained permission from the PhenoLeaf-TS dataset
corresponding author, Dr. Rijad Sarić, for the use of the specified research
materials and derived outputs. The original dataset remains available from its
official provider.

The first author's confirmation authorizes inclusion in this public repository
of the specified aggregate performance metrics/statistical summaries, plant-level
derived observation counts/QC, conformal evaluation results, numerical figures
and the exact original manuscript Figure 1. Both provider-requested publications
above must be cited. The authorization record is the first author's explicit
confirmation and instruction to publish these materials; private correspondence
is not published. See [the scope record](FINAL_RELEASE_APPROVALS.md).

Raw RGB, masks, archives, palette/source-label fragments, original observation/
window/leaf records and private review payloads remain excluded. Local evidence
is retained; the exact public allowlist prevents its inclusion. Provider
documentation is linked rather than republished.

[Figure 1](../figures/main/F1_workflow.png) is included in the manuscript and this
repository. It combines original PhenoLeaf-TS imagery with our experimental
workflow and is copied without modification. Its inclusion does not authorize
redistribution of the complete raw dataset or separate source RGB/mask files.
Original source images and masks remain available through the dataset provider.
The repository also provides the methodology, numerical evidence and other
publication figures. Manuscript submission PDFs and editable figure-design files
are not included.

## Reconstruction limits

The exact original archive identity/checksum was not retained, and 260 Ba4-1
RGB files were missing from the audited checkout. Complete extraction/windows/
training remain unvalidated end to end. Genuine compact archives support bounded
arithmetic checks and historical replay receipts, not new fits or independent
prospective validation.

Software MIT licensing is separate from data/figure permission; it does not
remove provider restrictions or make excluded source data publicly available.
