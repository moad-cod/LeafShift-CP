# Data and code availability

This local candidate contains original software and selected authorized derived
evidence. The intended destination is https://github.com/moad-cod/LeafShift-CP.
The repository exists, but the synchronized candidate's public access must be
verified after the author's manual push.

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

The first author reports that the dataset corresponding author granted sharing,
including GitHub, of aggregate performance metrics/statistical summaries,
plant-level derived observation counts/QC and conformal evaluation results and
numerical figures. Selected files fall within those categories. This is a
first-author confirmation, not an invented copy of private correspondence.
See [the scope record](FINAL_RELEASE_APPROVALS.md).

Raw RGB, masks, archives, palette/source-label fragments, original observation/
window/leaf records and private review payloads remain excluded. Local evidence
is retained; the exact public allowlist prevents its inclusion. Provider
documentation is linked rather than republished.

Figure 1 has first-author-confirmed manuscript permission. Public redistribution
of the composite/source RGB/mask is not confirmed, so Figure 1 and manuscript
PDFs embedding it remain excluded from this repository.

## Reconstruction limits

The exact original archive identity/checksum was not retained, and 260 Ba4-1
RGB files were missing from the audited checkout. Complete extraction/windows/
training remain unvalidated end to end. Genuine compact archives support bounded
arithmetic checks and historical replay receipts, not new fits or independent
prospective validation.

Software MIT licensing is separate from data/figure permission; it does not
remove provider restrictions or make excluded source data publicly available.
