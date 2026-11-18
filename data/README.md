# Dataset acquisition

Obtain authorized PhenoLeaf-TS data from the
[provider dataset card](https://huggingface.co/datasets/rick77a/PhenoLeaf_TS).
Follow its access instructions, terms and citation requirements. The
[additional provider resource](https://arabidopsis-rgb.latrobe.edu.au) is listed
for context/access, not as an established experimental input.

Cite both requested works:

- [PhenoLeaf-TS, ECCV 2026 — 10.1007/978-3-032-37359-5_23](https://doi.org/10.1007/978-3-032-37359-5_23).
- [ALeRT, Computers and Electronics in Agriculture, 2026 — 10.1016/j.compag.2026.112620](https://doi.org/10.1016/j.compag.2026.112620).

The unchanged extraction code expects an authorized future local dataset with:

```text
data/raw/PhenoLeaf-TS_DS/
  01_raw_dataset/
  02_leaf_labels/
  leaves_unique_colours.xlsx
```

This tree is not bundled or upload-eligible. The runner can link an external
authorized directory into an isolated run; see
[the recipe](../docs/EXTERNAL_DATA_RECIPE.md).

Mouad el Baz, the first author, obtained permission from the PhenoLeaf-TS dataset
corresponding author, Dr. Rijad Sarić, for the use of the specified research
materials and derived outputs. The original dataset remains available from its
official provider.

The specified aggregate/statistical, plant-count/QC and conformal outputs are
included in this public repository, together with the unchanged manuscript
[Figure 1](../figures/main/F1_workflow.png), as authorized by the first author's
confirmation. Cite both required publications above. Figure 1 combines
PhenoLeaf-TS imagery with our experimental workflow; its inclusion does not
permit unrestricted redistribution of original source images or masks.

Original observation/window/leaf records, RGB datasets, segmentation masks,
palette/source-label fragments, archives and private review payloads remain
excluded. Obtain original source images and masks through the dataset provider.
See [availability](../docs/DATA_AVAILABILITY.md) and
[permission scope](../docs/FINAL_RELEASE_APPROVALS.md).

Original source archive identity/checksum is unresolved; the audited source
lacked 260 Ba4-1 RGB files. No dataset was downloaded or replacement data created.
