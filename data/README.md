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

Selected metadata and plant-level count/QC/statistical summaries are included
under the reported derived-output permission. Original observation/window/leaf
records, RGB, masks, palette/source-label fragments, archives and private review
payloads remain excluded. See [availability](../docs/DATA_AVAILABILITY.md) and
[permission scope](../docs/FINAL_RELEASE_APPROVALS.md).

Original source archive identity/checksum is unresolved; the audited source
lacked 260 Ba4-1 RGB files. No dataset was downloaded or replacement data created.
