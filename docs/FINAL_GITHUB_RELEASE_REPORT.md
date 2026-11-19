# Public research release and Figure 1 validation

## Published software archive — 19 November 2026

The published [GitHub release v0.1.1](https://github.com/moad-cod/LeafShift-CP/releases/tag/v0.1.1)
is archived at [Zenodo DOI 10.5281/zenodo.24370590](https://doi.org/10.5281/zenodo.24370590).
The Zenodo record and publisher-deposited DataCite metadata were checked: the
DOI is findable, the version is v0.1.1, and the title, three-author order, both
ORCIDs and MIT software license agree. This software DOI is separate from any
future journal article DOI. The archived tag is unchanged; DOI links added to
main are post-release metadata/documentation updates, not a new release.
Historical preparation and approval records below remain preserved.

## Version 0.1.1 preparation — 19 November 2026

Current main prepares software version `0.1.1`: the CFF and runtime version
identifiers agree. Author order, both confirmed ORCIDs, corresponding-author
details, MIT licensing and scientific content are preserved. Current hashes
are recorded in `PUBLIC_UPLOAD_MANIFEST.csv`.

The existing `v0.1.0` tag and release remain unchanged; that tag retains the
earlier invalid nested CFF author structure. The author will create `v0.1.1`
after reviewing the corrected metadata. This preparation creates no tag,
GitHub release or Zenodo DOI. The earlier records below remain historical.

## Earlier metadata clarification — 19 November 2026

Current author order: **Mouad el Baz**, **Sahil Khan**, **Qasim Khan**. Mouad is
the first and sole corresponding author, Independent Researcher, Morocco;
[elbaz.ouad1249@gmail.com](mailto:elbaz.ouad1249@gmail.com),
[ORCID 0009-0005-0640-4166](https://orcid.org/0009-0005-0640-4166).
All three authors' approval of this arrangement and the research release is
author-reported confirmed; no signatures or approval documents are invented.

README contact information and the CFF author-list structure are corrected.
The CFF retains all three authors, Mouad's supplied ORCID, software version
`0.1.0` and MIT identifier. No article or Zenodo DOI is invented. The exact
current file membership and hashes are in `PUBLIC_UPLOAD_FILES.txt` and
`PUBLIC_UPLOAD_MANIFEST.csv`. Scientific source, configurations, data, figures,
results and the existing MIT copyright holders remain unchanged.

Validation for this metadata update: **474/474** existing archived-evidence and
copy-integrity checks and **9/9** existing unit tests pass. All **49** local
README links resolve to tracked public paths. Unique-key YAML parsing and the
official CFF 1.2.0 schema pass; Mouad's ORCID format and MOD 11-2 checksum are
valid. The ORCID is author-supplied; its ownership was not independently
authenticated. Full end-to-end training was not rerun or newly claimed.

The existing `v0.1.0` tag points to commit
`e0cd7438be990d95491afc28e48b00dcaad25dcf`, before this metadata correction.
Updating main does not change that tagged snapshot. No tag or GitHub release is
created, deleted, moved or edited by this update. A subsequent archive of the
corrected snapshot will require an explicit release/version decision; retain
the existing tag and do not force-push it. A future release description should
identify Mouad as first and sole corresponding author and preserve Sahil and
Qasim as second and third authors, respectively.

At this check, the GitHub release-by-tag endpoint returned 404 and the accessible
release list returned no published or draft entries. This records the state
visible to the connection, not proof about inaccessible private drafts. The
existing tag remains present. If `v0.1.0` is used for a future release without
moving that tag, it will still archive the earlier metadata snapshot; a new
tag/version is the normal way to publish the corrected files.

The dated report below is preserved as historical validation evidence, including
its earlier corresponding-author arrangement and file-size/test receipts.
Those historical statements do not override the current clarification above.

## Historical validation snapshot — 18 November 2026

18 November 2026. The research repository is published at
[moad-cod/LeafShift-CP](https://github.com/moad-cod/LeafShift-CP).
The authorized Figure 1 and permission-documentation update passed the checks
below; the original 87 import commits are retained.
The current GitHub main branch identifies the published version.

## Reviewer-oriented README update

The root README now presents the research question, three contributions, verified
headline results, experimental design, U0–U8 methods, curated-release provenance,
reviewer navigation and three reproducibility levels. The existing authorized
Figure 1 is embedded without changing its bytes; all 13 figure links and the
three manuscript table evidence links are checked.

Current lightweight checks pass: 9/9 tests and 474/474 scientific/copy checks.
Headline values match the genuine archived point, conformal and paired adaptive
tables. No experiment or figure replay was rerun. This documentation update
changes only README.md, this release summary and PUBLIC_UPLOAD_MANIFEST.csv.
It adds no public files and changes no scientific source, configuration, result,
figure, author/license metadata, manuscript or CEA submission material.

## Release contents

- **302 files, 16388291 bytes (15.63 MiB)**.
- Exact membership: [PUBLIC_UPLOAD_FILES.txt](PUBLIC_UPLOAD_FILES.txt).
- Current sizes/hashes: [PUBLIC_UPLOAD_MANIFEST.csv](PUBLIC_UPLOAD_MANIFEST.csv).
- Original software, configurations, authorized aggregate/conformal and plant-
  count/QC evidence, paper tables, numerical figures and unchanged Figure 1.
- Only one new public file: [figures/main/F1_workflow.png](../figures/main/F1_workflow.png).
- No separate raw RGB images, segmentation masks, dataset archives, observation/
  window/leaf exports, private correspondence/reviews, editable designs or
  manuscript submission PDFs are added.

## Permission and attribution

Mouad el Baz, the first author, obtained permission from the PhenoLeaf-TS dataset
corresponding author, Dr. Rijad Sarić, for the use of the specified research
materials and derived outputs. The original dataset remains available from its
official provider.

The first author explicitly authorized the exact manuscript Figure 1 in this
public repository. That composite combines PhenoLeaf-TS imagery with the
experimental workflow; it was copied unchanged, not recreated. The authorization
record is the first author's confirmation. No email/private correspondence was
accessed or published for this update.

The specified aggregate/statistical, plant-count/QC and conformal results and
publication figures remain within the confirmed scope. Both required works are
cited in README and dataset documentation:
[PhenoLeaf-TS](https://doi.org/10.1007/978-3-032-37359-5_23) and
[ALeRT](https://doi.org/10.1016/j.compag.2026.112620).
This does not grant unrestricted redistribution of the complete raw dataset or
separate source images/masks; obtain those from the
[official provider](https://huggingface.co/datasets/rick77a/PhenoLeaf_TS).

## Figure and scientific integrity

Figure 1 source: figures/final/F1_dataset_workflow.png, the exact asset referenced
by the original manuscript Methods and the submission's Figure_1.png.
Repository destination: figures/main/F1_workflow.png.

- Bytes: **941,668**.
- SHA-256: **78860f5b016e30f9a96974629ad5f4110a74a45baa04268d4722b9644fb1cbfd**.
- Exact byte/hash match with the manuscript source and submission asset.
- All 36 pre-existing canonical numerical PNG/PDF/SVG files remain unchanged;
  with Figure 1, 37 canonical media files match their original sources.
- The other Figure 2–9, S1–S4 and table mapping rows are unchanged.

Existing scientific code, configurations, numerical outputs and tables are
unchanged. The original manuscript and CEA_Submission are untouched. No image
was decoded, rendered, visually inspected, resized, regenerated or recreated.

## Validation and reproducibility

Existing lightweight tests: **9/9 PASS**.
Scientific and source-copy integrity checks: **474/474 PASS**:
**275** verified original copies plus the same **199** scientific/protocol checks.
The extra check verifies the added Figure 1; no research result changed.
Current receipt: results/verification/release_validation.json.

README/local documentation links resolve to public allowlisted files. The
default-deny ignore rules and upload list allow the exact new Figure 1 file,
without admitting raw dataset directories or local evidence.

Historical downstream replay, archived-only evidence and untested full training
remain distinguished in [REPRODUCIBILITY.md](REPRODUCIBILITY.md). No training,
new numerical table generation or figure replay was run. Archive identity,
missing 260 RGB files, optional PyArrow branches and end-to-end reconstruction
limitations are preserved.

## Authors, citation and software license

Mouad el Baz remains first author, Sahil Khan second/corresponding author and
Qasim Khan third. CITATION.cff author order, title, version and MIT license
metadata are unchanged; only its repository-status abstract was updated.
Root LICENSE and README's software-license/citation section are unchanged.
MIT software licensing does not automatically cover dataset materials or figures.
No own-paper DOI, acceptance date or journal publication was invented.

## Changed files and retained evidence

Documentation/status corrections: README.md, data/README.md,
docs/DATA_AVAILABILITY.md, docs/FINAL_RELEASE_APPROVALS.md,
docs/PUBLIC_EXCLUSIONS.md, docs/EXTERNAL_DATA_RECIPE.md,
docs/REPRODUCIBILITY.md and the status abstract in CITATION.cff.
Figure/provenance/upload records: docs/FIGURE_TABLE_MAP.csv,
docs/FILE_INVENTORY.csv, docs/PUBLIC_UPLOAD_FILES.txt,
docs/PUBLIC_UPLOAD_MANIFEST.csv and .gitignore.
This report and the validation receipt are refreshed; the exact new Figure 1
is the only added upload path. Previous public records are retained privately
under docs/local_release_evidence/, outside the public allowlist.

The earlier ALeRT volume 257 field remains author-provided because DOI metadata
had no volume; this documentation update changes no manuscript references.
Scientific manuscript/journal declarations and approvals are outside this task.

**VALIDATED DOCUMENTATION AND FIGURE 1 UPDATE.** Normal commit/push preserves the
original import history. No force-push, raw dataset redistribution, email access
or journal submission is part of this update.
