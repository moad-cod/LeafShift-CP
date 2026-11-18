# Reproducibility scope

## Evidence levels

1. **Byte-verified copies:** the public inventory identifies selected original
   implementations and genuine archived evidence by SHA-256.
2. **Current archive verification:** the verifier recomputes documented aggregates
   in memory and checks split/hierarchy/cohort rules and reference metrics. It
   performs no model fitting, numerical-table export or plot generation.
3. **Historical downstream replay:** prior local records document table
   re-aggregation, selected P3 comparisons and 12 numerical figure replays from
   archived fitted outputs. These were not rerun in finalization; some detailed
   old stdout logs are not bundled.
4. **Untested full reconstruction:** raw extraction, windows, point/conformal
   fitting, U5 export and adaptive refitting require authorized external/generated
   inputs. No complete independent end-to-end training reproduction is claimed.

## Current exact-candidate checks

After installing the Python 3.12 requirements:

```sh
.venv/bin/python scripts/verify.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python scripts/reproduce.py list
.venv/bin/python scripts/reproduce.py plan data-preparation
.venv/bin/python scripts/reproduce.py plan conformal
```

Current outcomes are in [the release report](FINAL_GITHUB_RELEASE_REPORT.md) and
`results/verification/release_validation.json`. The public inventory covers only
selected copies, so integrity-check counts differ from the historical 383-copy
local archive. No numerical target or scientific CSV was changed to make checks pass.

The existing feature-constructor unit fixture is a software test, not synthetic
research evidence. Tests do not open images or train models. Optional
PyArrow-dependent Parquet/Feather inventory branches remain untested.

## Historical replay commands

These genuine workflows were previously validated from archived inputs. They
were **not executed during this finalization**. They write new outputs in an
isolated directory; choose a new output and review permissions before future runs.

```sh
.venv/bin/python scripts/reproduce.py run tables --output results/runs/tables
.venv/bin/python scripts/reproduce.py run growth-robustness --output results/runs/growth
.venv/bin/python scripts/reproduce.py run adaptive-comparisons --output results/runs/adaptive
.venv/bin/python scripts/reproduce.py run figures --output results/runs/figures
```

Table generation re-aggregates four-assignment U0–U8 fold evidence into 18
method/horizon rows. Fold metrics already average windows within plants;
assignments are averaged within genotype, then genotypes equally. U5 coverage
is independently reconstructed from plant-by-assignment summaries. The compact
bundle cannot recompute all width/score values from forecast windows because
row-level predictions are excluded.

Figure 9 and P3.7 preserve different original bootstrap draw orderings for
percentile intervals. Historical PNG equality is not a new visual assessment.
The coarse `plan figures` and `plan tables` routes omit some implicit CSV
requirements: consult their implementations and the [map](FIGURE_TABLE_MAP.csv).

## Full experiments

See [EXTERNAL_DATA_RECIPE.md](EXTERNAL_DATA_RECIPE.md). Expensive routes require
`--allow-expensive` and real inputs. Missing files are not synthesized. Full
execution, cross-platform portability and a complete locked training environment
remain unvalidated; original archive identity and missing RGB provenance remain
limitations. Repeated windows/assignments are not independent biological
replication, and retrospective adaptive comparisons are not prospective validation.
