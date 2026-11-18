# Table 02 Point Forecasting Baselines

| Horizon_h | Model_ID | Point_predictor | Genotype_macro_plant_MAE_px2 | Selection_note |
|---|---|---|---|---|
| 24 | B0 | Persistence | 710.07 | No fitted model |
| 24 | B2 | Ridge regression | 327.76 | Nested genotype CV |
| 24 | B3 | Histogram gradient boosting | 309.92 | Nested genotype selection |
| 48 | B0 | Persistence | 1355.39 | No fitted model |
| 48 | B2 | Ridge regression | 586.80 | Nested genotype CV |
| 48 | B3 | Histogram gradient boosting | 518.51 | Nested genotype selection |

MAE is averaged over windows within plant and then equally over plants/genotypes.
