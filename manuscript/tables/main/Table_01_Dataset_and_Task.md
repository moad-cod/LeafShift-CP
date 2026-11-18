# Table 01 Dataset and Task

| Characteristic | Value | Unit_or_statistical_level | Source |
|---|---|---|---|
| Arabidopsis genotypes | 21 | genotype | E0 / forecast manifest |
| Plants | 318 | plant | E0 / forecast manifest |
| Paired top-view image/mask frames | 17082 | image pair | E0_1_image_pairing.csv |
| Forecast windows: 24 h | 121431 | window | forecast_windows.csv |
| Forecast windows: 48 h | 107214 | window | forecast_windows.csv |
| Historical observations per forecast | 6 | observations | manifest protocol |
| Outer LOGO test groups | 21 | genotype | E1_1_genotype_folds.json |
| Inner assignments per outer group | 4 | paired assignment | E1_1_genotype_folds.json |
| Fit / calibration / test genotypes | 15 / 5 / 1 | genotypes | E1_1_genotype_folds.json |

Windows overlap within leaf trajectories and are not independent biological replicates.
