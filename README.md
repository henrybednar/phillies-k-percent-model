# Phillies K% Prediction Model

A pitcher strikeout-rate (K%) forecasting model, originally built as a trial project for a Quantitative Analyst Associate role with the Philadelphia Phillies. It predicts each pitcher's 2025 K% using only information available before Opening Day 2025, then scores those predictions against the real 2025 season. The full write-up is [`methodology.pdf`](methodology.pdf).

## Results

2025 accuracy for all 873 pitchers, weighted by batters faced (TBF):

| Prediction | RMSE | MAE | R² | Log-loss |
|---|---|---|---|---|
| League average K% from 2024 | 0.0539 | 0.0419 | 0.00 | 0.5297 |
| Pitcher's own recent K% | 0.0512 | 0.0383 | 0.09 | 0.5311 |
| **This model** | **0.0411** | **0.0317** | **0.41** | **0.5261** |

Every prediction also carries an 80% interval, and 80% of actual 2025 values landed inside theirs at every workload level.

## How it works

Each pitcher is routed to one of four groups using only seasons before the one being predicted: established pitchers, newcomers with minor-league data, newcomers without it, and position players who pitched. Established pitchers get a blend of three models that all predict a probability, so every forecast stays between 0 and 1:

- empirical Bayes (beta-binomial) shrinkage of the pitcher's recent K% toward the league average
- a binomial elastic net
- logistic XGBoost

The elastic net's penalty mix and XGBoost's depth, leaf size, and tree count are chosen by pitcher-grouped cross-validation inside each training window. The whole design was tested on a rolling-origin backtest (train through 2022 and predict 2023, then train through 2023 and predict 2024) before 2025 was scored. Each group is then calibrated on its own backtest, and the 80% intervals combine the model's backtested error with binomial sampling noise.

## Files

- `data_merge.R` merges the provided K% file with four Baseball Savant pulls (MLB pitching, MLB pitch outcomes, MLB hitting, and minor-league pitching), all from 2024 or earlier.
- `k_percentage_model.R` builds the features, runs the backtest, fits the final models, and writes the 2025 predictions.
- `methodology.Rmd` is the write-up; `render_methodology.R` knits it to `methodology.pdf`, computing every number from the model's outputs.
- `tests/test_pipeline.R` injects edge cases (0-for-35 and 5-for-5 histories, missing Stuff+ or Age, impossible telemetry, duplicated rows, stray text values) and checks that predictions stay bounded, that bad inputs are rejected, and that the 2025 predictions are calibrated.
- `experiments/` holds the backtest comparisons behind each design choice (`design_decisions.csv` is Table 4 of the methodology).
- `k_2026_2025_predictions.csv` has each pitcher's 2025 prediction, 80% interval, and actual K%.
- `sessionInfo.txt` records the R and package versions used.

## Reproducing

The input data isn't included in this repo. To run the pipeline, put these five files in the repo root:

| File | Contents | Source |
|---|---|---|
| `k_2026.csv` | Season, Name, PlayerId (MLBAM), Team, K%, TBF, Stuff+, Age for 2021-2025 | Provided for the trial (FanGraphs data) |
| `MLBBaseballSavantPitcherData2015-2024.csv` | Pitcher-seasons 2015-2024, minimum 1 IP | [Savant Custom Leaderboard](https://baseballsavant.mlb.com/leaderboard/custom), pitchers |
| `MLBPitchersBallsStrikes2015-2024.csv` | Pitch-outcome counts for the same pitcher-seasons | [Savant Custom Leaderboard](https://baseballsavant.mlb.com/leaderboard/custom), pitchers |
| `MLBBaseballSavantHittersData2015-2024.csv` | Batter-seasons 2015-2024, minimum 25 PA | [Savant Custom Leaderboard](https://baseballsavant.mlb.com/leaderboard/custom), batters |
| `MiLBBaseballSavantPitcherData2021-2024.csv` | Pitcher-seasons 2022-2024, grouped by pitcher and season | [Savant Minor League Statcast Search](https://baseballsavant.mlb.com/statcast-search-minors) |

The minor-league file appears to be Triple-A only: it has no 2021 rows, and Savant's 2021 minor-league data covers Single-A only.

Then, with R 4.5 and the packages `readr`, `dplyr`, `tidyr`, `glmnet`, and `xgboost`:

```
Rscript data_merge.R
Rscript k_percentage_model.R
Rscript tests/test_pipeline.R
Rscript render_methodology.R
```

Knitting the write-up also needs `rmarkdown`, `knitr`, `kableExtra`, `ggplot2`, a LaTeX install such as TinyTeX, and pandoc. The pipeline is deterministic: repeated runs produce identical predictions.

## Citations

The code was developed with assistance from Claude (Anthropic), https://claude.ai, as noted at the top of each script. References for the statistical methods are listed in the methodology and in the header of `k_percentage_model.R`.
