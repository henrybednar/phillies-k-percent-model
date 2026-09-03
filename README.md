# Phillies K% Prediction Model

A pitcher strikeout-rate (K%) forecasting model, originally built as a trial project for a Quantitative Analyst Associate role with the Philadelphia Phillies.

## What it does

- **`data_merge.R`** — merges MLB/MiLB Statcast pitcher and hitter data (2015-2024), ball/strike counts, and prior-season K% into a single modeling dataset.
- **`k_percentage_model.R`** — predicts each pitcher's next-season K% using an ensemble of empirical Bayes shrinkage, elastic net, and gradient boosting across five pitcher-population buckets (starters, relievers, rookies, etc.), with per-bucket bias correction and a plate-appearance-adaptive blend.
- **`k_2026_2025_predictions.csv`** — sample output: predicted vs. actual K% for the 2025 validation season.

No feature derived from the target season is ever used as a predictive input — all trailing/lag stats are shifted forward to avoid leakage.

## Data

Built on Baseball Savant Statcast data (pitcher and hitter, 2015-2024) and MiLB Statcast data (2021-2024). Raw source CSVs are not included in this repo (large, publicly re-derivable) — pull them from [Baseball Savant](https://baseballsavant.mlb.com/) if you want to reproduce the pipeline end to end.

## Tech stack

R, `tidyverse`, `glmnet`, `gbm`.
