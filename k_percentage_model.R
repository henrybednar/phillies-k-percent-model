# k_percentage_model.R
#
# Developed with assistance from Claude (Anthropic), https://claude.ai
#
# Step 2 of 2. Predicts each pitcher's 2025 K% from k_2026_final_merged.csv
# (written by data_merge.R, which must be run first). Blends
# empirical Bayes shrinkage, a binomial elastic net, and logistic xgboost
# for pitchers with MLB history, with simpler rules for three other pitcher
# groups, and a per-group calibration on top. Every model predicts a
# probability, so every prediction lies inside (0, 1).
#
# No 2025 value except Age is ever used as a predictive input: all
# lag/trailing features are shifted forward, and group routing uses only
# seasons before the one being predicted. 2025 K% and TBF are used only
# to score the predictions (TBF as the accuracy weight).
#
# References for the techniques used (links checked 2026-09-24):
# [1] Empirical Bayes / beta-binomial shrinkage of a rate stat:
#     D. Robinson, "Understanding empirical Bayes estimation (using baseball
#     statistics)", http://varianceexplained.org/r/empirical_bayes_baseball/
# [2] K% stabilization benchmark (70 BF): FanGraphs Library, "Sample Size",
#     https://library.fangraphs.com/principles/sample-size/
# [3] Elastic net for binomial GLMs: Friedman, Hastie & Tibshirani (2010),
#     "Regularization Paths for Generalized Linear Models via Coordinate
#     Descent", J. Stat. Softw. 33(1), https://www.jstatsoft.org/v33/i01/
# [4] xgboost parameters and the binary:logistic objective:
#     https://xgboost.readthedocs.io/en/stable/parameter.html
# [5] Rolling-origin backtest: Hyndman & Athanasopoulos, Forecasting:
#     Principles and Practice (3rd ed.), sec. 5.10,
#     https://otexts.com/fpp3/tscv.html
# [6] Grouped (blocked) cross-validation for repeated-measures data:
#     Roberts et al. (2017), Ecography 40(8), https://doi.org/10.1111/ecog.02881

library(readr)
library(dplyr)
library(tidyr)
library(glmnet)
library(xgboost)

set.seed(2026)


# =============================================================================
# Configuration
# =============================================================================

data_dir <- "."

# z for an 80% two-sided normal interval
interval_z <- qnorm(0.9)

# Velocity/spin for the six most common pitch types.
arsenal_cols <- c(
  "ff_avg_speed", "ff_avg_spin", "sl_avg_speed", "sl_avg_spin",
  "cu_avg_speed", "cu_avg_spin", "ch_avg_speed", "ch_avg_spin",
  "si_avg_speed", "si_avg_spin", "fc_avg_speed", "fc_avg_spin"
)

# K_pct/TBF/StuffPlus/role_ratio need both lag1 and lag2; every rate-stat
# and arsenal column only ever appears as _lag1 in feat_cols.
lag_both_cols <- c("K_pct", "TBF", "StuffPlus", "role_ratio")
lag1_only_cols <- c(
  "bb_percent", "whiff_percent", "oz_swing_percent", "z_swing_miss_percent",
  "hard_hit_percent", "barrel_batted_rate", "edge_percent", "f_strike_percent",
  "groundballs_percent", "fastball_avg_speed", "csw_percent", arsenal_cols
)

# Prior MiLB season stats used by the newcomer translation model (Step 7).
milb_feat_cols <- c(
  "milb_k_percent", "milb_bb_percent", "milb_swing_miss_percent",
  "milb_xwoba", "milb_velocity", "milb_hardhit_percent"
)

# Every column of the merged file that feeds a model, as named on disk.
raw_input_cols <- c("K%", "TBF", "Stuff+", "Age", "Season", "p_game", "p_starting_p",
                    lag1_only_cols, milb_feat_cols)


# =============================================================================
# Utility Functions
# =============================================================================

wrmse <- function(pred, actual, w) sqrt(weighted.mean((pred - actual)^2, w))
wmae <- function(pred, actual, w) weighted.mean(abs(pred - actual), w)

#' Binomial noise floor for a rate stat: sqrt(weighted mean of p*(1-p)/TBF),
#' using the prediction as the stand-in for true rate p.
noise_floor_rmse <- function(pred, w) sqrt(weighted.mean(pred * (1 - pred) / w, w))

#' Achieved RMSE vs. its noise floor, plus the model-attributable "excess"
#' (assumes independent error sources: achieved^2 ~= floor^2 + excess^2).
report_noise_floor <- function(pred, actual, w, label) {
  achieved <- wrmse(pred, actual, w)
  floor <- noise_floor_rmse(pred, w)
  excess <- sqrt(max(0, achieved^2 - floor^2))
  cat(sprintf("  %-28s achieved RMSE %.4f | noise floor %.4f | approx excess (model) RMSE %.4f\n",
              label, achieved, floor, excess))
}

#' Binomial log-loss per batter faced: -mean over batters of
#' y*log(p) + (1-y)*log(1-p). The loss the probability models minimize.
binom_logloss <- function(pred, actual, w) {
  p <- pmin(pmax(pred, 1e-6), 1 - 1e-6)
  -sum(w * (actual * log(p) + (1 - actual) * log(1 - p))) / sum(w)
}

#' TBF-weighted R^2 against the TBF-weighted mean of actual K%.
wr2 <- function(pred, actual, w) {
  1 - sum(w * (actual - pred)^2) / sum(w * (actual - weighted.mean(actual, w))^2)
}

#' Model-error variance left after removing binomial sampling noise:
#' E[(pred - actual)^2] ~= sigma2_model + p(1-p)/TBF, so
#' sigma2_model ~= weighted mean of (squared error - p(1-p)/TBF), floored at 0.
model_error_var <- function(pred, actual, w) {
  max(0, weighted.mean((pred - actual)^2 - pred * (1 - pred) / w, w))
}


# ── Step 1: Load Merged Data ─────────────────────────────────────────────────
cat("\n=== Step 1: Loading merged data ===\n")

# guess_max raised: readr's default 1000-row type guess misreads blank
# rare-pitch-type columns (e.g. forkball) as logical, corrupting later rows.
d <- read_csv(file.path(data_dir, "k_2026_final_merged.csv"), show_col_types = FALSE,
              guess_max = 10000)

# Schema check: every model input must exist and be numeric. A stray text
# token (e.g. "--") makes readr read the whole column as text, which would
# otherwise surface only as an opaque error deep inside model fitting.
# (A column that is entirely blank reads as logical; that is harmless.)
missing_cols <- setdiff(raw_input_cols, names(d))
if (length(missing_cols) > 0) {
  stop("k_2026_final_merged.csv is missing input column(s): ",
       paste(missing_cols, collapse = ", "), call. = FALSE)
}
not_numeric <- raw_input_cols[!vapply(d[raw_input_cols],
                                      function(x) is.numeric(x) || all(is.na(x)), logical(1))]
if (length(not_numeric) > 0) {
  examples <- vapply(not_numeric, function(col) {
    x <- d[[col]]
    bad_value <- x[!is.na(x) & is.na(suppressWarnings(as.numeric(x)))][1]
    sprintf("%s (e.g. \"%s\" in row %d)", col, bad_value, match(bad_value, x))
  }, character(1))
  stop("Non-numeric values in input column(s): ", paste(examples, collapse = "; "),
       ". Fix or blank them in the source file and re-run the merge.", call. = FALSE)
}

# lag() below assumes exactly one row per player-season; a duplicate row
# would make a lag feature equal that row's own K% (target leakage).
stopifnot(!anyDuplicated(d[c("PlayerId", "Season")]), all(d$TBF > 0))

d <- d %>% rename(K_pct = `K%`, StuffPlus = `Stuff+`)


# ── Step 2: Feature Engineering ──────────────────────────────────────────────
cat("\n=== Step 2: Building features ===\n")

d <- d %>%
  mutate(role_ratio = ifelse(p_game > 0, p_starting_p / p_game, NA_real_))

# Trailing league K%, shifted one season so a row never sees its own year.
league_by_season <- d %>%
  group_by(Season) %>%
  summarise(league_k_this_season = weighted.mean(K_pct, TBF), .groups = "drop") %>%
  arrange(Season) %>%
  mutate(league_k_trailing = lag(league_k_this_season))
d <- d %>% left_join(league_by_season %>% select(Season, league_k_trailing), by = "Season")

d <- d %>% arrange(PlayerId, Season)

d <- d %>%
  group_by(PlayerId) %>%
  mutate(across(all_of(lag_both_cols), list(lag1 = ~lag(., 1), lag2 = ~lag(., 2)),
                .names = "{.col}_{.fn}")) %>%
  mutate(across(all_of(lag1_only_cols), ~lag(., 1), .names = "{.col}_lag1")) %>%
  # lag() steps back one ROW, not one season. Blank any lag whose source
  # row isn't exactly 1 (or 2) seasons back, so a 2023 season is never
  # presented to a 2025 row as "last season".
  mutate(gap1 = Season - lag(Season, 1), gap2 = Season - lag(Season, 2)) %>%
  mutate(across(ends_with("_lag1"), ~ if_else(gap1 %in% 1, .x, NA)),
         across(ends_with("_lag2"), ~ if_else(gap2 %in% 2, .x, NA))) %>%
  ungroup() %>%
  select(-gap1, -gap2)

# TBF-weighted trailing avg K% over up to 3 prior seasons played, vectorized.
# (Deliberately seasons *played*, not calendar seasons: an older season is
# still real history, and EB weights it by its TBF.)
d <- d %>%
  group_by(PlayerId) %>%
  mutate(
    K_trail3 = lag(K_pct, 1), TBF_trail3 = lag(TBF, 1),
    K_trail2 = lag(K_pct, 2), TBF_trail2 = lag(TBF, 2),
    K_trail1 = lag(K_pct, 3), TBF_trail1 = lag(TBF, 3)
  ) %>%
  ungroup()

tbf_mat <- as.matrix(d[, c("TBF_trail3", "TBF_trail2", "TBF_trail1")])
k_mat   <- as.matrix(d[, c("K_trail3", "K_trail2", "K_trail1")])
d$trail_TBF_sum <- rowSums(tbf_mat, na.rm = TRUE)
weighted_k_sum   <- rowSums(k_mat * tbf_mat, na.rm = TRUE)
d$trail_K_weighted <- ifelse(d$trail_TBF_sum > 0, weighted_k_sum / d$trail_TBF_sum, NA_real_)
d <- d %>% select(-K_trail3, -TBF_trail3, -K_trail2, -TBF_trail2, -K_trail1, -TBF_trail1)

d <- d %>%
  group_by(PlayerId) %>%
  mutate(
    n_prior_seasons = row_number() - 1,
    career_TBF_to_date = lag(cumsum(TBF), 1),
    prior_season_gap = (Season - lag(Season, 1)) > 1 & !is.na(lag(Season, 1))
  ) %>%
  ungroup()

# Role change flag: clean flip between mostly-starting and mostly-relieving
# across the two most recent prior seasons.
d <- d %>%
  mutate(
    role_change_recent = case_when(
      is.na(role_ratio_lag1) | is.na(role_ratio_lag2) ~ NA,
      (role_ratio_lag1 > 0.7 & role_ratio_lag2 < 0.3) |
        (role_ratio_lag1 < 0.3 & role_ratio_lag2 > 0.7) ~ TRUE,
      TRUE ~ FALSE
    ),
    Age_sq = Age^2,
    role_change_recent_num = as.numeric(role_change_recent),
    prior_season_gap_num   = as.numeric(prior_season_gap)
  )

cat("Rows:", nrow(d), " Columns:", ncol(d), "\n")
cat("Rows with a usable trailing K% (n_prior_seasons > 0):",
    sum(!is.na(d$trail_K_weighted)), "\n")
cat("Rows with zero prior seasons (true cold start):",
    sum(d$n_prior_seasons == 0), "\n")


# ── Step 3: Population Routing (Bucket Assignment) ───────────────────────────
cat("\n=== Step 3: Assigning buckets ===\n")

# Routing uses only information from seasons before the row's own (flags
# built as-of in the merge), so every season, 2021-2024 and 2025 alike, is
# routed by the same rule and each bucket means the same thing every year:
#   C  position player: an MLB hitting season on file, no MLB pitching season
#   A  pitcher with at least one prior season in k_2026
#   B1 no prior k_2026 season, but a prior MiLB season on file
#   B2 no prior k_2026 season and no MiLB record
d <- d %>%
  mutate(bucket = case_when(
    has_prior_batting & !has_prior_pitching ~ "C_position_player",
    n_prior_seasons > 0                     ~ "A_full_model",
    !is.na(milb_year)                       ~ "B1_no_history_milb",
    TRUE                                    ~ "B2_no_history_fallback"
  ))

cat("Bucket sizes by season:\n")
print(table(d$Season, d$bucket))

# Bucket C context mean: position-player cohort runs far lower K% than
# real pitchers, so it gets its own trailing mean instead of league-wide.
pp_context <- d %>%
  filter(bucket == "C_position_player") %>%
  group_by(Season) %>%
  summarise(pp_k_this_season = weighted.mean(K_pct, TBF), .groups = "drop") %>%
  arrange(Season) %>%
  mutate(pp_k_trailing = lag(pp_k_this_season))
pp_context$pp_k_trailing[1] <- pp_context$pp_k_this_season[1]  # bootstrap first season
d <- d %>% left_join(pp_context %>% select(Season, pp_k_trailing), by = "Season")


# =============================================================================
# Shared Model-Fitting Functions
#
# One set of functions fits/predicts EB + glmnet + xgboost on any
# train/target split. Steps 4-6 and 10 all call these.
# =============================================================================

feat_cols <- c(
  "Age", "Age_sq", "role_ratio_lag1", "role_ratio_lag2",
  "n_prior_seasons", "career_TBF_to_date",
  "trail_K_weighted", "trail_TBF_sum",
  "K_pct_lag1", "K_pct_lag2", "TBF_lag1", "TBF_lag2",
  "StuffPlus_lag1", "StuffPlus_lag2",
  "bb_percent_lag1", "whiff_percent_lag1", "oz_swing_percent_lag1",
  "z_swing_miss_percent_lag1", "hard_hit_percent_lag1",
  "barrel_batted_rate_lag1", "edge_percent_lag1", "f_strike_percent_lag1",
  "groundballs_percent_lag1", "fastball_avg_speed_lag1", "csw_percent_lag1",
  "league_k_trailing",
  "role_change_recent_num", "prior_season_gap_num",
  paste0(arsenal_cols, "_lag1")
)
k_grid <- seq(10, 400, by = 5)

#' EB shrinkage prediction: pred = w*trail_K + (1-w)*league context,
#' w = trail_TBF_sum / (trail_TBF_sum + k). Equivalent to the beta-binomial
#' posterior mean (K + k*m)/(TBF + k) with prior mean m and prior strength
#' k batters faced [1]. k is fit to minimize next-season error, so it is
#' larger than K%'s within-season stabilization point (~70 BF [2]): it must
#' also absorb real year-to-year changes in talent.
eb_predict <- function(dat, k) {
  w <- dat$trail_TBF_sum / (dat$trail_TBF_sum + k)
  w * dat$trail_K_weighted + (1 - w) * dat$league_k_trailing
}
eb_wrmse <- function(k, dat) wrmse(eb_predict(dat, k), dat$K_pct, dat$TBF)

#' Grid-search the EB stabilization constant k on `train`.
fit_eb <- function(train) k_grid[which.min(sapply(k_grid, eb_wrmse, dat = train))]

#' Build a model matrix with median imputation + missingness flags.
#' Returns list(X, impute_vals); pass a train fit's impute_vals back in to
#' build a matching matrix for new data.
build_matrix <- function(dat, feats, impute_vals = NULL) {
  X <- dat[, feats]
  miss_flags <- as.data.frame(lapply(X, function(col) as.numeric(is.na(col))))
  names(miss_flags) <- paste0(names(X), "_missing")
  if (is.null(impute_vals)) {
    impute_vals <- sapply(X, function(col) median(col, na.rm = TRUE))
    impute_vals[!is.finite(impute_vals)] <- 0  # column 100% NA in this fold
  }
  for (nm in names(X)) X[[nm]][is.na(X[[nm]])] <- impute_vals[[nm]]
  keep <- sapply(miss_flags, function(col) length(unique(col)) > 1)  # drop constant flags
  list(X = as.matrix(cbind(X, miss_flags[, keep, drop = FALSE])), impute_vals = impute_vals)
}

#' Align a new-data matrix to a train matrix's exact column set: pad any
#' missing flag column with 0, then reorder to match train_cols.
align_matrix <- function(X, train_cols) {
  missing_cols <- setdiff(train_cols, colnames(X))
  if (length(missing_cols) > 0) {
    pad <- matrix(0, nrow = nrow(X), ncol = length(missing_cols), dimnames = list(NULL, missing_cols))
    X <- cbind(X, pad)
  }
  X[, train_cols, drop = FALSE]
}

# Hyperparameters are chosen inside every training window by pitcher-grouped
# cross-validation on binomial log-loss (fit_stage_models below), never on
# the 2023/2024 backtest seasons, so those stay an honest test.
#   Elastic net mixing alpha: 0 = ridge (L2) ... 1 = lasso (L1); lambda
#     (penalty strength) is tuned along its whole path for each alpha.
#   xgboost tree depth and minimum leaf size [4]. min_child_weight is in TBF
#     units (the weights), so it is the fewest batters faced a leaf may hold.
#   xgboost rounds: early stopping on the same CV folds.
# Learning rate and row/column subsampling stay fixed at conventional values:
# with early stopping, a smaller learning rate mostly trades speed for a
# slightly smoother fit.
glmnet_alpha_grid <- c(0, 0.25, 0.5, 0.75, 1)
xgb_tune_grid <- expand.grid(max_depth = c(2, 3, 4, 6), min_child_weight = c(20, 50, 100))
xgb_base_params <- list(objective = "binary:logistic", eval_metric = "logloss",
                        eta = 0.02, subsample = 0.7, colsample_bytree = 0.7,
                        lambda = 1, nthread = 2)
n_cv_folds <- 5

#' Cross-validation folds that keep all of a pitcher's seasons together, so
#' tuning never scores a pitcher the model already saw [6] (used only inside
#' a training window; the backtest itself is split by season [5]).
pitcher_folds <- function(ids, n = n_cv_folds) {
  u <- unique(ids)
  f <- sample(rep(seq_len(n), length.out = length(u)))
  f[match(ids, u)]
}

#' Shrink the K% inputs before the elastic net and xgboost see them:
#' (K + k*m)/(TBF + k), the beta-binomial posterior mean [1] toward the prior
#' season's league rate m, with the same k (in TBF) as the EB model. So a
#' 3-for-3 season enters the models as ~league average, not 100%.
#' (Shrinking the other prior-season rates, e.g. whiff%, the same way was
#' also tested; it made the 2023-2024 backtest worse, 0.0398 vs 0.0395
#' RMSE, so they enter raw.)
shrink_k_inputs <- function(dat, k) {
  dat %>% mutate(
    K_pct_lag1 = (K_pct_lag1 * TBF_lag1 + k * league_k_trailing) / (TBF_lag1 + k),
    K_pct_lag2 = (K_pct_lag2 * TBF_lag2 + k * league_k_trailing) / (TBF_lag2 + k),
    trail_K_weighted = (trail_K_weighted * trail_TBF_sum + k * league_k_trailing) / (trail_TBF_sum + k)
  )
}

#' Binomial elastic net on a (non-strikeouts, strikeouts) count matrix. Each
#' alpha gets its own lambda path on the same pitcher-grouped folds; keep the
#' alpha whose best lambda has the lowest CV binomial deviance.
#' See reference [3].
fit_glmnet_tuned <- function(X, y_counts, folds) {
  fits <- lapply(glmnet_alpha_grid, function(a)
    cv.glmnet(X, y_counts, family = "binomial", alpha = a, foldid = folds))
  best <- which.min(sapply(fits, function(f) min(f$cvm)))
  list(fit = fits[[best]], alpha = glmnet_alpha_grid[best])
}

#' Fit EB + binomial elastic net + logistic xgboost on `train`. All three
#' predict a probability, so every output is inside (0, 1) by construction.
fit_stage_models <- function(train, feats = feat_cols) {
  k <- fit_eb(train)
  train <- shrink_k_inputs(train, k)
  mm <- build_matrix(train, feats)
  folds <- pitcher_folds(train$PlayerId)
  strikeouts <- train$K_pct * train$TBF

  # Elastic net on strikeout counts, alpha and lambda tuned by grouped CV.
  en <- fit_glmnet_tuned(mm$X, cbind(train$TBF - strikeouts, strikeouts), folds)
  fit_glmnet <- en$fit

  # xgboost on K% with TBF weights = the same binomial log-likelihood. Each
  # depth / leaf-size pair is early-stopped on the same folds; keep the pair
  # (and its round count) with the lowest CV log-loss.
  dtrain <- xgb.DMatrix(mm$X, label = train$K_pct, weight = train$TBF)
  fold_index <- lapply(seq_len(n_cv_folds), function(f) which(folds == f))
  xgb_cv <- lapply(seq_len(nrow(xgb_tune_grid)), function(g) {
    params <- c(xgb_base_params, as.list(xgb_tune_grid[g, ]))
    log <- xgb.cv(params, dtrain, nrounds = 3000, folds = fold_index,
                  early_stopping_rounds = 100, verbose = 0)$evaluation_log
    list(params = params, n_rounds = which.min(log$test_logloss_mean),
         logloss = min(log$test_logloss_mean))
  })
  best_xgb <- xgb_cv[[which.min(sapply(xgb_cv, `[[`, "logloss"))]]
  fit_xgb <- xgb.train(best_xgb$params, dtrain, nrounds = best_xgb$n_rounds, verbose = 0)

  list(k = k, fit_glmnet = fit_glmnet, glmnet_alpha = en$alpha,
       fit_xgb = fit_xgb, xgb_params = best_xgb$params, n_rounds = best_xgb$n_rounds,
       impute_vals = mm$impute_vals, train_cols = colnames(mm$X), feats = feats)
}

#' Apply a model bundle to newdata; adds pred_EB/pred_glmnet/pred_xgb.
predict_stage_models <- function(models, newdata) {
  newdata$pred_EB <- eb_predict(newdata, models$k)
  shrunk <- shrink_k_inputs(newdata, models$k)
  mm <- build_matrix(shrunk, models$feats, impute_vals = models$impute_vals)
  Xaligned <- align_matrix(mm$X, models$train_cols)
  newdata$pred_glmnet <- as.numeric(predict(models$fit_glmnet, newx = Xaligned, s = "lambda.min", type = "response"))
  newdata$pred_xgb <- predict(models$fit_xgb, xgb.DMatrix(Xaligned))
  newdata
}

blend_predict <- function(dat, w) w["EB"] * dat$pred_EB + w["glmnet"] * dat$pred_glmnet + w["xgb"] * dat$pred_xgb

#' Grid-search non-negative, sum-to-one blend weights on `dat`. Constrained
#' rather than an unconstrained lm(), which can hand a model a negative
#' weight — a sign of overfitting the blend itself.
fit_blend_weights <- function(dat, grid_step = 0.05) {
  best <- list(rmse = Inf, w = c(EB = 1/3, glmnet = 1/3, xgb = 1/3))
  for (w_eb in seq(0, 1, by = grid_step)) {
    for (w_glm in seq(0, 1 - w_eb, by = grid_step)) {
      w <- c(EB = w_eb, glmnet = w_glm, xgb = 1 - w_eb - w_glm)
      r <- wrmse(blend_predict(dat, w), dat$K_pct, dat$TBF)
      if (r < best$rmse) best <- list(rmse = r, w = w)
    }
  }
  best$w
}


# ── Steps 4-5: Rolling-Origin Backtest (Bucket A) ────────────────────────────
# Train on seasons before t, predict t, for t = 2023 and 2024 [5]. Nothing
# about the models is tuned on the season being predicted; blend weights are
# then fit on the pooled out-of-sample predictions (Step 6).
cat("\n=== Steps 4-5: Bucket A rolling-origin backtest ===\n")

bucket_A <- d %>% filter(bucket == "A_full_model")

folds <- list(
  list(train = bucket_A %>% filter(Season <= 2022), valid = bucket_A %>% filter(Season == 2023)),
  list(train = bucket_A %>% filter(Season <= 2023), valid = bucket_A %>% filter(Season == 2024))
)

pooled_valid <- list()
for (i in seq_along(folds)) {
  tr <- folds[[i]]$train; va <- folds[[i]]$valid
  cat("Fold", i, "→ train n =", nrow(tr), " valid n =", nrow(va),
      " (valid season", unique(va$Season), ")\n")
  models <- fit_stage_models(tr)
  va <- predict_stage_models(models, va)
  cat("  EB k:", models$k, " elastic net alpha:", models$glmnet_alpha,
      " xgboost depth / min leaf TBF / rounds:", models$xgb_params$max_depth, "/",
      models$xgb_params$min_child_weight, "/", models$n_rounds, "\n")
  pooled_valid[[i]] <- va
}
pooled_valid <- bind_rows(pooled_valid)

metrics <- tibble(
  model = c("League mean (naive)", "Trailing weighted avg (naive)",
            "Empirical Bayes shrinkage", "Elastic net (binomial)", "XGBoost (logistic)"),
  RMSE = c(wrmse(pooled_valid$league_k_trailing, pooled_valid$K_pct, pooled_valid$TBF),
           wrmse(pooled_valid$trail_K_weighted, pooled_valid$K_pct, pooled_valid$TBF),
           wrmse(pooled_valid$pred_EB, pooled_valid$K_pct, pooled_valid$TBF),
           wrmse(pooled_valid$pred_glmnet, pooled_valid$K_pct, pooled_valid$TBF),
           wrmse(pooled_valid$pred_xgb, pooled_valid$K_pct, pooled_valid$TBF)),
  MAE  = c(wmae(pooled_valid$league_k_trailing, pooled_valid$K_pct, pooled_valid$TBF),
           wmae(pooled_valid$trail_K_weighted, pooled_valid$K_pct, pooled_valid$TBF),
           wmae(pooled_valid$pred_EB, pooled_valid$K_pct, pooled_valid$TBF),
           wmae(pooled_valid$pred_glmnet, pooled_valid$K_pct, pooled_valid$TBF),
           wmae(pooled_valid$pred_xgb, pooled_valid$K_pct, pooled_valid$TBF))
)
cat("\nPooled 2023+2024 backtest (n =", nrow(pooled_valid), "):\n")
print(metrics)


# ── Step 6: Blend Weights (Bucket A) ─────────────────────────────────────────
# One set of non-negative, sum-to-one weights for all Bucket A pitchers.
# (An earlier version switched weights by batters-faced tier; it chose
# tiers from the season being predicted and did not beat this out of
# sample, so it was removed.)
cat("\n=== Step 6: Fitting blend weights ===\n")

blend_w <- fit_blend_weights(pooled_valid)
cat("Blend weights (fit on pooled 2023+2024 validation):\n")
print(round(blend_w, 3))
pooled_valid$pred_blend <- blend_predict(pooled_valid, blend_w)
cat("Blend RMSE (pooled):", round(wrmse(pooled_valid$pred_blend, pooled_valid$K_pct, pooled_valid$TBF), 4),
    " MAE:", round(wmae(pooled_valid$pred_blend, pooled_valid$K_pct, pooled_valid$TBF), 4),
    " log-loss:", round(binom_logloss(pooled_valid$pred_blend, pooled_valid$K_pct, pooled_valid$TBF), 5), "\n")

# Diagnostic only (scored after the fact, never used to make a prediction):
# how much of the error at each workload level is irreducible sampling noise.
tbf_breaks <- quantile(pooled_valid$TBF, probs = c(1 / 3, 2 / 3))
pooled_valid$tbf_tier <- findInterval(pooled_valid$TBF, tbf_breaks, left.open = TRUE) + 1
cat("\nNoise floor check, Bucket A pooled backtest, by TBF tercile:\n")
for (t in 1:3) {
  sub <- pooled_valid %>% filter(tbf_tier == t)
  report_noise_floor(sub$pred_blend, sub$K_pct, sub$TBF,
                     sprintf("tercile %d (median TBF %d)", t, round(median(sub$TBF))))
}


# ── Step 7: MiLB-to-MLB Translation Model (Bucket B1) ────────────────────────
# Trained on early-career players (n_prior_seasons <= 1) with a prior MiLB
# season on file — excludes established MLB pitchers on a rehab stint,
# whose true K% is driven by MLB track record, not one minor-league season.
# The MiLB file has no level column and covers 2022-2024.
cat("\n=== Step 7: MiLB translation model ===\n")

translate_pop <- d %>% filter(!is.na(milb_year), n_prior_seasons <= 1, Season <= 2024)
cat("Training population by season:\n")
print(table(translate_pop$Season))

#' Predict with a fitted MiLB glmnet, using the same imputation, missing
#' flags, and column alignment as its training matrix.
predict_milb <- function(fit, mm_train, newdata) {
  mm_new <- build_matrix(newdata, milb_feat_cols, impute_vals = mm_train$impute_vals)
  as.numeric(predict(fit, newx = align_matrix(mm_new$X, colnames(mm_train$X)),
                     s = "lambda.min", type = "response"))
}

#' Binomial elastic net on strikeout counts, tuned the same way as the
#' Bucket A elastic net (alpha and lambda by pitcher-grouped CV).
fit_milb <- function(dat, mm) {
  strikeouts <- dat$K_pct * dat$TBF
  fit_glmnet_tuned(mm$X, cbind(dat$TBF - strikeouts, strikeouts), pitcher_folds(dat$PlayerId))$fit
}

# Backtest: train on 2023, score 2024 Bucket B1 (2024 is the only season
# with an earlier season to train on, since MiLB data starts in 2022).
milb_train1 <- translate_pop %>% filter(Season == 2023)
mm_milb_train1 <- build_matrix(milb_train1, milb_feat_cols)
fit_milb1 <- fit_milb(milb_train1, mm_milb_train1)
bucketB1_hist <- d %>% filter(bucket == "B1_no_history_milb", Season == 2024)
bucketB1_hist$pred_rule <- predict_milb(fit_milb1, mm_milb_train1, bucketB1_hist)

cat("\n2024 backtest, Bucket B1 (n =", nrow(bucketB1_hist), "):\n")
cat("  League mean baseline: RMSE", round(wrmse(bucketB1_hist$league_k_trailing, bucketB1_hist$K_pct, bucketB1_hist$TBF), 4), "\n")
cat("  MiLB translation:     RMSE", round(wrmse(bucketB1_hist$pred_rule, bucketB1_hist$K_pct, bucketB1_hist$TBF), 4), "\n")
cat("\nCoefficients:\n")
print(coef(fit_milb1, s = "lambda.min"))

mm_milb_final <- build_matrix(translate_pop, milb_feat_cols)
fit_milb_final <- fit_milb(translate_pop, mm_milb_final)


# ── Step 8: Backtested Error for Buckets B2 / C ──────────────────────────────
# Each bucket's prediction rule already existed in 2022-2024 and routing is
# identical in every season, so the backtest population matches 2025's.
cat("\n=== Step 8: Backtested error for B2 / C prediction rules (2022-2024) ===\n")

bucketB2_hist <- d %>% filter(bucket == "B2_no_history_fallback", Season %in% 2022:2024) %>%
  mutate(pred_rule = league_k_trailing)
bucketC_hist <- d %>% filter(bucket == "C_position_player", Season %in% 2022:2024) %>%
  mutate(pred_rule = pp_k_trailing)

cat("Bucket B2 rule (trailing league mean):         n =", nrow(bucketB2_hist),
    " RMSE", round(wrmse(bucketB2_hist$pred_rule, bucketB2_hist$K_pct, bucketB2_hist$TBF), 4), "\n")
cat("Bucket C rule (trailing position-player mean): n =", nrow(bucketC_hist),
    " RMSE", round(wrmse(bucketC_hist$pred_rule, bucketC_hist$K_pct, bucketC_hist$TBF), 4), "\n")


# ── Step 9: Per-Bucket Calibration and Interval Width ────────────────────────
# Both come from pre-2025 backtests only, and every bucket gets the same
# treatment. Calibration ratio = actual strikeouts / predicted strikeouts
# over the bucket's backtest (sum of K_pct*TBF over sum of pred*TBF);
# multiplying by it re-centers the bucket. A ratio rather than a subtracted
# bias because K% is a rate: for pitchers (~22% K) the two are nearly
# identical, but subtracting a 2-point bias from a ~1% position-player
# rate would predict 0%. Interval width = model-error variance left after
# removing binomial noise, measured on the calibrated predictions.
cat("\n=== Step 9: Per-bucket calibration and interval width ===\n")

backtests <- list(
  A_full_model           = tibble(pred = pooled_valid$pred_blend, actual = pooled_valid$K_pct, w = pooled_valid$TBF),
  B1_no_history_milb     = tibble(pred = bucketB1_hist$pred_rule, actual = bucketB1_hist$K_pct, w = bucketB1_hist$TBF),
  B2_no_history_fallback = tibble(pred = bucketB2_hist$pred_rule, actual = bucketB2_hist$K_pct, w = bucketB2_hist$TBF),
  C_position_player      = tibble(pred = bucketC_hist$pred_rule,  actual = bucketC_hist$K_pct,  w = bucketC_hist$TBF)
)
bucket_calib <- sapply(backtests, function(b) sum(b$actual * b$w) / sum(b$pred * b$w))
bucket_sigma2 <- mapply(function(b, ratio) {
  model_error_var(pmin(b$pred * ratio, 1), b$actual, b$w)
}, backtests, bucket_calib)

calib_table <- tibble(bucket = names(backtests), n_backtest = sapply(backtests, nrow),
                      backtest_rmse = sapply(backtests, function(b) wrmse(b$pred, b$actual, b$w)),
                      actual_K = sapply(backtests, function(b) sum(b$actual * b$w)),
                      predicted_K = sapply(backtests, function(b) sum(b$pred * b$w)),
                      calib_ratio = bucket_calib, model_error_sd = sqrt(bucket_sigma2))
print(calib_table %>% mutate(across(where(is.numeric), ~ round(.x, 4))))


# ── Step 10: Final Refit and 2025 Predictions (All Buckets) ─────────────────
cat("\n=== Step 10: Final refit and 2025 predictions ===\n")

final_train <- bucket_A %>% filter(Season <= 2024)
test2025    <- bucket_A %>% filter(Season == 2025)

# Same procedure as each backtest fold, now on all of 2022-2024: the
# xgboost round count comes from grouped CV inside this training set.
final_models <- fit_stage_models(final_train)
cat("EB k re-estimated on 2022-2024:", final_models$k,
    " elastic net alpha:", final_models$glmnet_alpha,
    " xgboost depth / min leaf TBF / rounds:", final_models$xgb_params$max_depth, "/",
    final_models$xgb_params$min_child_weight, "/", final_models$n_rounds, "\n")
test2025 <- predict_stage_models(final_models, test2025)
test2025$pred_final <- blend_predict(test2025, blend_w)

cat("\nTop 15 xgboost features by share of gain, Bucket A production model:\n")
imp <- xgb.importance(model = final_models$fit_xgb)
tree_influence <- data.frame(var = imp$Feature, rel.inf = 100 * imp$Gain / sum(imp$Gain))
print(head(tree_influence, 15))
cat("\nTop 15 |glmnet coef|, Bucket A production model:\n")
gc <- as.matrix(coef(final_models$fit_glmnet, s = "lambda.min"))
gc <- gc[order(-abs(gc[,1])), , drop = FALSE]
print(head(gc, 15))

bucket_B1_2025 <- d %>% filter(bucket == "B1_no_history_milb", Season == 2025)
bucket_B1_2025$pred_final <- predict_milb(fit_milb_final, mm_milb_final, bucket_B1_2025)

bucket_B2_2025 <- d %>% filter(bucket == "B2_no_history_fallback", Season == 2025) %>%
  mutate(pred_final = league_k_trailing)

bucket_C_2025 <- d %>% filter(bucket == "C_position_player", Season == 2025) %>%
  mutate(pred_final = pp_k_trailing)


# ── Step 11: Combine, Score, and Write Output ────────────────────────────────
cat("\n=== Step 11: Combining predictions and scoring against real 2025 K% ===\n")

out_cols <- c("PlayerId", "Name", "Season", "K_pct", "TBF", "bucket", "pred_final")
all_2025 <- bind_rows(test2025[, out_cols], bucket_B1_2025[, out_cols],
                      bucket_B2_2025[, out_cols], bucket_C_2025[, out_cols])
n_2025 <- sum(d$Season == 2025)
stopifnot(nrow(all_2025) == n_2025, !anyNA(all_2025$pred_final))

# pred_final_raw keeps the pre-calibration number; pred_final (calibrated,
# clipped to [0, 1]) is what's scored and written out from here on.
all_2025 <- all_2025 %>%
  mutate(pred_final_raw = pred_final,
         pred_final = pmin(pmax(pred_final * bucket_calib[bucket], 0), 1))
cat("Effect of calibration on overall 2025 accuracy:\n")
cat("  RMSE before correction:", round(wrmse(all_2025$pred_final_raw, all_2025$K_pct, all_2025$TBF), 4), "\n")
cat("  RMSE after  correction:", round(wrmse(all_2025$pred_final, all_2025$K_pct, all_2025$TBF), 4), "\n")

# ~80% prediction interval for the observed 2025 K%: model error (from the
# backtest) plus binomial sampling noise, which shrinks as TBF grows. TBF
# sets only the interval width for the realized season, never the point
# prediction.
all_2025 <- all_2025 %>%
  mutate(
    interval_halfwidth = interval_z * sqrt(bucket_sigma2[bucket] + pred_final * (1 - pred_final) / TBF),
    pred_lower = pmax(0, pred_final - interval_halfwidth),
    pred_upper = pmin(1, pred_final + interval_halfwidth)
  ) %>%
  select(-interval_halfwidth)

cat("\nFINAL 2025 accuracy vs. actual 2025 K% (all metrics TBF-weighted):\n")
cat("Rows scored:", nrow(all_2025), "of", n_2025, "\n")
cat("Overall  RMSE:", round(wrmse(all_2025$pred_final, all_2025$K_pct, all_2025$TBF), 4),
    " MAE:", round(wmae(all_2025$pred_final, all_2025$K_pct, all_2025$TBF), 4),
    " R^2:", round(wr2(all_2025$pred_final, all_2025$K_pct, all_2025$TBF), 3), "\n")
cat("Binomial log-loss per batter:", round(binom_logloss(all_2025$pred_final, all_2025$K_pct, all_2025$TBF), 4),
    " (league-average prediction:", round(binom_logloss(d$league_k_trailing[d$Season == 2025][match(all_2025$PlayerId,
      d$PlayerId[d$Season == 2025])], all_2025$K_pct, all_2025$TBF), 4), ")\n")
cat("80% interval coverage:", round(mean(all_2025$K_pct >= all_2025$pred_lower &
                                         all_2025$K_pct <= all_2025$pred_upper), 3), "\n")

by_bucket <- all_2025 %>%
  group_by(bucket) %>%
  summarise(n = n(), RMSE = wrmse(pred_final, K_pct, TBF), MAE = wmae(pred_final, K_pct, TBF),
            coverage_80pct = mean(K_pct >= pred_lower & K_pct <= pred_upper),
            .groups = "drop")
print(by_bucket)

cat("\n80% interval coverage by 2025 TBF band:\n")
all_2025 %>%
  mutate(tbf_band = cut(TBF, c(0, 25, 117, 269, Inf))) %>%
  group_by(tbf_band) %>%
  summarise(n = n(), coverage_80pct = mean(K_pct >= pred_lower & K_pct <= pred_upper), .groups = "drop") %>%
  print()

cat("\nNoise floor check, 2025 predictions, by bucket:\n")
for (b in unique(all_2025$bucket)) {
  sub <- all_2025 %>% filter(bucket == b)
  report_noise_floor(sub$pred_final, sub$K_pct, sub$TBF, b)
}

write_csv(all_2025, file.path(data_dir, "k_2026_2025_predictions.csv"))
write_csv(d, file.path(data_dir, "k_2026_features.csv"))
writeLines(capture.output(sessionInfo()), file.path(data_dir, "sessionInfo.txt"))

# Everything the methodology write-up quotes, so its numbers always match this run.
saveRDS(list(
  backtest_metrics = metrics,
  blend_rmse = wrmse(pooled_valid$pred_blend, pooled_valid$K_pct, pooled_valid$TBF),
  blend_w = blend_w,
  xgb_params = final_models$xgb_params, xgb_rounds = final_models$n_rounds,
  glmnet_alpha = final_models$glmnet_alpha,
  eb_k = final_models$k,
  backtest_logloss = binom_logloss(pooled_valid$pred_blend, pooled_valid$K_pct, pooled_valid$TBF),
  milb_backtest = c(n = nrow(bucketB1_hist),
                    league_rmse = wrmse(bucketB1_hist$league_k_trailing, bucketB1_hist$K_pct, bucketB1_hist$TBF),
                    milb_rmse = wrmse(bucketB1_hist$pred_rule, bucketB1_hist$K_pct, bucketB1_hist$TBF)),
  calib_table = calib_table,
  tree_influence = tree_influence,
  glmnet_zero = rownames(gc)[gc[, 1] == 0]
), file.path(data_dir, "k_2026_model_summary.rds"))
cat("\nWrote k_2026_features.csv, k_2026_2025_predictions.csv, sessionInfo.txt\n")
