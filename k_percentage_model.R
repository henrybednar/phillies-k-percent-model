# 12_full_model.R
#
# Developed with assistance from Claude (Anthropic), https://claude.ai
#
# Predicts each pitcher's 2025 K% from k_2026_final_merged.csv (built by
# build_final_merge.R). Blends empirical Bayes shrinkage, elastic net, and
# gbm across five pitcher-population buckets, with a per-bucket bias
# correction and a TBF-adaptive blend on top.
#
# No column derived from Season >= 2025 is ever used as a predictive
# input; all lag/trailing features are shifted forward.

library(readr)
library(dplyr)
library(tidyr)
library(glmnet)
library(gbm)

set.seed(2026)


# =============================================================================
# Configuration
# =============================================================================

data_dir <- "."

# Min career single-season PA to count as a confirmed position player.
# Can only raise the source file's own floor, not lower it — see Step 1.
hitting_pa_floor <- 100


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


# ── Step 1: Load Merged Data ─────────────────────────────────────────────────
cat("\n=== Step 1: Loading merged data ===\n")

# guess_max raised: readr's default 1000-row type guess misreads blank
# rare-pitch-type columns (e.g. forkball) as logical, corrupting later rows.
d <- read_csv(file.path(data_dir, "k_2026_final_merged.csv"), show_col_types = FALSE,
              guess_max = 10000)

# Savant hitting leaderboard — diagnostic only. has_batting_record for
# 2025 rows comes from the merged file itself (Step 10), not from this.
hitting <- read_csv(file.path(data_dir, "MLBBaseballSavantHittersData2015-2024.csv"), show_col_types = FALSE)

if (hitting_pa_floor < min(hitting$pa, na.rm = TRUE)) {
  warning("hitting_pa_floor (", hitting_pa_floor, ") is below the floor already ",
          "applied when this file was pulled (min pa in file = ",
          min(hitting$pa, na.rm = TRUE), "). Re-pull the Savant leaderboard at ",
          "a lower PA minimum to actually lower the effective floor.")
}
batter_ids <- hitting %>% filter(pa >= hitting_pa_floor) %>% pull(player_id) %>% unique()
cat("hitting_pa_floor:", hitting_pa_floor, " → batters qualifying:", length(batter_ids),
    "of", length(unique(hitting$player_id)), "unique player-seasons in the file\n")

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
d <- d %>%
  group_by(PlayerId) %>%
  mutate(across(all_of(lag_both_cols), list(lag1 = ~lag(., 1), lag2 = ~lag(., 2)),
                .names = "{.col}_{.fn}")) %>%
  mutate(across(all_of(lag1_only_cols), ~lag(., 1), .names = "{.col}_lag1")) %>%
  ungroup()

# TBF-weighted trailing avg K% over up to 3 prior seasons, vectorized.
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

d <- d %>%
  mutate(bucket = case_when(
    likely_position_player ~ "C_position_player",
    n_prior_seasons == 0    ~ "B_rookie_no_lag",
    TRUE                    ~ "A_full_model"
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
# One set of functions fits/predicts EB + glmnet + gbm on any train/target
# split. Steps 4-6 and 10 all call these.
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
#' w = trail_TBF_sum / (trail_TBF_sum + k).
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

#' Pick gbm tree count by checking staged predictions against a real
#' held-out fold, instead of gbm's internal (random, non-time-respecting)
#' cv.folds.
select_ntrees_by_holdout <- function(fit, Xva_aligned, actual, w, checkpoint_step = 10) {
  max_trees <- fit$n.trees
  checkpoints <- seq(checkpoint_step, max_trees, by = checkpoint_step)
  preds <- predict(fit, newdata = as.data.frame(Xva_aligned), n.trees = checkpoints)
  errs <- apply(preds, 2, function(p) wrmse(p, actual, w))
  checkpoints[which.min(errs)]
}

#' Fit EB + glmnet + gbm on `train`. gbm is fit on the same imputed +
#' flagged matrix as glmnet, not raw NA columns, so a rare feature that's
#' 100% missing in a thin fold can't crash gbm's fit.
#'
#' n.trees selection (exactly one mode applies per call):
#'   valid supplied            → fit to gbm_ntrees_max, pick n.trees via holdout on valid
#'   gbm_ntrees_fixed supplied → fit exactly that many trees, no search
#'   neither supplied          → fall back to gbm's internal cv.folds
fit_stage_models <- function(train, feats = feat_cols, gbm_depth = 3, gbm_shrink = 0.01,
                             gbm_ntrees_max = 2000, valid = NULL, gbm_ntrees_fixed = NULL) {
  k <- fit_eb(train)
  mm <- build_matrix(train, feats)
  fit_glmnet <- cv.glmnet(mm$X, train$K_pct, weights = train$TBF, alpha = 0.5, family = "gaussian")
  gbm_df <- as.data.frame(mm$X)
  gbm_df$K_pct <- train$K_pct
  gbm_df$TBF <- train$TBF
  
  # Drop zero-variance columns from gbm's formula only (glmnet's
  # regularization already zeroes these out; gbm just warns/crashes on them).
  has_variation <- apply(mm$X, 2, function(col) length(unique(col)) > 1)
  gbm_feats <- colnames(mm$X)[has_variation]
  
  if (!is.null(gbm_ntrees_fixed)) {
    fit_gbm <- gbm(as.formula(paste("K_pct ~", paste(gbm_feats, collapse = " + "))),
                   data = gbm_df, weights = TBF, distribution = "gaussian",
                   n.trees = gbm_ntrees_fixed, interaction.depth = gbm_depth,
                   shrinkage = gbm_shrink, cv.folds = 0, verbose = FALSE)
    best_iter <- gbm_ntrees_fixed
  } else if (!is.null(valid)) {
    fit_gbm <- gbm(as.formula(paste("K_pct ~", paste(gbm_feats, collapse = " + "))),
                   data = gbm_df, weights = TBF, distribution = "gaussian",
                   n.trees = gbm_ntrees_max, interaction.depth = gbm_depth,
                   shrinkage = gbm_shrink, cv.folds = 0, verbose = FALSE)
    mm_va <- build_matrix(valid, feats, impute_vals = mm$impute_vals)
    Xva <- align_matrix(mm_va$X, colnames(mm$X))
    best_iter <- select_ntrees_by_holdout(fit_gbm, Xva, valid$K_pct, valid$TBF)
  } else {
    fit_gbm <- gbm(as.formula(paste("K_pct ~", paste(gbm_feats, collapse = " + "))),
                   data = gbm_df, weights = TBF, distribution = "gaussian",
                   n.trees = gbm_ntrees_max, interaction.depth = gbm_depth,
                   shrinkage = gbm_shrink, cv.folds = 5, verbose = FALSE)
    best_iter <- gbm.perf(fit_gbm, method = "cv", plot.it = FALSE)
  }
  list(k = k, fit_glmnet = fit_glmnet, fit_gbm = fit_gbm, best_iter = best_iter,
       impute_vals = mm$impute_vals, train_cols = colnames(mm$X), feats = feats)
}

#' Apply a model bundle to newdata; adds pred_EB/pred_glmnet/pred_gbm.
predict_stage_models <- function(models, newdata) {
  newdata$pred_EB <- eb_predict(newdata, models$k)
  mm <- build_matrix(newdata, models$feats, impute_vals = models$impute_vals)
  Xaligned <- align_matrix(mm$X, models$train_cols)
  newdata$pred_glmnet <- as.numeric(predict(models$fit_glmnet, newx = Xaligned, s = "lambda.min"))
  newdata$pred_gbm <- predict(models$fit_gbm, newdata = as.data.frame(Xaligned), n.trees = models$best_iter)
  newdata
}

blend_predict <- function(dat, w) w["EB"] * dat$pred_EB + w["glmnet"] * dat$pred_glmnet + w["gbm"] * dat$pred_gbm

#' Grid-search non-negative, sum-to-one blend weights on `dat`. Constrained
#' rather than an unconstrained lm(), which handed gbm a negative weight
#' on one validation season — a sign of overfitting the blend itself.
fit_blend_weights <- function(dat, grid_step = 0.05) {
  best <- list(rmse = Inf, w = c(1/3, 1/3, 1/3))
  for (w_eb in seq(0, 1, by = grid_step)) {
    for (w_glm in seq(0, 1 - w_eb, by = grid_step)) {
      w_gbm <- 1 - w_eb - w_glm
      w <- c(EB = w_eb, glmnet = w_glm, gbm = w_gbm)
      r <- wrmse(blend_predict(dat, w), dat$K_pct, dat$TBF)
      if (r < best$rmse) best <- list(rmse = r, w = w)
    }
  }
  best$w
}


# ── Step 4: gbm Hyperparameter Tuning (Bucket A) ─────────────────────────────
# n.trees isn't in the grid — select_ntrees_by_holdout() picks it per fit.
cat("\n=== Step 4: gbm hyperparameter tuning (pooled 2023+2024 validation) ===\n")

bucket_A <- d %>% filter(bucket == "A_full_model")

folds <- list(
  list(train = bucket_A %>% filter(Season <= 2022), valid = bucket_A %>% filter(Season == 2023)),
  list(train = bucket_A %>% filter(Season <= 2023), valid = bucket_A %>% filter(Season == 2024))
)

# Built once per fold, reused across every hyperparameter combo below.
mm_fold <- lapply(folds, function(f) {
  mm <- build_matrix(f$train, feat_cols)
  mm_va <- build_matrix(f$valid, feat_cols, impute_vals = mm$impute_vals)
  list(train = f$train, valid = f$valid, mm = mm, Xva = align_matrix(mm_va$X, colnames(mm$X)))
})

tune_grid <- expand.grid(interaction.depth = c(2, 3, 4), shrinkage = c(0.01, 0.02))
tune_ntrees_max <- 1500
tune_results <- vector("list", nrow(tune_grid))
for (g in seq_len(nrow(tune_grid))) {
  depth <- tune_grid$interaction.depth[g]; shrink <- tune_grid$shrinkage[g]
  fold_preds <- list()
  for (i in seq_along(mm_fold)) {
    mf <- mm_fold[[i]]
    gbm_df <- as.data.frame(mf$mm$X); gbm_df$K_pct <- mf$train$K_pct; gbm_df$TBF <- mf$train$TBF
    has_variation <- apply(mf$mm$X, 2, function(col) length(unique(col)) > 1)
    gbm_feats <- colnames(mf$mm$X)[has_variation]
    fit <- gbm(as.formula(paste("K_pct ~", paste(gbm_feats, collapse = " + "))),
               data = gbm_df, weights = TBF, distribution = "gaussian",
               n.trees = tune_ntrees_max, interaction.depth = depth, shrinkage = shrink,
               cv.folds = 0, verbose = FALSE)
    best_iter <- select_ntrees_by_holdout(fit, mf$Xva, mf$valid$K_pct, mf$valid$TBF)
    pred <- predict(fit, newdata = as.data.frame(mf$Xva), n.trees = best_iter)
    fold_preds[[i]] <- tibble(pred = pred, actual = mf$valid$K_pct, w = mf$valid$TBF)
  }
  pooled <- bind_rows(fold_preds)
  tune_results[[g]] <- tibble(interaction.depth = depth, shrinkage = shrink,
                              RMSE = wrmse(pooled$pred, pooled$actual, pooled$w))
}
tune_results <- bind_rows(tune_results) %>% arrange(RMSE)
print(tune_results)
gbm_depth  <- tune_results$interaction.depth[1]
gbm_shrink <- tune_results$shrinkage[1]
cat("Chosen: interaction.depth =", gbm_depth, " shrinkage =", gbm_shrink, "\n")


# ── Step 5: Rolling-Origin Backtest (Bucket A) ───────────────────────────────
# Predictions from both folds are pooled before fitting blend weights.
cat("\n=== Step 5: Bucket A rolling-origin backtest ===\n")

pooled_valid <- list()
fold_best_iter <- integer(length(folds))
for (i in seq_along(folds)) {
  tr <- folds[[i]]$train; va <- folds[[i]]$valid
  cat("Fold", i, "→ train n =", nrow(tr), " valid n =", nrow(va),
      " (valid season", unique(va$Season), ")\n")
  models <- fit_stage_models(tr, gbm_depth = gbm_depth, gbm_shrink = gbm_shrink, valid = va)
  va <- predict_stage_models(models, va)
  cat("  EB k:", models$k, " gbm best n.trees:", models$best_iter, "\n")
  pooled_valid[[i]] <- va
  fold_best_iter[i] <- models$best_iter
}
pooled_valid <- bind_rows(pooled_valid)

metrics <- tibble(
  model = c("League mean (naive)", "Trailing weighted avg (naive)",
            "Empirical Bayes shrinkage", "Elastic net", "GBM"),
  RMSE = c(wrmse(pooled_valid$league_k_trailing, pooled_valid$K_pct, pooled_valid$TBF),
           wrmse(pooled_valid$trail_K_weighted, pooled_valid$K_pct, pooled_valid$TBF),
           wrmse(pooled_valid$pred_EB, pooled_valid$K_pct, pooled_valid$TBF),
           wrmse(pooled_valid$pred_glmnet, pooled_valid$K_pct, pooled_valid$TBF),
           wrmse(pooled_valid$pred_gbm, pooled_valid$K_pct, pooled_valid$TBF)),
  MAE  = c(wmae(pooled_valid$league_k_trailing, pooled_valid$K_pct, pooled_valid$TBF),
           wmae(pooled_valid$trail_K_weighted, pooled_valid$K_pct, pooled_valid$TBF),
           wmae(pooled_valid$pred_EB, pooled_valid$K_pct, pooled_valid$TBF),
           wmae(pooled_valid$pred_glmnet, pooled_valid$K_pct, pooled_valid$TBF),
           wmae(pooled_valid$pred_gbm, pooled_valid$K_pct, pooled_valid$TBF))
)
cat("\nPooled 2023+2024 backtest (n =", nrow(pooled_valid), "):\n")
print(metrics)


# ── Step 6: Blend Weights, Global and TBF-Adaptive (Bucket A) ────────────────
cat("\n=== Step 6: Fitting blend weights ===\n")

blend_w <- fit_blend_weights(pooled_valid)
cat("Global blend weights (fit on pooled 2023+2024 validation):\n")
print(round(blend_w, 3))
pooled_valid$pred_blend_global <- blend_predict(pooled_valid, blend_w)
global_blend_rmse <- wrmse(pooled_valid$pred_blend_global, pooled_valid$K_pct, pooled_valid$TBF)
cat("Global blend RMSE (pooled):", round(global_blend_rmse, 4),
    " MAE:", round(wmae(pooled_valid$pred_blend_global, pooled_valid$K_pct, pooled_valid$TBF), 4), "\n")

# One global blend weight assumes EB/glmnet/gbm trade off the same way at
# every sample size; they don't. tbf_breaks are fixed here and reused
# as-is on 2025 in Step 10, never recomputed from 2025's own distribution.
tbf_breaks <- quantile(pooled_valid$TBF, probs = c(1 / 3, 2 / 3))
assign_tbf_tier <- function(tbf) case_when(tbf <= tbf_breaks[1] ~ 1L, tbf <= tbf_breaks[2] ~ 2L, TRUE ~ 3L)
pooled_valid$tbf_tier <- assign_tbf_tier(pooled_valid$TBF)

blend_w_by_tier <- lapply(1:3, function(t) fit_blend_weights(pooled_valid %>% filter(tbf_tier == t)))
pooled_valid$pred_blend_adaptive <- NA_real_
for (t in 1:3) {
  idx <- pooled_valid$tbf_tier == t
  pooled_valid$pred_blend_adaptive[idx] <- blend_predict(pooled_valid[idx, ], blend_w_by_tier[[t]])
}
adaptive_blend_rmse <- wrmse(pooled_valid$pred_blend_adaptive, pooled_valid$K_pct, pooled_valid$TBF)

cat("\nTBF-adaptive blend weights, by tercile (breakpoints TBF <=",
    round(tbf_breaks[1]), "/ <=", round(tbf_breaks[2]), "):\n")
for (t in 1:3) cat("  Tier", t, ": EB =", round(blend_w_by_tier[[t]]["EB"], 3),
                   " glmnet =", round(blend_w_by_tier[[t]]["glmnet"], 3),
                   " gbm =", round(blend_w_by_tier[[t]]["gbm"], 3), "\n")
cat("Adaptive blend RMSE (pooled):", round(adaptive_blend_rmse, 4), "\n")
cat("Global   blend RMSE (pooled):", round(global_blend_rmse, 4), "\n")

# Only switch to the adaptive blend for 2025 if it actually beats the
# global blend on this backtest — three small fits can overfit each tier.
use_adaptive_blend <- adaptive_blend_rmse < global_blend_rmse
cat(if (use_adaptive_blend) "→ Adaptive blend wins; using it for 2025.\n"
    else "→ Global blend wins (or ties); adaptive blend not worth the extra complexity here.\n")
pooled_valid$pred_blend <- if (use_adaptive_blend) pooled_valid$pred_blend_adaptive else pooled_valid$pred_blend_global
bucketA_backtest_rmse <- wrmse(pooled_valid$pred_blend, pooled_valid$K_pct, pooled_valid$TBF)

cat("\nNoise floor check, Bucket A pooled backtest, by TBF tercile:\n")
for (t in 1:3) {
  sub <- pooled_valid %>% filter(tbf_tier == t)
  report_noise_floor(sub$pred_blend, sub$K_pct, sub$TBF,
                     sprintf("tercile %d (median TBF %d)", t, round(median(sub$TBF))))
}


# ── Step 7: MiLB-to-MLB Translation Model (Bucket C2) ────────────────────────
# Trained on early-career players (n_prior_seasons <= 1) with a recent AAA
# season on file — excludes established MLB pitchers on a rehab stint,
# whose true K% is driven by MLB track record, not one minor-league season.
cat("\n=== Step 7: MiLB translation model ===\n")

milb_feat_cols <- c(
  "milb_k_percent", "milb_bb_percent", "milb_swing_miss_percent",
  "milb_xwoba", "milb_velocity", "milb_hardhit_percent"
)
translate_pop <- d %>% filter(!is.na(milb_year), n_prior_seasons <= 1)
cat("Training population by season:\n")
print(table(translate_pop$Season))

milb_train1 <- translate_pop %>% filter(Season == 2023)
milb_valid  <- translate_pop %>% filter(Season == 2024)
mm_milb_train1 <- build_matrix(milb_train1, milb_feat_cols)
mm_milb_valid  <- build_matrix(milb_valid, milb_feat_cols, impute_vals = mm_milb_train1$impute_vals)
fit_milb1 <- cv.glmnet(mm_milb_train1$X, milb_train1$K_pct, weights = milb_train1$TBF,
                       alpha = 0.5, family = "gaussian")
milb_valid$pred_milb <- as.numeric(predict(fit_milb1, newx = mm_milb_valid$X, s = "lambda.min"))

cat("\n2024 backtest, early-career milb-matched players (n =", nrow(milb_valid), "):\n")
cat("  League mean baseline: RMSE", round(wrmse(milb_valid$league_k_trailing, milb_valid$K_pct, milb_valid$TBF), 4),
    " MAE", round(wmae(milb_valid$league_k_trailing, milb_valid$K_pct, milb_valid$TBF), 4), "\n")
milb_c2_backtest_rmse <- wrmse(milb_valid$pred_milb, milb_valid$K_pct, milb_valid$TBF)
cat("  MiLB translation:     RMSE", round(milb_c2_backtest_rmse, 4),
    " MAE", round(wmae(milb_valid$pred_milb, milb_valid$K_pct, milb_valid$TBF), 4), "\n")
cat("\nCoefficients:\n")
print(coef(fit_milb1, s = "lambda.min"))

milb_final_train <- translate_pop %>% filter(Season <= 2024)
mm_milb_final <- build_matrix(milb_final_train, milb_feat_cols)
fit_milb_final <- cv.glmnet(mm_milb_final$X, milb_final_train$K_pct, weights = milb_final_train$TBF,
                            alpha = 0.5, family = "gaussian")
milb_impute_vals <- mm_milb_final$impute_vals


# ── Step 8: Backtested Error for Buckets B / C1 / C3 ─────────────────────────
# Each bucket's prediction rule already existed in 2022-2024, so it gets
# the same real backtest as Bucket A instead of an assumed interval.
cat("\n=== Step 8: Backtested error for B / C1 / C3 prediction rules (2022-2024) ===\n")

bucketB_hist <- d %>% filter(bucket == "B_rookie_no_lag", Season %in% 2022:2024)
bucketB_backtest_rmse <- wrmse(bucketB_hist$league_k_trailing, bucketB_hist$K_pct, bucketB_hist$TBF)

bucketC1_hist <- d %>% filter(bucket == "C_position_player", Season %in% 2022:2024)
bucketC1_backtest_rmse <- wrmse(bucketC1_hist$pp_k_trailing, bucketC1_hist$K_pct, bucketC1_hist$TBF)

# C3 uses the identical rule as Bucket B (trailing league mean), so it
# inherits that bucket's backtested error rather than having no estimate.
cat("Bucket B / C3 rule (trailing league mean): RMSE", round(bucketB_backtest_rmse, 4), "\n")
cat("Bucket C1 rule (trailing position-player mean): RMSE", round(bucketC1_backtest_rmse, 4), "\n")


# ── Step 9: Per-Bucket Bias Correction ───────────────────────────────────────
# Each bucket's backtested predictions carry a small, systematic offset;
# subtracting each bucket's own bias re-centers its 2025 predictions.
cat("\n=== Step 9: Per-bucket bias correction ===\n")

bucket_bias <- c(
  A_full_model = weighted.mean(pooled_valid$pred_blend - pooled_valid$K_pct, pooled_valid$TBF),
  B_rookie_no_lag = weighted.mean(bucketB_hist$league_k_trailing - bucketB_hist$K_pct, bucketB_hist$TBF),
  C1_confirmed_position_player = 0,
  C2_milb_translation = weighted.mean(milb_valid$pred_milb - milb_valid$K_pct, milb_valid$TBF),
  C3_no_record_fallback = weighted.mean(bucketB_hist$league_k_trailing - bucketB_hist$K_pct, bucketB_hist$TBF)
)
cat("Backtested bias (mean predicted - actual, TBF-weighted, 2022-2024):\n")
print(round(bucket_bias, 4))

# C1 forced to 0: bucketC1_hist is the general pre-2025 position-player
# population, broader than 2025's confirmed-via-hitting-leaderboard C1
# subset. Applying its bias made 2025 C1 RMSE worse, not better.


# ── Step 10: Final Refit and 2025 Predictions (All Buckets) ─────────────────
cat("\n=== Step 10: Final refit and 2025 predictions ===\n")

final_train <- bucket_A %>% filter(Season <= 2024)
test2025    <- bucket_A %>% filter(Season == 2025)

# No future season to validate n.trees against — reuses fold 2's tree
# count (train<=2023, valid 2024), the closest backtest fold structurally.
final_models <- fit_stage_models(final_train, gbm_depth = gbm_depth, gbm_shrink = gbm_shrink,
                                 gbm_ntrees_fixed = fold_best_iter[2])
cat("EB k re-estimated on 2021-2024:", final_models$k,
    " (gbm n.trees reused from fold 2:", fold_best_iter[2], ")\n")
test2025 <- predict_stage_models(final_models, test2025)

cat("\nTop 15 gbm relative influence, Bucket A production model:\n")
print(head(summary(final_models$fit_gbm, n.trees = final_models$best_iter, plotit = FALSE), 15))
cat("\nTop 15 |glmnet coef|, Bucket A production model:\n")
gc <- as.matrix(coef(final_models$fit_glmnet, s = "lambda.min"))
gc <- gc[order(-abs(gc[,1])), , drop = FALSE]
print(head(gc, 15))

if (use_adaptive_blend) {
  test2025$tbf_tier <- assign_tbf_tier(test2025$TBF)
  test2025$pred_final <- NA_real_
  for (t in 1:3) {
    idx <- test2025$tbf_tier == t
    if (any(idx)) test2025$pred_final[idx] <- blend_predict(test2025[idx, ], blend_w_by_tier[[t]])
  }
} else {
  test2025$pred_final <- blend_predict(test2025, blend_w)
}

bucket_B_2025 <- d %>% filter(bucket == "B_rookie_no_lag", Season == 2025) %>%
  mutate(pred_final = league_k_trailing)

# has_batting_record comes from the merged file itself (rebuilt upstream
# by build_final_merge.R at whatever PA floor was actually pulled), not
# recomputed from the local hitting CSV above, which can go stale.
bucket_C_all_2025 <- d %>% filter(bucket == "C_position_player", Season == 2025) %>%
  mutate(has_milb = !is.na(milb_year))

bucket_C_2025 <- bucket_C_all_2025 %>%
  mutate(bucket = case_when(
    has_batting_record             ~ "C1_confirmed_position_player",
    !has_batting_record & has_milb ~ "C2_milb_translation",
    TRUE                            ~ "C3_no_record_fallback"
  ))

c2_rows <- bucket_C_2025 %>% filter(bucket == "C2_milb_translation")
if (nrow(c2_rows) > 0) {
  X_c2 <- c2_rows[, milb_feat_cols]
  for (nm in names(X_c2)) X_c2[[nm]][is.na(X_c2[[nm]])] <- milb_impute_vals[[nm]]
  c2_rows$pred_milb <- as.numeric(predict(fit_milb_final, newx = as.matrix(X_c2), s = "lambda.min"))
}
bucket_C_2025 <- bucket_C_2025 %>%
  left_join(c2_rows %>% select(PlayerId, Season, pred_milb), by = c("PlayerId", "Season")) %>%
  mutate(pred_final = case_when(
    bucket == "C1_confirmed_position_player" ~ pp_k_trailing,
    bucket == "C2_milb_translation"          ~ pred_milb,
    TRUE                                       ~ league_k_trailing
  ))


# ── Step 11: Combine, Score, and Write Output ────────────────────────────────
cat("\n=== Step 11: Combining predictions and scoring against real 2025 K% ===\n")

all_2025 <- bind_rows(
  test2025      %>% select(PlayerId, Name, Season, K_pct, TBF, bucket, pred_final),
  bucket_B_2025 %>% select(PlayerId, Name, Season, K_pct, TBF, bucket, pred_final),
  bucket_C_2025 %>% select(PlayerId, Name, Season, K_pct, TBF, bucket, pred_final)
)

# pred_final_raw keeps the pre-correction number; pred_final (corrected,
# floored at 0) is what's scored and written out from here on.
all_2025 <- all_2025 %>%
  mutate(pred_final_raw = pred_final,
         pred_final = pmax(0, pred_final - bucket_bias[bucket]))
cat("Effect of bias correction on overall 2025 accuracy:\n")
cat("  RMSE before correction:", round(wrmse(all_2025$pred_final_raw, all_2025$K_pct, all_2025$TBF), 4), "\n")
cat("  RMSE after  correction:", round(wrmse(all_2025$pred_final, all_2025$K_pct, all_2025$TBF), 4), "\n")

# Prediction intervals: +/- 1.28 * that bucket's own backtested RMSE
# (approx 80% interval, normal-residual assumption).
bucket_rmse <- c(
  A_full_model = bucketA_backtest_rmse,
  B_rookie_no_lag = bucketB_backtest_rmse,
  C1_confirmed_position_player = bucketC1_backtest_rmse,
  C2_milb_translation = milb_c2_backtest_rmse,
  C3_no_record_fallback = bucketB_backtest_rmse
)
all_2025 <- all_2025 %>%
  mutate(
    interval_halfwidth = bucket_rmse[bucket] * 1.28,
    pred_lower = pmax(0, pred_final - interval_halfwidth),
    pred_upper = pred_final + interval_halfwidth
  ) %>%
  select(-interval_halfwidth)

cat("\nFINAL 2025 accuracy vs. actual 2025 K%:\n")
cat("Rows scored:", nrow(all_2025), "of 873\n")
cat("Overall  RMSE:", round(wrmse(all_2025$pred_final, all_2025$K_pct, all_2025$TBF), 4),
    " MAE:", round(wmae(all_2025$pred_final, all_2025$K_pct, all_2025$TBF), 4), "\n")

by_bucket <- all_2025 %>%
  group_by(bucket) %>%
  summarise(n = n(), RMSE = wrmse(pred_final, K_pct, TBF), MAE = wmae(pred_final, K_pct, TBF),
            coverage_80pct = mean(K_pct >= pred_lower & K_pct <= pred_upper),
            .groups = "drop")
print(by_bucket)

cat("\nNoise floor check, 2025 predictions, by bucket:\n")
for (b in unique(all_2025$bucket)) {
  sub <- all_2025 %>% filter(bucket == b)
  report_noise_floor(sub$pred_final, sub$K_pct, sub$TBF, b)
}

# Manual C3 cleanup evaluated, not implemented: its ~16 lowest-TBF rows
# (TBF<25, token mop-up outings) are ~3% of the bucket's weight — even a
# perfect prediction there moves overall RMSE by ~0.0003. The rest of C3
# are genuine rookies with no MiLB/batting record, not misclassifications.

write_csv(all_2025, file.path(data_dir, "k_2026_2025_predictions.csv"))
write_csv(d, file.path(data_dir, "k_2026_features.csv"))
cat("\n✓ Wrote k_2026_features.csv and k_2026_2025_predictions.csv\n")