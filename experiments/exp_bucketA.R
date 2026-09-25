# Pre-registered Bucket A experiments. Scored ONLY on the 2023/2024 rolling
# backtest (train on seasons < t, predict t). 2025 is never read here.
#
# Candidates (decided before running):
#   base      current production config (EB + gaussian glmnet + gaussian gbm, n.trees from valid fold)
#   honest    base, but gbm n.trees and glmnet lambda chosen by pitcher-grouped CV inside training
#   shrinkK   honest + K% lag features shrunk toward league (beta-binomial posterior mean)
#   newfeat   shrinkK + unused Savant signal: IVB, arm angle, chase-miss, SwStr%, CStr%
#   binom     newfeat with glmnet family = binomial on strikeout counts
#   roleEB    newfeat with EB shrinking toward a starter/reliever-specific prior
#   xgbLogit  newfeat with gbm replaced by xgboost binary:logistic
#   robust    (added after the first run, still backtest-only) the adopted
#             configuration's model types with the original feature set
#
# Evidence behind Table 4 of the methodology; logs of the runs used are in
# exp_bucketA_log.txt and exp_bucketA_robust_log.txt. The final shrinkage
# variants (K%-only vs. also tracking rates) were compared through the model
# script's own backtest output.
#
# Run from the experiments/ folder after the model script:  Rscript exp_bucketA.R
# Needs the (unmaintained) gbm package for the baseline candidates; the
# production model itself does not use gbm.
# Developed with assistance from Claude (Anthropic), https://claude.ai

suppressMessages({library(readr); library(dplyr); library(glmnet); library(gbm); library(xgboost)})
set.seed(2026)

feat_path <- "../k_2026_features.csv"
bs_path   <- "../MLBPitchersBallsStrikes2015-2024.csv"
d <- read_csv(feat_path, show_col_types = FALSE, guess_max = 10000) %>% filter(Season <= 2024)

# ---- extra same-season columns, then season-true lags --------------------------
bs <- read_csv(bs_path, show_col_types = FALSE) %>% distinct() %>%
  transmute(PlayerId = player_id, Season = year,
            swstr_percent = 100 * p_swinging_strike / (p_ball + p_total_strike),
            cstr_percent  = 100 * p_called_strike   / (p_ball + p_total_strike))
d <- d %>% left_join(bs, by = c("PlayerId", "Season"))

lag_season <- function(df, cols) {
  df %>% arrange(PlayerId, Season) %>% group_by(PlayerId) %>%
    mutate(.gap = Season - lag(Season),
           across(all_of(cols), ~ if_else(.gap %in% 1, lag(.x), NA), .names = "{.col}_lag1")) %>%
    ungroup() %>% select(-.gap)
}
new_src <- c("ff_avg_break_z_induced", "arm_angle", "oz_swing_miss_percent", "swstr_percent", "cstr_percent")
d <- lag_season(d, new_src)

# role-specific league K% of the prior season (starters vs relievers), for roleEB
role_prior <- d %>% filter(!is.na(role_ratio)) %>%
  mutate(role = if_else(role_ratio >= 0.5, "SP", "RP")) %>%
  group_by(Season, role) %>% summarise(k = weighted.mean(K_pct, TBF), .groups = "drop") %>%
  mutate(Season = Season + 1)
d <- d %>% mutate(role = if_else(role_ratio_lag1 >= 0.5, "SP", "RP")) %>%
  left_join(role_prior %>% rename(role_k = k), by = c("Season", "role")) %>%
  mutate(role_k = coalesce(role_k, league_k_trailing))

A <- d %>% filter(bucket == "A_full_model")
A$K <- A$K_pct * A$TBF

base_feats <- c("Age", "Age_sq", "role_ratio_lag1", "role_ratio_lag2", "n_prior_seasons", "career_TBF_to_date",
  "trail_K_weighted", "trail_TBF_sum", "K_pct_lag1", "K_pct_lag2", "TBF_lag1", "TBF_lag2",
  "StuffPlus_lag1", "StuffPlus_lag2", "bb_percent_lag1", "whiff_percent_lag1", "oz_swing_percent_lag1",
  "z_swing_miss_percent_lag1", "hard_hit_percent_lag1", "barrel_batted_rate_lag1", "edge_percent_lag1",
  "f_strike_percent_lag1", "groundballs_percent_lag1", "fastball_avg_speed_lag1", "csw_percent_lag1",
  "league_k_trailing", "role_change_recent_num", "prior_season_gap_num",
  paste0(c("ff_avg_speed", "ff_avg_spin", "sl_avg_speed", "sl_avg_spin", "cu_avg_speed", "cu_avg_spin",
           "ch_avg_speed", "ch_avg_spin", "si_avg_speed", "si_avg_spin", "fc_avg_speed", "fc_avg_spin"), "_lag1"))

# ---- shared helpers (same logic as the production script) ---------------------
wrmse <- function(p, y, w) sqrt(weighted.mean((p - y)^2, w))
logloss <- function(p, y, w) {               # per-batter binomial log-loss
  p <- pmin(pmax(p, 1e-4), 1 - 1e-4)
  -sum(w * (y * log(p) + (1 - y) * log(1 - p))) / sum(w)
}
build_matrix <- function(dat, feats, impute_vals = NULL) {
  X <- dat[, feats]
  miss <- as.data.frame(lapply(X, function(c) as.numeric(is.na(c)))); names(miss) <- paste0(names(X), "_missing")
  if (is.null(impute_vals)) {
    impute_vals <- sapply(X, function(c) median(c, na.rm = TRUE)); impute_vals[!is.finite(impute_vals)] <- 0
  }
  for (nm in names(X)) X[[nm]][is.na(X[[nm]])] <- impute_vals[[nm]]
  keep <- sapply(miss, function(c) length(unique(c)) > 1)
  list(X = as.matrix(cbind(X, miss[, keep, drop = FALSE])), impute_vals = impute_vals)
}
align <- function(X, cols) {
  pad <- setdiff(cols, colnames(X))
  if (length(pad)) X <- cbind(X, matrix(0, nrow(X), length(pad), dimnames = list(NULL, pad)))
  X[, cols, drop = FALSE]
}
k_grid <- seq(10, 400, 5)
eb <- function(dat, k, prior) (dat$trail_K_weighted * dat$trail_TBF_sum + k * prior) / (dat$trail_TBF_sum + k)
fit_k <- function(tr, prior) k_grid[which.min(sapply(k_grid, function(k) wrmse(eb(tr, k, prior), tr$K_pct, tr$TBF)))]
group_folds <- function(ids, n = 5) { u <- unique(ids); f <- sample(rep(1:n, length.out = length(u))); f[match(ids, u)] }

# gbm with n.trees picked by the validation season itself (production behaviour)
gbm_valid_pick <- function(X, y, w, Xva, yva, wva) {
  df <- as.data.frame(X); df$y <- y; df$w <- w
  keep <- colnames(X)[apply(X, 2, function(c) length(unique(c)) > 1)]
  f <- gbm(as.formula(paste("y ~", paste(keep, collapse = "+"))), data = df, weights = w, distribution = "gaussian",
           n.trees = 1500, interaction.depth = 4, shrinkage = 0.01, verbose = FALSE)
  cp <- seq(10, 1500, 10); P <- predict(f, as.data.frame(Xva), n.trees = cp)
  best <- cp[which.min(apply(P, 2, wrmse, y = yva, w = wva))]
  predict(f, as.data.frame(Xva), n.trees = best)
}
# gbm with n.trees picked by 5-fold CV grouped by pitcher inside the training seasons
gbm_grouped <- function(X, y, w, ids, Xva) {
  keep <- colnames(X)[apply(X, 2, function(c) length(unique(c)) > 1)]
  fml <- as.formula(paste("y ~", paste(keep, collapse = "+")))
  df <- as.data.frame(X); df$y <- y; df$w <- w
  fo <- group_folds(ids); cp <- seq(10, 1500, 10); err <- matrix(NA, 5, length(cp))
  for (k in 1:5) {
    f <- gbm(fml, data = df[fo != k, ], weights = w, distribution = "gaussian", n.trees = 1500,
             interaction.depth = 4, shrinkage = 0.01, verbose = FALSE)
    P <- predict(f, df[fo == k, ], n.trees = cp)
    err[k, ] <- apply(P, 2, function(p) sum(w[fo == k] * (p - y[fo == k])^2))
  }
  best <- cp[which.min(colSums(err))]
  f <- gbm(fml, data = df, weights = w, distribution = "gaussian", n.trees = best,
           interaction.depth = 4, shrinkage = 0.01, verbose = FALSE)
  predict(f, as.data.frame(Xva), n.trees = best)
}
xgb_grouped <- function(X, y, w, ids, Xva) {
  dtr <- xgb.DMatrix(X, label = y, weight = w)
  p <- list(objective = "binary:logistic", eval_metric = "logloss", max_depth = 4, eta = 0.02,
            subsample = 0.7, colsample_bytree = 0.7, min_child_weight = 50, lambda = 1, nthread = 2)
  fo <- group_folds(ids)
  cv <- xgb.cv(p, dtr, nrounds = 2000, folds = lapply(1:5, function(k) which(fo == k)),
               early_stopping_rounds = 100, verbose = 0)
  best <- which.min(cv$evaluation_log$test_logloss_mean)
  f <- xgb.train(p, dtr, nrounds = best, verbose = 0)
  predict(f, xgb.DMatrix(Xva))
}

run_candidate <- function(cfg) {
  out <- list()
  for (vs in c(2023, 2024)) {
    set.seed(2026 + vs)
    tr <- A %>% filter(Season < vs); va <- A %>% filter(Season == vs)
    prior_tr <- if (cfg$roleEB) tr$role_k else tr$league_k_trailing
    prior_va <- if (cfg$roleEB) va$role_k else va$league_k_trailing
    k <- fit_k(tr, prior_tr)
    feats <- cfg$feats
    if (cfg$shrinkK) {
      shr <- function(dd) dd %>% mutate(
        K_pct_lag1 = (K_pct_lag1 * TBF_lag1 + k * league_k_trailing) / (TBF_lag1 + k),
        K_pct_lag2 = (K_pct_lag2 * TBF_lag2 + k * league_k_trailing) / (TBF_lag2 + k))
      tr <- shr(tr); va <- shr(va)
    }
    mm <- build_matrix(tr, feats); Xva <- align(build_matrix(va, feats, mm$impute_vals)$X, colnames(mm$X))
    foldid <- if (cfg$grouped) group_folds(tr$PlayerId) else NULL
    if (cfg$binom) {
      yb <- cbind(tr$TBF - tr$K, tr$K)
      g <- cv.glmnet(mm$X, yb, family = "binomial", alpha = 0.5, foldid = foldid)
      p_glm <- as.numeric(predict(g, Xva, s = "lambda.min", type = "response"))
    } else {
      g <- cv.glmnet(mm$X, tr$K_pct, weights = tr$TBF, alpha = 0.5, family = "gaussian", foldid = foldid)
      p_glm <- as.numeric(predict(g, Xva, s = "lambda.min"))
    }
    p_tree <- if (cfg$xgb) xgb_grouped(mm$X, tr$K_pct, tr$TBF, tr$PlayerId, Xva)
              else if (cfg$grouped) gbm_grouped(mm$X, tr$K_pct, tr$TBF, tr$PlayerId, Xva)
              else gbm_valid_pick(mm$X, tr$K_pct, tr$TBF, Xva, va$K_pct, va$TBF)
    out[[as.character(vs)]] <- tibble(PlayerId = va$PlayerId, Season = vs, K_pct = va$K_pct, TBF = va$TBF,
                                      EB = eb(va, k, prior_va), glmnet = p_glm, tree = p_tree)
  }
  bind_rows(out)
}

fit_w <- function(dd) {
  best <- c(Inf, NA, NA, NA)
  for (a in seq(0, 1, .05)) for (b in seq(0, 1 - a, .05)) {
    p <- a * dd$EB + b * dd$glmnet + (1 - a - b) * dd$tree; r <- wrmse(p, dd$K_pct, dd$TBF)
    if (r < best[1]) best <- c(r, a, b, 1 - a - b)
  }
  best[2:4]
}
blend <- function(dd, w) w[1] * dd$EB + w[2] * dd$glmnet + w[3] * dd$tree

F <- FALSE; T <- TRUE
cands <- list(
  base     = list(feats = base_feats, grouped = F, shrinkK = F, binom = F, roleEB = F, xgb = F),
  honest   = list(feats = base_feats, grouped = T, shrinkK = F, binom = F, roleEB = F, xgb = F),
  shrinkK  = list(feats = base_feats, grouped = T, shrinkK = T, binom = F, roleEB = F, xgb = F),
  newfeat  = list(feats = c(base_feats, paste0(new_src, "_lag1")), grouped = T, shrinkK = T, binom = F, roleEB = F, xgb = F),
  binom    = list(feats = c(base_feats, paste0(new_src, "_lag1")), grouped = T, shrinkK = T, binom = T, roleEB = F, xgb = F),
  roleEB   = list(feats = c(base_feats, paste0(new_src, "_lag1")), grouped = T, shrinkK = T, binom = F, roleEB = T, xgb = F),
  xgbLogit = list(feats = c(base_feats, paste0(new_src, "_lag1")), grouped = T, shrinkK = T, binom = F, roleEB = F, xgb = T),
  # POST-HOC (added after the first run, still backtest-only): the robustness
  # config. Every component bounded in (0,1), original feature set.
  robust   = list(feats = base_feats, grouped = T, shrinkK = T, binom = T, roleEB = F, xgb = T)
)
only <- Sys.getenv("EXP_ONLY")
if (nzchar(only)) cands <- cands[strsplit(only, ",")[[1]]]

res <- list(); preds <- list()
for (nm in names(cands)) {
  cat("running", nm, "\n")
  pr <- run_candidate(cands[[nm]])
  w23 <- fit_w(pr %>% filter(Season == 2023))                     # honest: weights from 2023 only
  v24 <- pr %>% filter(Season == 2024); p24 <- blend(v24, w23)
  wall <- fit_w(pr); pall <- blend(pr, wall)                         # production-style pooled weights
  preds[[nm]] <- pr %>% mutate(blend_pooled = pall, cand = nm)
  res[[nm]] <- tibble(candidate = nm,
    EB_rmse = wrmse(pr$EB, pr$K_pct, pr$TBF), glmnet_rmse = wrmse(pr$glmnet, pr$K_pct, pr$TBF),
    tree_rmse = wrmse(pr$tree, pr$K_pct, pr$TBF),
    blend_pooled_rmse = wrmse(pall, pr$K_pct, pr$TBF), blend_pooled_logloss = logloss(pall, pr$K_pct, pr$TBF),
    honest_2024_rmse = wrmse(p24, v24$K_pct, v24$TBF), honest_2024_logloss = logloss(p24, v24$K_pct, v24$TBF),
    w_EB = wall[1], w_glmnet = wall[2], w_tree = wall[3])
}
res <- bind_rows(res)
print(res %>% mutate(across(where(is.numeric), ~ round(.x, 5))), width = 250)

# paired bootstrap (resampling pitchers) of the honest-2024 RMSE difference vs 'honest'
boot_diff <- function(a, b, n = 2000) {
  pa <- preds[[a]] %>% filter(Season == 2024); pb <- preds[[b]] %>% filter(Season == 2024)
  wa <- fit_w(preds[[a]] %>% filter(Season == 2023)); wb <- fit_w(preds[[b]] %>% filter(Season == 2023))
  ea <- blend(pa, wa); eb2 <- blend(pb, wb)
  idx <- replicate(n, sample(nrow(pa), replace = TRUE))
  diffs <- apply(idx, 2, function(i) wrmse(ea[i], pa$K_pct[i], pa$TBF[i]) - wrmse(eb2[i], pb$K_pct[i], pb$TBF[i]))
  c(diff = wrmse(ea, pa$K_pct, pa$TBF) - wrmse(eb2, pb$K_pct, pb$TBF), lo = quantile(diffs, .05), hi = quantile(diffs, .95))
}
cat("\nHonest-2024 RMSE difference vs 'honest' (negative = better), 90% paired bootstrap interval:\n")
for (nm in setdiff(names(cands), "honest")) if ("honest" %in% names(cands)) print(round(c(candidate = NA, boot_diff(nm, "honest")), 5) |> setNames(c(nm, "diff", "lo5", "hi95")))
saveRDS(list(res = res, preds = preds), paste0("exp_bucketA_results", if (nzchar(only)) "_subset" else "", ".rds"))
