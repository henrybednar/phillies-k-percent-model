# tests/test_pipeline.R — pressure tests for the K% pipeline.
# Run from the project folder AFTER the merge and model scripts:
#   Rscript tests/test_pipeline.R
#
# Test A  Edge cases: injects synthetic pitchers (0-for-5, 0-for-35, 3-for-3,
#         5-for-5, extreme and completely missing telemetry, missing Stuff+,
#         missing Age, an MiLB newcomer with no MiLB stats, a duplicated row)
#         into a copy of the merged data, reruns the
#         model in a temp folder, and checks every prediction is finite and
#         inside (0, 1), and that a duplicate key or a stray text token in
#         a numeric column is rejected with a clear message.
# Test B  Calibration: Brier score and binomial log-loss vs. naive baselines,
#         plus a reliability table (predicted vs. observed K rate by decile of
#         prediction) on the real 2025 predictions.
# Developed with assistance from Claude (Anthropic), https://claude.ai

suppressMessages({library(readr); library(dplyr)})
model_script <- normalizePath("k_percentage_model.R")
stopifnot(file.exists("k_2026_final_merged.csv"), file.exists("k_2026_2025_predictions.csv"))
rscript <- file.path(R.home("bin"), "Rscript")
failures <- character(0)
check <- function(ok, msg) {
  cat(if (ok) "  PASS " else "  FAIL ", msg, "\n", sep = "")
  if (!ok) failures <<- c(failures, msg)
}

# ── Test A: edge cases ───────────────────────────────────────────────────────
cat("\n=== Test A: synthetic edge cases ===\n")
m <- read_csv("k_2026_final_merged.csv", show_col_types = FALSE, guess_max = 10000)
template <- m[1, ]; template[1, ] <- NA
syn <- function(id, season, tbf, k, name, prior_pitching = TRUE, ...) {
  r <- template
  r$PlayerId <- id; r$Season <- season; r$Name <- name; r$Team <- "SYN"
  r$TBF <- tbf; r$`K%` <- k; r$`Stuff+` <- 100; r$Age <- 26
  r$has_prior_pitching <- prior_pitching; r$has_prior_batting <- FALSE
  extra <- list(...); for (nm in names(extra)) r[[nm]] <- extra[[nm]]
  r
}
cases <- bind_rows(
  syn(9e6 + 1, 2024, 5, 0, "SYN 0-for-5"),            syn(9e6 + 1, 2025, 5, 0, "SYN 0-for-5"),
  syn(9e6 + 2, 2024, 3, 1, "SYN 3-for-3"),            syn(9e6 + 2, 2025, 3, 1, "SYN 3-for-3"),
  syn(9e6 + 3, 2024, 3, 1, "SYN extreme-high", whiff_percent = 95, z_swing_miss_percent = 95,
      fastball_avg_speed = 108, `Stuff+` = 400, csw_percent = 90),
  syn(9e6 + 3, 2025, 3, 1, "SYN extreme-high"),
  syn(9e6 + 4, 2024, 900, 0, "SYN extreme-low", whiff_percent = 0, z_swing_miss_percent = 0,
      fastball_avg_speed = 60, `Stuff+` = -300),
  syn(9e6 + 4, 2025, 900, 0.01, "SYN extreme-low"),
  syn(9e6 + 5, 2024, 200, 0.2, "SYN no telemetry"),  syn(9e6 + 5, 2025, 200, 0.2, "SYN no telemetry"),
  syn(9e6 + 6, 2025, 40, 0.25, "SYN MiLB newcomer, no MiLB stats", prior_pitching = FALSE, milb_year = 2024),
  syn(9e6 + 7, 2024, 35, 0, "SYN 0-for-35"),           syn(9e6 + 7, 2025, 35, 0.1, "SYN 0-for-35"),
  syn(9e6 + 8, 2024, 5, 1, "SYN 5-for-5"),             syn(9e6 + 8, 2025, 5, 0.2, "SYN 5-for-5"),
  syn(9e6 + 9, 2024, 150, 0.2, "SYN missing Stuff+", `Stuff+` = NA),
  syn(9e6 + 9, 2025, 150, 0.2, "SYN missing Stuff+", `Stuff+` = NA),
  syn(9e6 + 10, 2024, 150, 0.2, "SYN missing Age", Age = NA),
  syn(9e6 + 10, 2025, 150, 0.2, "SYN missing Age", Age = NA)
)
syn_ids <- unique(cases$PlayerId)

run_model_on <- function(dat) {
  dir <- file.path(tempdir(), paste0("ktest_", as.integer(runif(1) * 1e9)))
  dir.create(dir)
  write_csv(dat, file.path(dir, "k_2026_final_merged.csv"))
  file.copy(model_script, file.path(dir, "model.R"))
  old <- setwd(dir); on.exit(setwd(old))
  status <- suppressWarnings(system2(rscript, "model.R", stdout = "log.txt", stderr = "log.txt"))
  list(status = status, dir = dir,
       preds = if (file.exists("k_2026_2025_predictions.csv")) read_csv("k_2026_2025_predictions.csv", show_col_types = FALSE))
}

r <- run_model_on(bind_rows(m, cases))
check(r$status == 0, "model runs with synthetic edge cases injected")
if (!is.null(r$preds)) {
  sp <- r$preds %>% filter(PlayerId %in% syn_ids)
  print(sp %>% select(Name, bucket, TBF, pred_final, pred_lower, pred_upper), width = 200)
  check(nrow(sp) == sum(cases$Season == 2025), "every synthetic 2025 pitcher gets a prediction")
  check(all(is.finite(r$preds$pred_final)), "all predictions finite (no NA/Inf)")
  check(all(r$preds$pred_final > 0 & r$preds$pred_final < 1), "all predictions strictly inside (0, 1)")
  check(all(r$preds$pred_final_raw > 0 & r$preds$pred_final_raw < 1), "raw (pre-calibration) predictions inside (0, 1) without clipping")
  check(all(r$preds$pred_lower >= 0 & r$preds$pred_upper <= 1 &
            r$preds$pred_lower <= r$preds$pred_final & r$preds$pred_final <= r$preds$pred_upper),
        "intervals ordered and inside [0, 1]")
  # A 3-TBF history carries almost no information, so the forecast should
  # sit near a pitcher with average everything (the no-telemetry case), not
  # jump toward 100%. Tolerance: 3 K% points.
  p33 <- sp$pred_final[sp$Name == "SYN 3-for-3"]; p05 <- sp$pred_final[sp$Name == "SYN 0-for-5"]
  ref <- sp$pred_final[sp$Name == "SYN no telemetry"]
  check(abs(p33 - ref) < 0.03 && abs(p05 - ref) < 0.03,
        sprintf("tiny-sample histories stay near average (3-for-3 %.3f, 0-for-5 %.3f, reference %.3f)", p33, p05, ref))
  p55 <- sp$pred_final[sp$Name == "SYN 5-for-5"]
  check(abs(p55 - ref) < 0.03, sprintf("a 1.000 K rate over 5 TBF stays near average (pred %.3f)", p55))
  # 0 strikeouts in 35 TBF is real evidence of a low K rate, but not of a
  # zero one: the forecast should drop below average and stay well above 0.
  p035 <- sp$pred_final[sp$Name == "SYN 0-for-35"]
  check(p035 < ref && p035 > 0.10, sprintf("0-for-35 drops below average but not toward 0 (pred %.3f)", p035))
  p_nostuff <- sp$pred_final[sp$Name == "SYN missing Stuff+"]; p_noage <- sp$pred_final[sp$Name == "SYN missing Age"]
  check(all(c(p_nostuff, p_noage) > 0.15 & c(p_nostuff, p_noage) < 0.30),
        sprintf("missing Stuff+ or Age is imputed, not fatal (preds %.3f, %.3f for a 20%% K history)", p_nostuff, p_noage))
}

dup <- bind_rows(m, m %>% filter(Season == 2024) %>% slice(1))
r_dup <- run_model_on(dup)
check(r_dup$status != 0, "a duplicated player-season row stops the model instead of leaking the target")

# A stray text token in a numeric column must stop the run at load time,
# naming the column and the value, not fail deep inside model fitting.
txt <- m %>% mutate(whiff_percent = as.character(whiff_percent))
txt$whiff_percent[which(!is.na(txt$whiff_percent))[5]] <- "--"
r_txt <- run_model_on(txt)
log_txt <- paste(readLines(file.path(r_txt$dir, "log.txt"), warn = FALSE), collapse = "\n")
check(r_txt$status != 0 && grepl("Non-numeric values in input column(s): whiff_percent", log_txt, fixed = TRUE),
      "a stray \"--\" in a numeric column stops the run with a message naming the column")

# ── Test B: calibration of the real 2025 predictions ─────────────────────────
cat("\n=== Test B: calibration of 2025 predictions ===\n")
p <- read_csv("k_2026_2025_predictions.csv", show_col_types = FALSE)
f <- read_csv("k_2026_features.csv", show_col_types = FALSE, guess_max = 10000) %>%
  filter(Season == 2025) %>% select(PlayerId, league_k_trailing)
p <- p %>% left_join(f, by = "PlayerId")

# Per-batter scores: each batter faced is a Bernoulli trial with prob. p.
brier   <- function(q, y, w) weighted.mean(y * (1 - q)^2 + (1 - y) * q^2, w)
logloss <- function(q, y, w) { q <- pmin(pmax(q, 1e-6), 1 - 1e-6); -weighted.mean(y * log(q) + (1 - y) * log(1 - q), w) }
scores <- tibble(
  prediction = c("Final model", "League average (naive)", "Overall 2025 rate (oracle constant)"),
  brier = c(brier(p$pred_final, p$K_pct, p$TBF), brier(p$league_k_trailing, p$K_pct, p$TBF),
            brier(rep(weighted.mean(p$K_pct, p$TBF), nrow(p)), p$K_pct, p$TBF)),
  logloss = c(logloss(p$pred_final, p$K_pct, p$TBF), logloss(p$league_k_trailing, p$K_pct, p$TBF),
              logloss(rep(weighted.mean(p$K_pct, p$TBF), nrow(p)), p$K_pct, p$TBF)))
print(scores %>% mutate(across(where(is.numeric), ~ round(.x, 5))))
check(scores$logloss[1] < scores$logloss[2], "model log-loss beats the league-average prediction")

rel <- p %>%
  mutate(decile = ntile(pred_final, 10)) %>%
  group_by(decile) %>%
  summarise(pitchers = n(), batters = sum(TBF),
            predicted = weighted.mean(pred_final, TBF),
            observed = sum(K_pct * TBF) / sum(TBF), .groups = "drop") %>%
  mutate(se = sqrt(predicted * (1 - predicted) / batters),   # binomial SE of the observed rate
         z = (observed - predicted) / se)
cat("\nReliability by decile of prediction (TBF-weighted):\n")
print(rel %>% mutate(across(c(predicted, observed, se), ~ round(.x, 4)), z = round(z, 2)))
# Decile z-scores ignore model error, so they overstate miscalibration; flag
# only large departures, and check the tails specifically.
check(all(abs(rel$z[rel$decile %in% c(1, 10)]) < 3), "tails (bottom/top decile) calibrated within 3 binomial SE")
slope <- coef(lm(observed ~ predicted, data = rel, weights = batters))[2]
cat(sprintf("\nCalibration slope (observed on predicted, deciles): %.2f  (1 = perfectly calibrated spread)\n", slope))

cat("\n", if (length(failures)) paste("FAILED:", length(failures), "check(s)") else "All checks passed.", "\n")
quit(status = as.integer(length(failures) > 0))
