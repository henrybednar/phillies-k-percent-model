# data_merge.R
#
# Developed with assistance from Claude (Anthropic), https://claude.ai
#
# Step 1 of 2. Run this first; it writes k_2026_final_merged.csv, which
# k_percentage_model.R reads.
#
# Combines k_2026.csv with all four Savant pulls (every pull is 2024 or
# earlier, so no data from 2025 Opening Day onward enters the merge):
#   - MLBBaseballSavantPitcherData2015-2024.csv: same-season merge, plus each
#     player's first MLB pitching season (used for as-of routing)
#   - MLBBaseballSavantHittersData2015-2024.csv: each player's first season
#     with >= hitting_pa_floor PA (used for as-of position-player routing)
#   - MiLBBaseballSavantPitcherData2021-2024.csv: prefixed with milb_ to
#     avoid name collisions with the MLB pitching columns, and joined
#     as-of, each row gets that player's most recent MiLB season strictly
#     before its own MLB season, never a same-year or future match
#   - MLBPitchersBallsStrikes2015-2024.csv: pitch-outcome counts, used to
#     compute csw_percent (called + swinging strike %) -- same-season
#     merge, joined directly on player_id/year, no name matching needed

library(readr)
library(dplyr)

data_dir <- "."

# Min single-season PA for a hitting season to count as position-player
# evidence. Kept at the Savant pull's own floor (25 PA): a 100 PA floor
# misses 23 real position players who pitched in 2021-2024 (bench/utility
# players with 25-99 PA seasons) and routes them to the league-mean rule.
hitting_pa_floor <- 25

k26     <- read_csv(file.path(data_dir, "k_2026.csv"), show_col_types = FALSE)
mlb_p   <- read_csv(file.path(data_dir, "MLBBaseballSavantPitcherData2015-2024.csv"), show_col_types = FALSE)
mlb_h   <- read_csv(file.path(data_dir, "MLBBaseballSavantHittersData2015-2024.csv"), show_col_types = FALSE)
milb_p  <- read_csv(file.path(data_dir, "MiLBBaseballSavantPitcherData2021-2024.csv"), show_col_types = FALSE)
balls_strikes <- read_csv(file.path(data_dir, "MLBPitchersBallsStrikes2015-2024.csv"), show_col_types = FALSE)

stopifnot(!anyDuplicated(k26[c("PlayerId", "Season")]))

# Hard stop: nothing from the 2025 season onward may enter the pipeline. The
# only 2025 rows allowed anywhere are k_2026.csv's own, which hold the
# targets being predicted (their K%, TBF, and Stuff+ are never model inputs).
latest_season <- c(`MLB pitching` = max(mlb_p$year), `MLB hitting` = max(mlb_h$year),
                   `MiLB pitching` = max(milb_p$year), `pitch outcomes` = max(balls_strikes$year))
if (any(latest_season > 2024)) {
  stop("Season 2025+ data found in: ",
       paste(names(latest_season)[latest_season > 2024], collapse = ", "),
       ". Only seasons through 2024 may be used.", call. = FALSE)
}

# ---- MLB pitching: same-season merge ----
# Only duplicate keys in the file are Pat Venditte's L/R split rows
# (2015-2020), which never match a k_2026 row (2021+).
mlb_p_dedup <- mlb_p %>% distinct(player_id, year, .keep_all = TRUE)
mlb_p_clean <- mlb_p_dedup %>%
  select(-p_opp_batting_avg, -p_opp_on_base_avg, -`last_name, first_name`) %>%
  mutate(obp_allowed = on_base_plus_slg - slg_percent)

merged <- k26 %>%
  left_join(mlb_p_clean, by = c("PlayerId" = "player_id", "Season" = "year"))

# ---- As-of history flags: only seasons strictly before the row's own ----
# Pooling every year (including the season being predicted) would mark a
# 2023 debut pitcher as having a track record but a 2025 debut pitcher as
# not, so rookies would be routed differently in the backtest than in 2025.
first_pitch <- mlb_p_clean %>%
  group_by(player_id) %>%
  summarise(first_pitch_year = min(year), .groups = "drop")
first_bat <- mlb_h %>%
  filter(pa >= hitting_pa_floor) %>%
  group_by(player_id) %>%
  summarise(first_bat_year = min(year), .groups = "drop")

merged <- merged %>%
  left_join(first_pitch, by = c("PlayerId" = "player_id")) %>%
  left_join(first_bat,   by = c("PlayerId" = "player_id")) %>%
  mutate(
    has_prior_pitching = !is.na(first_pitch_year) & first_pitch_year < Season,
    has_prior_batting  = !is.na(first_bat_year)   & first_bat_year   < Season
  )

# ---- MiLB pitching: prefix columns, as-of join on most recent prior year ----
id_cols <- c("player_id", "player_name", "year")
milb_renamed <- milb_p %>%
  rename_with(~ paste0("milb_", .x), .cols = -all_of(id_cols)) %>%
  rename(milb_year = year)

candidates <- merged %>%
  distinct(PlayerId, Season) %>%
  inner_join(milb_renamed, by = c("PlayerId" = "player_id"),
             relationship = "many-to-many") %>%  # each player has several seasons on both sides
  filter(milb_year < Season) %>%
  group_by(PlayerId, Season) %>%
  slice_max(milb_year, n = 1, with_ties = FALSE) %>%
  ungroup() %>%
  select(-player_name)

merged <- merged %>% left_join(candidates, by = c("PlayerId", "Season"))

# ---- CSW%: called + swinging strike %, from pitch-outcome counts ----
# total_pitches = p_ball + p_total_strike: p_total_strike already covers
# every non-ball pitch (called, foul, foul tip, swinging, in-play), so
# adding p_ball to it gives the full pitch count.
balls_strikes <- balls_strikes %>%
  distinct() %>%  # Pat Venditte's switch-pitcher rows are exact duplicates
  rename(PlayerId = player_id, Season = year) %>%
  mutate(csw_percent = 100 * (p_called_strike + p_swinging_strike) / (p_ball + p_total_strike)) %>%
  select(PlayerId, Season, csw_percent)

merged <- merged %>% left_join(balls_strikes, by = c("PlayerId", "Season"))

stopifnot(nrow(merged) == nrow(k26), !anyDuplicated(merged[c("PlayerId", "Season")]))

# ---- Diagnostics ----
cat("Final rows:", nrow(merged), " columns:", ncol(merged), "\n")
cat("Matched to MLB pitching (same season):", sum(!is.na(merged$p_game)), "of", nrow(merged), "\n")
cat("Rows with an as-of MiLB match:", sum(!is.na(merged$milb_year)), "\n")
cat("Rows with csw_percent:", sum(!is.na(merged$csw_percent)), "\n\n")

cat("As-of history flags by season (prior seasons only):\n")
merged %>%
  group_by(Season) %>%
  summarise(n = n(),
            prior_pitching = sum(has_prior_pitching),
            prior_batting_no_pitching = sum(has_prior_batting & !has_prior_pitching),
            neither_with_milb = sum(!has_prior_pitching & !has_prior_batting & !is.na(milb_year)),
            neither_no_milb = sum(!has_prior_pitching & !has_prior_batting & is.na(milb_year)),
            .groups = "drop") %>%
  print()

write_csv(merged, file.path(data_dir, "k_2026_final_merged.csv"))
cat("\nWrote k_2026_final_merged.csv\n")
