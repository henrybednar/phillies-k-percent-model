# build_final_merge.R
#
# Developed with assistance from Claude (Anthropic), https://claude.ai
#
# Combines k_2026.csv with all four final Savant pulls into one file:
#   - MLBBaseballSavantPitcherData20152024.csv: same-season merge (as before)
#   - MLBBaseballSavantHittersData20152024.csv: collapsed to a single
#     has_batting_record flag, PlayerId match only, no season alignment
#     needed since it's just presence/absence evidence
#   - MiLBBaseballSavantPitcherData20212024.csv: prefixed with milb_ to
#     avoid name collisions with the MLB pitching columns, and joined
#     as-of, each row gets that player's most recent MiLB season strictly
#     before its own MLB season, never a same-year or future match
#   - MLBPitchersBallsStrikes20152024.csv: pitch-outcome counts, used to
#     compute csw_percent (called + swinging strike %) -- same-season
#     merge, joined directly on player_id/year, no name matching needed

library(readr)
library(dplyr)

data_dir <- "."
k26     <- read_csv(file.path(data_dir, "k_2026.csv"), show_col_types = FALSE)
mlb_p   <- read_csv(file.path(data_dir, "MLBBaseballSavantPitcherData20152024.csv"), show_col_types = FALSE)
mlb_h   <- read_csv(file.path(data_dir, "MLBBaseballSavantHittersData20152024.csv"), show_col_types = FALSE)
milb_p  <- read_csv(file.path(data_dir, "MiLBBaseballSavantPitcherData20212024.csv"), show_col_types = FALSE)
balls_strikes <- read_csv(file.path(data_dir, "MLBPitchersBallsStrikes20152024.csv"), show_col_types = FALSE)

# ---- MLB pitching: same steps as the original merge ----
mlb_p_dedup <- mlb_p %>% distinct(player_id, year, .keep_all = TRUE)
mlb_p_clean <- mlb_p_dedup %>%
  select(-p_opp_batting_avg, -p_opp_on_base_avg, -`last_name, first_name`) %>%
  mutate(obp_allowed = on_base_plus_slg - slg_percent)

merged <- k26 %>%
  left_join(mlb_p_clean, by = c("PlayerId" = "player_id", "Season" = "year"))

pitcher_ids <- unique(mlb_p_clean$player_id)
merged <- merged %>%
  mutate(
    has_pitching_track_record = PlayerId %in% pitcher_ids,
    likely_position_player = TBF > 0 & !has_pitching_track_record
  )

# ---- MLB hitting: presence-only flag, no season key ----
batter_ids <- unique(mlb_h$player_id)
merged <- merged %>% mutate(has_batting_record = PlayerId %in% batter_ids)

# ---- MiLB pitching: prefix columns, as-of join on most recent prior year ----
id_cols <- c("player_id", "player_name", "year")
milb_renamed <- milb_p %>%
  rename_with(~ paste0("milb_", .x), .cols = -all_of(id_cols)) %>%
  rename(milb_year = year)

candidates <- merged %>%
  distinct(PlayerId, Season) %>%
  inner_join(milb_renamed, by = c("PlayerId" = "player_id")) %>%
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

# ---- Diagnostics ----
cat("Final rows:", nrow(merged), " columns:", ncol(merged), "\n")
cat("Matched to MLB pitching (same season):", sum(!is.na(merged$p_game)), "of", nrow(merged), "\n")
cat("has_batting_record TRUE:", sum(merged$has_batting_record), "\n")
cat("Rows with an as-of MiLB match:", sum(!is.na(merged$milb_year)), "\n")
cat("Rows with csw_percent:", sum(!is.na(merged$csw_percent)), "\n\n")

cat("Coverage among the 'no MLB pitching record' rows, by season:\n")
merged %>%
  filter(likely_position_player) %>%
  group_by(Season) %>%
  summarise(n = n(),
            has_batting = sum(has_batting_record),
            has_milb = sum(!is.na(milb_year)),
            has_neither = sum(!has_batting_record & is.na(milb_year)),
            .groups = "drop") %>%
  print()

write_csv(merged, file.path(data_dir, "k_2026_final_merged.csv"))
cat("\nWrote k_2026_final_merged.csv\n")