install.packages("devtools")
library(devtools)

devtools::install_github("JaseZiv/worldfootballR")

library(worldfootballR)


# The Big 5 Euro League Players
big5_player_possession <- fb_big5_advanced_season_stats(season_end_year= 2025, stat_type= "possession", team_or_player= "player")
dplyr::glimpse(big5_player_possession)

View(big5_player_possession)
write.csv(big5_player_possession, "big5_player_pos_2024.csv", row.names = FALSE)
getwd()

