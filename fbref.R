install.packages("devtools")
library(devtools)

devtools::install_github("JaseZiv/worldfootballR")

library(worldfootballR)


mapped_players <- player_dictionary_mapping()
dplyr::glimpse(mapped_players)

#football players list
print(player_dictionary_mapping(), n = Inf)

View(player_dictionary_mapping())


#Premier League Liverpool
fb_league_urls(country = "ENG", gender = "M", season_end_year = 2015, tier = '2nd')

fb_teams_urls("https://fbref.com/en/comps/9/Premier-League-Stats")
View(fb_teams_urls("https://fbref.com/en/comps/9/Premier-League-Stats"))

#Player URLs
man_city_players <- fb_player_urls("https://fbref.com/en/squads/b8fd03ef/Manchester-City-Stats")

View(man_city_players)

#Big 5 Leagues
#----- Get data for big five leagues for TEAMS -----#
big5_team_shooting <- fb_big5_advanced_season_stats(season_end_year= 2017, stat_type= "shooting", team_or_player= "team")
dplyr::glimpse(big5_team_shooting)

View(big5_team_shooting)

write.csv(big5_team_shooting, "big5_team_shooting.csv", row.names = FALSE, fileEncoding = "UTF-8")
getwd()


#----- Get data for big five leagues for PLAYERS -----#
big5_player_shooting <- fb_big5_advanced_season_stats(season_end_year= 2019, stat_type= "shooting", team_or_player= "player")
dplyr::glimpse(big5_player_shooting)


View(big5_player_shooting)

write.csv(big5_player_shooting, "big5_player_shooting.csv", row.names = FALSE)
getwd()


# The Big 5 Euro League Players
big5_player_possession <- fb_big5_advanced_season_stats(season_end_year= 2025, stat_type= "possession", team_or_player= "player")
dplyr::glimpse(big5_player_possession)

View(big5_player_possession)
write.csv(big5_player_possession, "big5_player_pos_2024.csv", row.names = FALSE)
getwd()


