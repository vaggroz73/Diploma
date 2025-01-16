import json
import os
import requests
from bs4 import BeautifulSoup
import pandas as pd

# Base URL for scraping
BASE_LEAGUE_URL = "https://understat.com/league/"

# Define the directory to save the data
CWD = os.getcwd()
DIR = os.path.join(CWD, "datasets")

def scrape_script_tags_season(season):
    """
    Scrapes all <script> tags for a given league/season URL.
    """
    URL = BASE_LEAGUE_URL + season
    print(f"Scraping URL: {URL}")  # Debugging: Print the URL being scraped
    response = requests.get(URL)
    if response.status_code != 200:
        print(f"Error: Unable to fetch URL {URL} - Status code: {response.status_code}")
        return None
    soup = BeautifulSoup(response.content, "lxml")
    soup_scripts = soup.find_all("script")
    return soup_scripts

def generate_players_dict(season):
    """
    Extracts player data from the script tags and converts it to a dictionary.
    """
    soup_scripts = scrape_script_tags_season(season)
    if not soup_scripts:
        return None
    
    # Get the specific script containing player data
    script = soup_scripts[3].string
    start_index = script.index("('") + 2
    end_index = script.index("')")
    json_string = script[start_index:end_index]
    json_string = json_string.encode("utf8").decode("unicode_escape")
    players_dict = json.loads(json_string)
    return players_dict

def generate_players_csv(season):
    """
    Fetches player data for the specified season and saves it as a CSV.
    """
    players_dict = generate_players_dict(season)
    if players_dict is None:
        print(f"No data available for season: {season}")
        return

    # Convert dictionary to DataFrame
    players_df = pd.DataFrame.from_dict(players_dict)
    
    # Define output file and directory
    league, year_range = season.split("/")
    output_file = f"players_{league.lower()}_{year_range}.csv"
    output_dir = os.path.join(DIR, league.lower())
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, output_file)
    
    # Save DataFrame to CSV
    players_df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

# Define the season (EPL 2021-2022)
season = "Serie_A/2017"

# Generate and save the player data
generate_players_csv(season)

