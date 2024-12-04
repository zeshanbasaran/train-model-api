import requests
import pandas as pd
import random

# API endpoint
API_URL = "https://nfl-api.onrender.com/games/all"

# Fetch data from the API
response = requests.get(API_URL)
if response.status_code == 200:
    raw_data = response.json()
else:
    raise Exception(f"Failed to fetch data from API: {response.status_code}")

# Process the data
games = []
for year, year_games in raw_data["all_years_data"].items():
    for game in year_games:
        total_score = game["score1"] + game["score2"]
        score_diff = abs(game["score1"] - game["score2"])
        confidence_level = None

        # Generate synthetic confidence level based on assumptions
        if total_score > 60:  # High-scoring games favor "over"
            confidence_level = round(random.uniform(0.7, 1.0), 2)
        elif score_diff < 10:  # Close games favor "under"
            confidence_level = round(random.uniform(0.5, 0.8), 2)
        else:  # Default medium confidence
            confidence_level = round(random.uniform(0.3, 0.7), 2)

        games.append({
            "team1": game["team1"],
            "team2": game["team2"],
            "year": int(year),
            "score1": game["score1"],
            "score2": game["score2"],
            "winner": game["winner"],
            "over_or_under": "over" if total_score > 50 else "under",
            "amount": abs(total_score - 50),
            "confidence_level": confidence_level
        })

# Convert to DataFrame
df = pd.DataFrame(games)

# Save to CSV
csv_file = "nfl_games_with_confidence.csv"
df.to_csv(csv_file, index=False)
csv_file
