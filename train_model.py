import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# API endpoint
API_URL = "https://nfl-api.onrender.com/games/all"

def fetch_data_from_api():
    try:
        response = requests.get(API_URL)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from API: {e}")
        return None

def preprocess_data(raw_data):
    # Flatten the JSON data into a DataFrame
    games = []
    for year, year_games in raw_data["all_years_data"].items():
        for game in year_games:
            games.append({
                "team1": game["team1"],
                "team2": game["team2"],
                "year": int(year),
                "score1": game["score1"],
                "score2": game["score2"],
                "winner": game["winner"],
                "over_or_under": "over" if (game["score1"] + game["score2"]) > 50 else "under",  # Example rule
                "amount": abs((game["score1"] + game["score2"]) - 50),  # Example amount
                "confidence_level": 0.7  # Replace with actual target if available
            })

    # Convert to DataFrame
    df = pd.DataFrame(games)

    # Encode categorical variables (team1, team2)
    label_encoders = {
        "team1": LabelEncoder(),
        "team2": LabelEncoder(),
    }
    df["team1"] = label_encoders["team1"].fit_transform(df["team1"])
    df["team2"] = label_encoders["team2"].fit_transform(df["team2"])

    # Encode 'over_or_under' as binary
    df["over_or_under"] = df["over_or_under"].apply(lambda x: 1 if x == "over" else 0)

    # Add derived features
    df["total_score"] = df["score1"] + df["score2"]
    df["score_difference"] = abs(df["score1"] - df["score2"])

    return df, label_encoders

def train_model(data, label_encoders):
    # Define features (X) and target (y)
    X = data[[
        "team1", "team2", "year", "over_or_under", "amount",
        "total_score", "score_difference"
    ]]
    y = data["confidence_level"]

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train a Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)

    print(f"Training Score: {train_score:.2f}")
    print(f"Testing Score: {test_score:.2f}")

    # Save the trained model and label encoders
    joblib.dump(model, "model_with_scores.pkl")
    joblib.dump(label_encoders, "label_encoders.pkl")

    print("Model training completed. Model and encoders saved.")

def main():
    # Fetch data from API
    raw_data = fetch_data_from_api()
    if not raw_data:
        return

    # Preprocess the data
    data, label_encoders = preprocess_data(raw_data)

    # Train the model
    train_model(data, label_encoders)

if __name__ == "__main__":
    main()
