from fastapi import FastAPI, HTTPException
import joblib
import numpy as np
from pydantic import BaseModel
import requests

app = FastAPI()

# Load the pre-trained model
model = joblib.load("ml/over_under_model.joblib")

class PredictionRequest(BaseModel):
    team1: str
    team2: str
    year: int
    over_or_under: str  # "over" or "under"
    amount: float

@app.post("/predict")
def predict(request: PredictionRequest):
    # Validate input
    if request.over_or_under not in ["over", "under"]:
        raise HTTPException(status_code=400, detail="Invalid option. Must be 'over' or 'under'.")

    # Fetch team data from your existing API
    api_url = "https://nfl-api.onrender.com/games/all"
    response = requests.get(api_url)
    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Error fetching data from the existing API.")

    historical_data = response.json()

    # Example preprocessing (adjust based on your actual data)
    team1_scores = []
    team2_scores = []

    for year, games in historical_data["all_years_data"].items():
        if int(year) <= request.year:
            for game in games:
                if game["team1"] == request.team1 or game["team2"] == request.team1:
                    team1_scores.append(game["score1"] + game["score2"])
                if game["team1"] == request.team2 or game["team2"] == request.team2:
                    team2_scores.append(game["score1"] + game["score2"])

    # Calculate features for prediction
    avg_team1_score = np.mean(team1_scores) if team1_scores else 0
    avg_team2_score = np.mean(team2_scores) if team2_scores else 0
    input_features = np.array([[avg_team1_score, avg_team2_score, request.year]])

    # Make prediction
    probabilities = model.predict_proba(input_features)[0]
    confidence = probabilities[1] * 100  # Probability of "over"

    # Determine result
    if request.over_or_under == "over":
        confidence_level = confidence
    else:
        confidence_level = 100 - confidence

    return {
        "team1": request.team1,
        "team2": request.team2,
        "year": request.year,
        "over_or_under": request.over_or_under,
        "confidence_level": f"{confidence_level:.2f}%"
    }
