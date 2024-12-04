from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Create FastAPI app
app = FastAPI()

# Load the trained model and other dependencies
try:
    model = joblib.load("model_with_scores.pkl")
    label_encoders = joblib.load("label_encoders.pkl")  # For team name encoding
except Exception as e:
    raise RuntimeError(f"Error loading model or encoders: {e}")

# Define the request body structure
class BetInput(BaseModel):
    team1: str
    team2: str
    year: int
    over_or_under: str
    amount: float
    score1: int
    score2: int

@app.get("/")
def read_root():
    return {"message": "Welcome to the Betting Confidence API"}

@app.post("/calculate-confidence")
def calculate_confidence(input_data: BetInput):
    try:
        # Encode team names using label encoders
        if input_data.team1 not in label_encoders["team1"].classes_:
            raise HTTPException(status_code=400, detail=f"Unknown team: {input_data.team1}")
        if input_data.team2 not in label_encoders["team2"].classes_:
            raise HTTPException(status_code=400, detail=f"Unknown team: {input_data.team2}")

        team1_encoded = label_encoders["team1"].transform([input_data.team1])[0]
        team2_encoded = label_encoders["team2"].transform([input_data.team2])[0]

        # Encode 'over_or_under'
        over_or_under_encoded = 1 if input_data.over_or_under.lower() == "over" else 0

        # Calculate derived features
        total_score = input_data.score1 + input_data.score2
        score_difference = abs(input_data.score1 - input_data.score2)

        # Create feature array
        features = np.array([[team1_encoded, team2_encoded, input_data.year,
                              over_or_under_encoded, input_data.amount,
                              total_score, score_difference]])

        # Predict confidence level
        confidence = model.predict(features)[0]
        return {"confidence_level": round(confidence * 100, 2)}  # Confidence as a percentage

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error calculating confidence: {e}")
