from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the trained model and encoders
try:
    model = joblib.load("model_with_scores.pkl")
    label_encoders = joblib.load("label_encoders.pkl")
except Exception as e:
    raise RuntimeError(f"Error loading model or encoders: {e}")

# Define input schema
class BetInput(BaseModel):
    team1: str
    team2: str
    year: int
    over_or_under: str
    amount: float

@app.get("/")
def read_root():
    return {"message": "Welcome to the Over/Under Confidence Prediction API"}

import logging

logging.basicConfig(level=logging.INFO)

@app.post("/calculate-confidence")
def calculate_confidence(input_data: BetInput):
    try:
        logging.info(f"Received input: {input_data}")

        # Encode team names
        if input_data.team1 not in label_encoders["team1"].classes_:
            raise HTTPException(status_code=400, detail=f"Unknown team: {input_data.team1}")
        if input_data.team2 not in label_encoders["team2"].classes_:
            raise HTTPException(status_code=400, detail=f"Unknown team: {input_data.team2}")

        team1_encoded = label_encoders["team1"].transform([input_data.team1])[0]
        team2_encoded = label_encoders["team2"].transform([input_data.team2])[0]
        over_or_under_encoded = 1 if input_data.over_or_under.lower() == "over" else 0

        # Prepare feature array
        features = np.array([[team1_encoded, team2_encoded, input_data.year, over_or_under_encoded, input_data.amount]])

        # Predict confidence level
        confidence = model.predict(features)[0]
        return {"confidence_level": round(confidence * 100, 2)}

    except Exception as e:
        logging.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {e}")

