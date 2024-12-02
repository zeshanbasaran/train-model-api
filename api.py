# api.py
from fastapi import FastAPI
from pydantic import BaseModel
import joblib

# Load model and encoder
model = joblib.load("betting_model.pkl")
encoder = joblib.load("team_encoder.pkl")

app = FastAPI()

# Define input schema
class PredictionInput(BaseModel):
    team1: str
    team2: str
    over_under_line: float

# Define output schema
class PredictionOutput(BaseModel):
    prediction: str  # "Over" or "Under"
    probability: float  # Confidence score

@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    # Encode teams
    team1_encoded = encoder.transform([input_data.team1])[0]
    team2_encoded = encoder.transform([input_data.team2])[0]
    
    # Prepare features
    features = [[team1_encoded, team2_encoded, input_data.over_under_line]]
    
    # Make prediction
    probabilities = model.predict_proba(features)[0]
    prediction = "Over" if probabilities[1] > probabilities[0] else "Under"
    
    return PredictionOutput(
        prediction=prediction,
        probability=max(probabilities)
    )
