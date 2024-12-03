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
    amount: float  # Over/under value

@app.post("/predict")
def predict(request: PredictionRequest):
    # Validate input
    if request.over_or_under not in ["over", "under"]:
        raise HTTPException(status_code=400, detail="Invalid option. Must be 'over' or 'under'.")

    # Prepare the data for the model
    input_features = pd.DataFrame([{
        "year": request.year,
        "amount": request.amount
    }])

    try:
        # Predict probabilities
        probabilities = model.predict_proba(input_features)[0]

        # Calculate confidence
        if request.over_or_under == "over":
            confidence = probabilities[1] * 100  # Probability of "over"
        else:
            confidence = probabilities[0] * 100  # Probability of "under"

        return {"confidence": confidence}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {e}")

