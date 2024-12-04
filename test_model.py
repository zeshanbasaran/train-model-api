import joblib
import numpy as np

# Load the trained model and encoders
model = joblib.load("model_with_scores.pkl")
label_encoders = joblib.load("label_encoders.pkl")

def predict_confidence(team1, team2, year, over_or_under, amount, score1, score2):
    # Encode team names
    if team1 not in label_encoders["team1"].classes_:
        raise ValueError(f"Unknown team: {team1}")
    if team2 not in label_encoders["team2"].classes_:
        raise ValueError(f"Unknown team: {team2}")
    
    team1_encoded = label_encoders["team1"].transform([team1])[0]
    team2_encoded = label_encoders["team2"].transform([team2])[0]
    over_or_under_encoded = 1 if over_or_under.lower() == "over" else 0

    # Calculate derived features
    total_score = score1 + score2
    score_difference = abs(score1 - score2)

    # Prepare input features
    features = np.array([[team1_encoded, team2_encoded, year, over_or_under_encoded, amount, total_score, score_difference]])

    # Predict confidence level
    confidence = model.predict(features)[0]
    return confidence

# Example Input
team1 = "Ravens"  # Replace with your first team
team2 = "Bills"  # Replace with your second team
year = 2024       # Replace with the year of the game
over_or_under = "over"  # Replace with 'over' or 'under'
amount = 20     # Replace with the amount for the bet
score1 = 28       # Replace with the score for Team 1
score2 = 24       # Replace with the score for Team 2

try:
    confidence = predict_confidence(team1, team2, year, over_or_under, amount, score1, score2)
    print(f"Predicted Confidence Level for {over_or_under.capitalize()} bet: {confidence * 100:.2f}%")
except ValueError as e:
    print(f"Error: {e}")
