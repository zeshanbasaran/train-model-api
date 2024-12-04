import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load your dataset
# Replace 'historical_data.csv' with your actual dataset file
data = pd.read_csv("nfl_games_with_confidence.csv")

# Encode categorical variables (team1, team2)
label_encoders = {
    "team1": LabelEncoder(),
    "team2": LabelEncoder(),
}

data["team1"] = label_encoders["team1"].fit_transform(data["team1"])
data["team2"] = label_encoders["team2"].fit_transform(data["team2"])

# Encode 'over_or_under' as binary
data["over_or_under"] = data["over_or_under"].apply(lambda x: 1 if x.lower() == "over" else 0)

# Add derived features
data["total_score"] = data["score1"] + data["score2"]
data["score_difference"] = abs(data["score1"] - data["score2"])

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

# Save the trained model and label encoders
joblib.dump(model, "model_with_scores.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")

print("Model training completed. Model and encoders saved.")
