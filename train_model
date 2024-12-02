# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
data = pd.read_csv("historical_games.csv")  # Replace with your dataset

# Feature Engineering
data["total_score"] = data["score1"] + data["score2"]
data["outcome"] = (data["total_score"] > data["over_under_line"]).astype(int)

# Prepare Features and Target
X = data[["team1", "team2", "over_under_line"]]
y = data["outcome"]

# Encode team names
encoder = LabelEncoder()
X["team1"] = encoder.fit_transform(X["team1"])
X["team2"] = encoder.fit_transform(X["team2"])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Save Model and Encoder
joblib.dump(model, "betting_model.pkl")
joblib.dump(encoder, "team_encoder.pkl")
