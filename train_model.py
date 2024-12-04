import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
data = pd.read_csv("nfl_games_processed.csv")

# Encode categorical variables
label_encoders = {
    "team1": LabelEncoder(),
    "team2": LabelEncoder(),
}
data["team1"] = label_encoders["team1"].fit_transform(data["team1"])
data["team2"] = label_encoders["team2"].fit_transform(data["team2"])

# Encode 'over_or_under' as binary
data["over_or_under"] = data["over_or_under"].apply(lambda x: 1 if x == "over" else 0)

# Define features (X) and target (y)
X = data[["team1", "team2", "year", "over_or_under", "amount"]]
y = data["confidence_level"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
train_score = model.score(X_train, y_train)
test_score = model.score(X_test, y_test)
print(f"Training Score: {train_score:.2f}")
print(f"Testing Score: {test_score:.2f}")

# Save the trained model and encoders
joblib.dump(model, "model_with_scores.pkl")
joblib.dump(label_encoders, "label_encoders.pkl")
print("Model training completed. Model and encoders saved.")
