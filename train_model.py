import requests
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import pandas as pd

# Fetch data from your existing API
url = "https://your-existing-api.onrender.com/games/all"
response = requests.get(url)

if response.status_code == 200:
    historical_data = response.json()
else:
    raise Exception(f"Error fetching data: {response.status_code}")

# Preprocess data
games = []
for year, year_games in historical_data["all_years_data"].items():
    for game in year_games:
        games.append({
            "year": int(year),
            "score1": game["score1"],
            "score2": game["score2"],
            "total_score": game["score1"] + game["score2"],
            "over": game["score1"] + game["score2"] > 45  # Example threshold
        })

df = pd.DataFrame(games)

# Split features and target
X = df[["score1", "score2", "year"]]
y = df["over"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
print(f"Model accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Save model
joblib.dump(model, "ml/over_under_model.joblib")
print("Model saved to ml/over_under_model.joblib")

