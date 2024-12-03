import requests
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Fetch data from your existing API
url = "https://nfl-api.onrender.com/games/all"
response = requests.get(url)

if response.status_code == 200:
    historical_data = response.json()  # Contains game data
else:
    raise Exception(f"Error fetching data: {response.status_code}")

# Preprocess the data
# Convert the API data into a DataFrame
games = []
for year, game_list in historical_data["all_years_data"].items():
    for game in game_list:
        games.append({
            "year": int(year),
            "team1": game["team1"],
            "team2": game["team2"],
            "score1": game["score1"],
            "score2": game["score2"],
            "total_score": game["score1"] + game["score2"],
        })

data = pd.DataFrame(games)

# Feature engineering: Create "over" target column
OVER_UNDER_THRESHOLD = 45  # Define your threshold for "over"
data["over"] = (data["total_score"] > OVER_UNDER_THRESHOLD).astype(int)

# Select features and target
X = data[["score1", "score2", "year"]]  # Features: team scores and year
y = data["over"]  # Target: whether total score is over the threshold

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy: {accuracy * 100:.2f}%")

# Save the model
joblib.dump(model, "ml/over_under_model.joblib")
print("Model saved as ml/over_under_model.joblib")
