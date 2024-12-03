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
        total_score = game["score1"] + game["score2"]
        games.append({
            "year": int(year),
            "team1": game["team1"],
            "team2": game["team2"],
            "score1": game["score1"],
            "score2": game["score2"],
            "amount": total_score,  # Use total score as the "amount"
            "over": 1 if total_score > 45 else 0,  # Binary target for "over"
        })

data = pd.DataFrame(games)

# Select features and target
X = data[["year", "amount"]]  # Features: year and "amount" (total score)
y = data["over"]  # Target: whether the total score is over the threshold

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
