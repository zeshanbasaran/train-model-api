from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Float, Table
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import joblib

# Load the ML model and encoder
model = joblib.load("betting_model.pkl")
encoder = joblib.load("team_encoder.pkl")

# Database setup
DATABASE_URL = "postgresql://nfl_data_user:f943VkuXHasDOiJJVqd17V6zduxP5uv4@dpg-ct70st3tq21c73ed1p4g-a.oregon-postgres.render.com/nfl_data"
Base = declarative_base()
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# FastAPI app
app = FastAPI()

# Database tables
class Game(Base):
    __tablename__ = "games"
    id = Column(Integer, primary_key=True, index=True)
    year = Column(Integer, nullable=False)
    team1 = Column(String, nullable=False)
    team2 = Column(String, nullable=False)
    score1 = Column(Integer, nullable=False)
    score2 = Column(Integer, nullable=False)
    winner = Column(String, nullable=False)


class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    team1 = Column(String, nullable=False)
    team2 = Column(String, nullable=False)
    over_under_line = Column(Float, nullable=False)
    prediction = Column(String, nullable=False)
    probability = Column(Float, nullable=False)


# Create tables in the database
Base.metadata.create_all(bind=engine)

# Input and output schemas
class PredictionInput(BaseModel):
    team1: str
    team2: str
    over_under_line: float


class PredictionOutput(BaseModel):
    prediction: str  # "Over" or "Under"
    probability: float  # Confidence score


# Helper function to simulate game results
def simulate_game(team1, team2, year):
    team1_score = randint(0, 30)
    team2_score = randint(0, 30)
    winner = team1 if team1_score > team2_score else team2

    return {
        "team1": team1,
        "team2": team2,
        "score1": team1_score,
        "score2": team2_score,
        "winner": winner,
        "year": year,
    }


# API endpoints
@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    # Encode team names
    try:
        team1_encoded = encoder.transform([input_data.team1])[0]
        team2_encoded = encoder.transform([input_data.team2])[0]
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid team name provided")

    # Prepare features and make predictions
    features = [[team1_encoded, team2_encoded, input_data.over_under_line]]
    probabilities = model.predict_proba(features)[0]
    prediction = "Over" if probabilities[1] > probabilities[0] else "Under"
    probability = max(probabilities)

    # Save prediction to database
    session = SessionLocal()
    db_prediction = Prediction(
        team1=input_data.team1,
        team2=input_data.team2,
        over_under_line=input_data.over_under_line,
        prediction=prediction,
        probability=probability,
    )
    session.add(db_prediction)
    session.commit()
    session.close()

    return PredictionOutput(prediction=prediction, probability=probability)


@app.get("/games/{year}")
def get_games(year: int):
    session = SessionLocal()
    games = session.query(Game).filter(Game.year == year).all()
    session.close()
    if not games:
        raise HTTPException(status_code=404, detail="No games found for the given year")
    return [{"team1": g.team1, "team2": g.team2, "score1": g.score1, "score2": g.score2, "winner": g.winner} for g in games]


@app.post("/simulate-games/{year}")
def simulate_games(year: int):
    session = SessionLocal()
    for i in range(10):  # Simulate 10 games
        game_result = simulate_game("Team A", "Team B", year)  # Example teams
        db_game = Game(
            year=game_result["year"],
            team1=game_result["team1"],
            team2=game_result["team2"],
            score1=game_result["score1"],
            score2=game_result["score2"],
            winner=game_result["winner"],
        )
        session.add(db_game)
    session.commit()
    session.close()
    return {"message": f"Simulated games for the year {year}"}


@app.get("/predictions")
def get_predictions():
    session = SessionLocal()
    predictions = session.query(Prediction).all()
    session.close()
    return [{"team1": p.team1, "team2": p.team2, "over_under_line": p.over_under_line,
             "prediction": p.prediction, "probability": p.probability} for p in predictions]


@app.post("/clear-data")
def clear_data():
    session = SessionLocal()
    session.query(Game).delete()
    session.query(Prediction).delete()
    session.commit()
    session.close()
    return {"message": "All data has been cleared."}
