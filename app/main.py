# main.py
from fastapi import FastAPI
from pydantic import BaseModel

from contextlib import asynccontextmanager


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib


scaler = joblib.load("models/scaler.pkl")
ml_models = {
    "p10": joblib.load("models/xgboost_p10_model.pkl"),
    "p50": joblib.load("models/xgboost_p50_model.pkl"),
    "p90": joblib.load("models/xgboost_p90_model.pkl"),

}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model

    print("started")
    yield
    # Clean up the ML models and release the resources

app = FastAPI()


# Prediction function

@app.get("/")
async def predict():

    print(ml_models)
    return "Day-Ahead Market (DAM) price "


class Features(BaseModel):
    hour: int
    dayofweek: int
    month: int

    price_lag_1: float
    price_lag_2: float
    price_lag_7: float
    price_ma7: float
    price_ma2: float
    Solar_2da: float
    Wind_2da: float
    DOM_dem_7da: float
    Oil: float
    Hydro: float
    Coal: float
    Gas: float
    Nuclear: float


@app.post("/predict_xgboost")
def predict_xhboost(features: Features):

    # Convert input to DataFrame
    input_df = pd.DataFrame([features.dict()])

    # Scale input
    input_scaled = scaler.transform(input_df)

    # Predict quantiles
    predictions = {q: str(model.predict(
        input_scaled)[0]) for q, model in ml_models.items()}

    return {"prediction": predictions}
