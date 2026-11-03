"""FastAPI service for the engineered Titanic survival pipeline.

Run:  uvicorn main:app --reload
Docs: http://127.0.0.1:8000/docs
"""
from contextlib import asynccontextmanager
from typing import Literal, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import features  # noqa: F401  (makes `features.engineer_features` importable for joblib)

MODEL_PATH = "titanic_engineered_pipeline.joblib"
THRESHOLD = 0.5
COLUMNS = ["age", "fare", "sibsp", "parch", "pclass", "sex", "embarked", "name"]

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    state["model"] = joblib.load(MODEL_PATH)
    yield
    state.clear()


app = FastAPI(
    title="Titanic Survival API",
    description="Serves the engineered logistic-regression pipeline from the notebook.",
    version="1.0.0",
    lifespan=lifespan,
)


class Passenger(BaseModel):
    name: str = Field(..., examples=["Braund, Mr. Owen Harris"],
                      description="Full name in 'Surname, Title. Given names' format (title is extracted from it).")
    pclass: Literal[1, 2, 3] = Field(..., examples=[3])
    sex: Literal["male", "female"]
    age: Optional[float] = Field(None, ge=0, le=120, examples=[22])
    sibsp: int = Field(0, ge=0, examples=[1])
    parch: int = Field(0, ge=0, examples=[0])
    fare: Optional[float] = Field(None, ge=0, examples=[7.25])
    embarked: Optional[Literal["C", "Q", "S"]] = Field(None, examples=["S"])


class Prediction(BaseModel):
    survival_probability: float
    survived: bool


def to_frame(passengers: list[Passenger]) -> pd.DataFrame:
    df = pd.DataFrame([p.model_dump() for p in passengers])[COLUMNS]
    # None -> NaN so the pipeline's imputers treat them as missing
    return df.astype(object).where(df.notna(), np.nan)


def predict(passengers: list[Passenger]) -> list[Prediction]:
    model = state.get("model")
    if model is None:
        raise HTTPException(503, "Model not loaded")
    probs = model.predict_proba(to_frame(passengers))[:, 1]
    return [
        Prediction(survival_probability=round(float(p), 4), survived=bool(p >= THRESHOLD))
        for p in probs
    ]


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": "model" in state}


@app.post("/predict", response_model=Prediction)
def predict_one(passenger: Passenger):
    return predict([passenger])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(passengers: list[Passenger]):
    if not passengers:
        raise HTTPException(422, "Send at least one passenger")
    return predict(passengers)
