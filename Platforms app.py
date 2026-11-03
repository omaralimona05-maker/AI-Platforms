"""Streamlit UI for the engineered Titanic survival pipeline.

Run:  streamlit run streamlit_app.py
"""
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

import features  # noqa: F401  (lets joblib find features.engineer_features)
from train import MODEL_PATH, main as train_model

COLUMNS = ["age", "fare", "sibsp", "parch", "pclass", "sex", "embarked", "name"]

st.set_page_config(page_title="Titanic Survival Predictor", page_icon="🚢")


@st.cache_resource(show_spinner="Loading model...")
def load_model():
    if not os.path.exists(MODEL_PATH):
        # First run: train it (needs internet to download the dataset from OpenML)
        train_model()
    return joblib.load(MODEL_PATH)


def predict(row: dict) -> float:
    df = pd.DataFrame([row])[COLUMNS].astype(object)
    df = df.where(df.notna(), np.nan)
    return float(load_model().predict_proba(df)[0, 1])


st.title("🚢 Titanic Survival Predictor")
st.caption("Engineered logistic-regression pipeline from the notebook.")

try:
    load_model()
except Exception as e:
    st.error(f"Could not load or train the model: {e}")
    st.info("Run `python train.py` once (needs internet), then refresh.")
    st.stop()

tab_single, tab_batch = st.tabs(["Single passenger", "Batch (CSV)"])

with tab_single:
    with st.form("passenger"):
        name = st.text_input("Name", "Braund, Mr. Owen Harris",
                             help="Format: 'Surname, Title. Given names' (the title is used as a feature)")
        c1, c2 = st.columns(2)
        pclass = c1.selectbox("Class", [1, 2, 3], index=2)
        sex = c2.selectbox("Sex", ["male", "female"])
        age_known = c1.checkbox("Age known", True)
        age = c1.number_input("Age", 0.0, 120.0, 22.0, step=1.0, disabled=not age_known)
        fare_known = c2.checkbox("Fare known", True)
        fare = c2.number_input("Fare", 0.0, 600.0, 7.25, step=0.5, disabled=not fare_known)
        sibsp = c1.number_input("Siblings/spouses aboard", 0, 10, 1)
        parch = c2.number_input("Parents/children aboard", 0, 10, 0)
        embarked = st.selectbox("Port of embarkation", ["S", "C", "Q", "Unknown"])
        submitted = st.form_submit_button("Predict")

    if submitted:
        prob = predict({
            "name": name, "pclass": pclass, "sex": sex,
            "age": age if age_known else np.nan,
            "fare": fare if fare_known else np.nan,
            "sibsp": sibsp, "parch": parch,
            "embarked": np.nan if embarked == "Unknown" else embarked,
        })
        st.metric("Survival probability", f"{prob:.1%}")
        st.progress(prob)
        (st.success if prob >= 0.5 else st.error)(
            "Predicted: survived" if prob >= 0.5 else "Predicted: did not survive"
        )

with tab_batch:
    st.write("Upload a CSV with columns: " + ", ".join(f"`{c}`" for c in COLUMNS))
    file = st.file_uploader("CSV file", type="csv")
    if file:
        data = pd.read_csv(file)
        missing = [c for c in COLUMNS if c not in data.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            probs = load_model().predict_proba(data[COLUMNS])[:, 1]
            out = data.assign(survival_probability=probs.round(4), survived=probs >= 0.5)
            st.dataframe(out)
            st.download_button("Download results", out.to_csv(index=False), "predictions.csv")
