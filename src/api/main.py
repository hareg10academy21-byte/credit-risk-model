from fastapi import FastAPI
import pandas as pd
import joblib

from src.api.pydantic_models import CustomerRequest, PredictionResponse

app = FastAPI(title="Credit Risk API")

# =========================
# LOAD MODEL
# =========================
model = joblib.load("src/models/model.pkl")


@app.get("/")
def home():
    return {"message": "Credit Risk API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(data: CustomerRequest):

    input_df = pd.DataFrame([data.dict()])

    # align columns with training (important fix)
    input_df = pd.get_dummies(input_df)
    input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)

    prediction = model.predict(input_df)[0]

    return PredictionResponse(
        customer_id=data.CustomerId,
        risk_probability=float(prediction),
        risk_label="HIGH_RISK" if prediction == 1 else "LOW_RISK"
    )