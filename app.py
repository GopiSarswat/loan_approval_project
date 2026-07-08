"""
app.py
------
ONE JOB: load the trained model (model.pkl) and serve predictions.

Run:
    uvicorn app:app --reload

Then open frontend/index.html in your browser (it calls this API).
Docs available at: http://127.0.0.1:8000/docs
"""

import joblib
import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = "model.pkl"

app = FastAPI(title="Loan Approval Prediction API")

# Allow the simple HTML/JS frontend (opened as a local file or on another
# port) to call this API from the browser.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
encoders = bundle["encoders"]
feature_cols = bundle["feature_cols"]
categorical_cols = bundle["categorical_cols"]


class LoanApplication(BaseModel):
    Gender: str = Field(examples=["Male"])
    Married: str = Field(examples=["Yes"])
    Dependents: str = Field(examples=["0"])
    Education: str = Field(examples=["Graduate"])
    Self_Employed: str = Field(examples=["No"])
    ApplicantIncome: float = Field(examples=[5000])
    CoapplicantIncome: float = Field(examples=[0])
    LoanAmount: float = Field(examples=[150])
    Loan_Amount_Term: float = Field(examples=[360])
    Credit_History: float = Field(examples=[1.0])
    Property_Area: str = Field(examples=["Urban"])


def encode_input(application: LoanApplication) -> pd.DataFrame:
    row = application.model_dump()
    df = pd.DataFrame([row])

    for col in categorical_cols:
        le = encoders[col]
        value = str(df.loc[0, col])
        if value not in le.classes_:
            # Unseen category at prediction time -> fall back to the
            # most common training value instead of crashing.
            value = le.classes_[0]
        df[col] = le.transform([value])

    return df[feature_cols]


@app.get("/")
def health_check():
    return {"status": "ok", "message": "Loan Approval Prediction API is running"}


@app.post("/predict")
def predict(application: LoanApplication):
    X = encode_input(application)

    prediction = model.predict(X)[0]
    proba = model.predict_proba(X)[0]

    target_encoder = encoders["Loan_Status"]
    label = target_encoder.inverse_transform([prediction])[0]
    approved = label == "Y"

    approve_idx = list(target_encoder.classes_).index("Y")

    return {
        "approved": bool(approved),
        "decision": "Approved" if approved else "Rejected",
        "confidence": round(float(proba[approve_idx] if approved else 1 - proba[approve_idx]), 3),
        "approval_probability": round(float(proba[approve_idx]), 3),
    }
