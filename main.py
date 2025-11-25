from fastapi import FastAPI
from pydantic import BaseModel
from datetime import datetime
import numpy as np
import joblib

app = FastAPI()

model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")

class PredictionRequest(BaseModel):
    Age: float
    Income: float
    LoanAmount: float
    CreditScore: float
    MonthsEmployed: float
    NumCreditLines: float
    InterestRate: float
    LoanTerm: float
    DTIRatio: float
    HasMortgage: int
    HasDependents: int
    HasCoSigner: int

@app.get("/")
def hello():
    return {
            "success": True,
            "message": "Hello From Server"
        }

@app.post("/predict")
def predict(data: PredictionRequest):
    try:
        features = np.array([[
            data.Age,
            data.Income,
            data.LoanAmount,
            data.CreditScore,
            data.MonthsEmployed,
            data.NumCreditLines,
            data.InterestRate,
            data.LoanTerm,
            data.DTIRatio,
            data.HasMortgage,
            data.HasDependents,
            data.HasCoSigner
        ]])

        features_scaled = scaler.transform(features)

        probability = model.predict_proba(features_scaled)[0][1]
        risk_score = int(probability * 100)

        # Risk Tier logic
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.6:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return {
            "success": True,
            "message": "Prediction completed successfully",
            "data": {
                "default_probability": round(float(probability), 4),
                "risk_score": risk_score,
                "risk_level": risk_level,
                "threshold_used": 0.5
            },
            "meta": {
                "model_version": "v1.0",
                "timestamp": datetime.utcnow().isoformat()
            }
        }

    except Exception as e:
        return {
            "success": False,
            "message": "Prediction failed",
            "error": str(e)
        }
