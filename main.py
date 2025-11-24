from fastapi import FastAPI
import joblib
import numpy as np

model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Welcome to the Loan Default Prediction API"}

@app.post("/predict")
def predict(data: dict):

    features = np.array([[
        data["Age"],
        data["Income"],
        data["LoanAmount"],
        data["CreditScore"],
        data["MonthsEmployed"],
        data["NumCreditLines"],
        data["InterestRate"],
        data["LoanTerm"],
        data["DTIRatio"],
        data["HasMortgage"],
        data["HasDependents"],
        data["HasCoSigner"]
    ]])

    features_scaled = scaler.transform(features)
    
    probability = model.predict_proba(features_scaled)[0][1]

    return {
        "default_probability": float(probability)
    }
