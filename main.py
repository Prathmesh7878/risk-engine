from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
from xgboost import XGBClassifier

app = FastAPI()

# Allow frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # for debugging, restrict later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and features
model = XGBClassifier()
model.load_model("credit_model.json")
features = joblib.load("feature_columns.pkl")

profit_per_good = 15000
loss_per_default = 60000


# ===== INPUT STRUCTURE =====

class Applicant(BaseModel):
    age: float
    employment_years: float
    monthly_income: float
    loan_amount: float
    monthly_emi: float
    previous_loans: int
    overdue_amount: float
    late_payment_ratio: float


@app.get("/")
def root():
    return {"status": "Risk Engine Running"}


@app.post("/predict")
def predict(data: Applicant):

    # Feature Engineering
    credit_income_ratio = data.loan_amount / max(data.monthly_income, 1)
    emi_income_ratio = data.monthly_emi / max(data.monthly_income, 1)

    input_dict = {
        "age": data.age,
        "employment_years": data.employment_years,
        "credit_income_ratio": credit_income_ratio,
        "emi_income_ratio": emi_income_ratio,
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.5,
        "EXT_SOURCE_3": 0.5,
        "income_type_encoded": 1,
        "loan_count": data.previous_loans,
        "total_overdue": data.overdue_amount,
        "overdue_count": 1 if data.overdue_amount > 0 else 0,
        "late_ratio": data.late_payment_ratio,
        "avg_delay": 0
    }

    input_df = pd.DataFrame([input_dict])[features]

    pd_prob = model.predict_proba(input_df)[0][1]

    # Business Logic
    ev = (1 - pd_prob) * profit_per_good - pd_prob * loss_per_default
    approve = ev > 0
    credit_score = 300 + (1 - pd_prob) * 600

    if credit_score >= 750:
        risk_band = "Prime"
    elif credit_score >= 650:
        risk_band = "Near Prime"
    elif credit_score >= 550:
        risk_band = "Subprime"
    else:
        risk_band = "High Risk"

    # RESPONSE STRUCTURE MATCHING UI
    return {
        "creditScore": round(float(credit_score)),
        "riskBand": risk_band,
        "recommendation": "Approve" if approve else "Reject",
        "confidence": round(float(1 - pd_prob), 3),
        "predictionProbability": round(float(pd_prob), 4),
        "debtToIncomeRatio": round(float(emi_income_ratio), 3),
        "expectedValue": round(float(ev), 2),
        "approve": bool(approve),
    }
