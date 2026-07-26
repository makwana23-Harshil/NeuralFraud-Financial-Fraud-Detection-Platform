"""
NeuralFraud API — FastAPI backend
Serves fraud predictions from the trained RandomForest + IsolationForest models.
Run:
    uvicorn backend.main:app --reload --port 8000
Docs:
    http://127.0.0.1:8000/docs
"""
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Optional
import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
DB_PATH = BASE_DIR / "data" / "fraud_detection.db"

app = FastAPI(title="NeuralFraud API",description="Hybrid (supervised + unsupervised) financial fraud detection API",version="1.0.0",)

# Allow the Streamlit dashboard (or any frontend) to call this API
app.add_middleware(CORSMiddleware,allow_origins=["*"],allow_methods=["*"],allow_headers=["*"],)

# --- Load models once at startup ---
rf_model = joblib.load(MODELS_DIR / "random_forest.joblib")
iso_model = joblib.load(MODELS_DIR / "isolation_forest.joblib")
scaler = joblib.load(MODELS_DIR / "scaler.joblib")
encoders = joblib.load(MODELS_DIR / "encoders.joblib")

FEATURE_COLS = [
    "Transaction_Amount",
    "Is_International",
    "Previous_Transactions",
    "Average_Spend",
    "Account_Age_Days",
    "Suspicious_Keyword",
    "spend_ratio",
    "is_spend_spike",
    "is_new_account",
    "is_low_history",
    "txn_hour",
    "is_night_txn",
    "is_weekend",
    "Merchant_Category_enc",
    "Payment_Method_enc",
    "Device_Type_enc",
    "Location_enc",
]

CATEGORICAL_COLS = ["Merchant_Category", "Payment_Method", "Device_Type", "Location"]


class TransactionInput(BaseModel):
    Transaction_Amount: float = Field(..., gt=0, example=250.75)
    Merchant_Category: str = Field(..., example="Electronics")
    Payment_Method: str = Field(..., example="Credit Card")
    Device_Type: str = Field(..., example="Mobile")
    Location: str = Field(..., example="Mumbai")
    Is_International: int = Field(..., ge=0, le=1, example=0)
    Previous_Transactions: int = Field(..., ge=0, example=45)
    Average_Spend: float = Field(..., gt=0, example=180.0)
    Account_Age_Days: int = Field(..., ge=0, example=400)
    Suspicious_Keyword: int = Field(0, ge=0, le=1, example=0)
    txn_hour: Optional[int] = Field(None, ge=0, le=23, example=14)

class PredictionResponse(BaseModel):
    transaction_amount: float
    rf_fraud_probability: float
    rf_prediction: int
    isolation_forest_anomaly: int
    combined_risk_score: float
    risk_level: str

def _build_feature_row(payload: TransactionInput) -> pd.DataFrame:
    hour = payload.txn_hour if payload.txn_hour is not None else datetime.now().hour
    spend_ratio = payload.Transaction_Amount / max(payload.Average_Spend, 0.01)
    is_spend_spike = int(spend_ratio > 3)
    is_new_account = int(payload.Account_Age_Days < 30)
    is_low_history = int(payload.Previous_Transactions < 5)
    is_night_txn = int(0 <= hour < 5)
    is_weekend = 0  
    row = {
        "Transaction_Amount": payload.Transaction_Amount,
        "Is_International": payload.Is_International,
        "Previous_Transactions": payload.Previous_Transactions,
        "Average_Spend": payload.Average_Spend,
        "Account_Age_Days": payload.Account_Age_Days,
        "Suspicious_Keyword": payload.Suspicious_Keyword,
        "spend_ratio": spend_ratio,
        "is_spend_spike": is_spend_spike,
        "is_new_account": is_new_account,
        "is_low_history": is_low_history,
        "txn_hour": hour,
        "is_night_txn": is_night_txn,
        "is_weekend": is_weekend,
    }

    cat_values = {
        "Merchant_Category": payload.Merchant_Category,
        "Payment_Method": payload.Payment_Method,
        "Device_Type": payload.Device_Type,
        "Location": payload.Location,
    }
    for col, value in cat_values.items():
        le = encoders[col]
        known = set(le.classes_)
        row[f"{col}_enc"] = int(le.transform([value])[0]) if value in known else -1
    return pd.DataFrame([row])[FEATURE_COLS]

@app.get("/")
def root():
    return {"service": "NeuralFraud API","status": "online","endpoints": ["/predict", "/health", "/stats", "/recent-alerts"],}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict", response_model=PredictionResponse)
def predict(payload: TransactionInput):
    try:
        X = _build_feature_row(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature build failed: {e}")
    rf_proba = float(rf_model.predict_proba(X)[0, 1])
    rf_pred = int(rf_model.predict(X)[0])
    X_scaled = scaler.transform(X)
    iso_flag = int(iso_model.predict(X_scaled)[0] == -1)

    # Combine both signals into one risk score (simple weighted blend)
    combined = 0.7 * rf_proba + 0.3 * iso_flag
    if combined >= 0.7:
        risk_level = "HIGH"
    elif combined >= 0.4:
        risk_level = "MEDIUM"
    else:
        risk_level = "LOW"

    return PredictionResponse(
        transaction_amount=payload.Transaction_Amount,
        rf_fraud_probability=round(rf_proba, 4),
        rf_prediction=rf_pred,
        isolation_forest_anomaly=iso_flag,
        combined_risk_score=round(combined, 4),
        risk_level=risk_level,
    )

@app.get("/stats")
def stats():
    """Aggregate stats for the dashboard — pulled straight from SQLite."""
    conn = sqlite3.connect(DB_PATH)
    try:
        total = pd.read_sql("SELECT COUNT(*) as c FROM transactions", conn).iloc[0]["c"]
        fraud_count = pd.read_sql(
            "SELECT COUNT(*) as c FROM transactions WHERE fraudulent = 1", conn
        ).iloc[0]["c"]
        avg_amount = pd.read_sql(
            "SELECT AVG(transaction_amount) as a FROM transactions", conn
        ).iloc[0]["a"]
        by_location = pd.read_sql(
            """SELECT location, COUNT(*) as total,
                      SUM(fraudulent) as fraud_count
               FROM transactions GROUP BY location""",
            conn,
        ).to_dict(orient="records")
        by_category = pd.read_sql(
            """SELECT merchant_category, COUNT(*) as total,
                      SUM(fraudulent) as fraud_count
               FROM transactions GROUP BY merchant_category""",
            conn,
        ).to_dict(orient="records")
    finally:
        conn.close()

    return {
        "total_transactions": int(total),
        "fraud_count": int(fraud_count),
        "fraud_rate_pct": round(fraud_count / total * 100, 2) if total else 0,
        "avg_transaction_amount": round(float(avg_amount), 2) if avg_amount else 0,
        "by_location": by_location,
        "by_category": by_category,
    }

@app.get("/recent-alerts")
def recent_alerts(limit: int = 20):
    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql(
            """SELECT transaction_id, customer_id, transaction_date, transaction_amount,
                      merchant_category, location, fraudulent
               FROM transactions
               WHERE fraudulent = 1
               ORDER BY transaction_date DESC
               LIMIT ?""",
            conn,
            params=(limit,),
        )
    finally:
        conn.close()
    return df.to_dict(orient="records")
