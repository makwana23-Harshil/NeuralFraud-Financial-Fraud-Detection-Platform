"""
NeuralFraud — Simulated Real-Time Monitor
Streams transactions from the processed dataset one-by-one (simulating a live feed),
scores each with the trained models, and prints/alerts on high-risk transactions.
This is a solo-feasible stand-in for a Kafka/Spark streaming pipeline: it demonstrates
the same real-time-monitoring concept without requiring a message broker to run.
Run:
    python backend/stream_monitor.py --delay 0.2 --limit 50
"""
import argparse
import os
import smtplib
import time
from email.mime.text import MIMEText
from pathlib import Path
import joblib
import pandas as pd
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
PROCESSED_PATH = BASE_DIR / "data" / "processed_transactions.csv"

load_dotenv(BASE_DIR / ".env")

FEATURE_COLS = [
    "Transaction_Amount", "Is_International", "Previous_Transactions", "Average_Spend",
    "Account_Age_Days", "Suspicious_Keyword", "spend_ratio", "is_spend_spike",
    "is_new_account", "is_low_history", "txn_hour", "is_night_txn", "is_weekend",
    "Merchant_Category_enc", "Payment_Method_enc", "Device_Type_enc", "Location_enc",
]

def send_email_alert(transaction_row: pd.Series, risk_score: float) -> bool:
    sender = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    recipient = os.environ.get("ALERT_RECIPIENT")

    if not (sender and password and recipient):
        return False  # not configured — caller falls back to console alert

    subject = f"[NeuralFraud] High-risk transaction {transaction_row['Transaction_ID']}"
    body = (
        f"Transaction ID: {transaction_row['Transaction_ID']}\n"
        f"Amount: ${transaction_row['Transaction_Amount']:.2f}\n"
        f"Location: {transaction_row['Location']}\n"
        f"Merchant Category: {transaction_row['Merchant_Category']}\n"
        f"Risk score: {risk_score:.2f}\n"
    )
    message = MIMEText(body)
    message["Subject"] = subject
    message["From"] = sender
    message["To"] = recipient
    try:
        server = smtplib.SMTP("smtp.gmail.com", 587)
        server.starttls()
        server.login(sender, password)
        server.sendmail(sender, recipient, message.as_string())
        server.quit()
        return True
    except Exception as e:
        print(f"  [email alert failed, falling back to console: {e}]")
        return False

def send_alert(transaction_row: pd.Series, risk_score: float) -> None:
    emailed = send_email_alert(transaction_row, risk_score)
    tag = "EMAIL SENT" if emailed else "CONSOLE ONLY"
    print(
        f"  ALERT! [{tag}] Transaction {transaction_row['Transaction_ID']} | "
        f"Amount: ${transaction_row['Transaction_Amount']:.2f} | "
        f"Location: {transaction_row['Location']} | "
        f"Risk score: {risk_score:.2f}"
    )

def load_encoded_stream() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH)
    encoders = joblib.load(MODELS_DIR / "encoders.joblib")
    for col in ["Merchant_Category", "Payment_Method", "Device_Type", "Location"]:
        le = encoders[col]
        known = set(le.classes_)
        df[f"{col}_enc"] = df[col].astype(str).apply(
            lambda v: le.transform([v])[0] if v in known else -1
        )
    return df

def run_stream(delay: float, limit: int) -> None:
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    iso = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    df = load_encoded_stream().sample(frac=1, random_state=None).head(limit)
    print(f"Streaming {len(df)} transactions (simulated real-time feed)...\n")
    alert_count = 0
    for _, row in df.iterrows():
        X = row[FEATURE_COLS].to_frame().T
        rf_proba = float(rf.predict_proba(X)[0, 1])
        X_scaled = scaler.transform(X)
        iso_flag = int(iso.predict(X_scaled)[0] == -1)
        combined = 0.7 * rf_proba + 0.3 * iso_flag
        if combined >= 0.7:
            send_alert(row, combined)
            alert_count += 1
        else:
            print(f"  OK    {row['Transaction_ID']} | risk={combined:.2f}")
        time.sleep(delay)
    print(f"\nStream complete. {alert_count} high-risk alerts raised out of {len(df)} transactions.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--delay", type=float, default=0.1, help="Seconds between transactions")
    parser.add_argument("--limit", type=int, default=50, help="Number of transactions to stream")
    args = parser.parse_args()
    run_stream(args.delay, args.limit)
