"""
ETL Pipeline for NeuralFraud
Loads raw transaction CSV, cleans it, engineers behavioral features,
and loads the result into a SQLite database + a processed CSV/Parquet
that the model training script consumes.

Run:
    python backend/etl.py
"""
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw_transactions.csv"
DB_PATH = BASE_DIR / "data" / "fraud_detection.db"
PROCESSED_PATH = BASE_DIR / "data" / "processed_transactions.csv"

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Standardize column names to snake_case for downstream consistency
    df.columns = [c.strip() for c in df.columns]
    df["Transaction_Date"] = pd.to_datetime(
        df["Transaction_Date"], format="%d-%m-%Y %H:%M", errors="coerce"
    )
    df = df.dropna(subset=["Transaction_Date"])
    # Normalize Yes/No -> 1/0 for the keyword flag
    df["Suspicious_Keyword"] = (
        df["Suspicious_Keyword"].astype(str).str.strip().str.lower().eq("yes").astype(int)
    )

    # Guard against divide-by-zero / negative values
    df["Average_Spend"] = df["Average_Spend"].clip(lower=0.01)
    df["Transaction_Amount"] = df["Transaction_Amount"].clip(lower=0)
    df["Account_Age_Days"] = df["Account_Age_Days"].clip(lower=0)
    df["Previous_Transactions"] = df["Previous_Transactions"].clip(lower=0)
    return df

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # How far this transaction deviates from the customer's historical average spend.
    df["spend_ratio"] = df["Transaction_Amount"] / df["Average_Spend"]
    df["spend_deviation_abs"] = (df["Transaction_Amount"] - df["Average_Spend"]).abs()

    # Flag transactions that are a large multiple of the customer's usual spend
    df["is_spend_spike"] = (df["spend_ratio"] > 3).astype(int)

    # --- Account maturity signals ---
    df["is_new_account"] = (df["Account_Age_Days"] < 30).astype(int)
    df["account_age_years"] = df["Account_Age_Days"] / 365.0

    # --- Behavioral history signals ---
    df["is_low_history"] = (df["Previous_Transactions"] < 5).astype(int)

    # --- Time-based signals ---
    df["txn_hour"] = df["Transaction_Date"].dt.hour
    df["txn_day_of_week"] = df["Transaction_Date"].dt.dayofweek  # 0=Mon
    df["is_night_txn"] = df["txn_hour"].apply(lambda h: 1 if (h >= 0 and h < 5) else 0)
    df["is_weekend"] = (df["txn_day_of_week"] >= 5).astype(int)

    # --- Composite risk-flag count (purely for EDA / dashboard, not fed as a leak-y feature) ---
    df["risk_flag_count"] = (
        df["Is_International"] + df["Suspicious_Keyword"] + df["is_spend_spike"] + df["is_new_account"] + df["is_night_txn"]
    )
    return df

def save_to_sqlite(df: pd.DataFrame, db_path: Path = DB_PATH) -> None:
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("DROP TABLE IF EXISTS transactions")
    cursor.execute(
        """
        CREATE TABLE transactions (
            transaction_id TEXT PRIMARY KEY,
            customer_id TEXT,
            transaction_date TEXT,
            transaction_amount REAL,
            merchant_category TEXT,
            payment_method TEXT,
            device_type TEXT,
            location TEXT,
            is_international INTEGER,
            previous_transactions INTEGER,
            average_spend REAL,
            account_age_days INTEGER,
            suspicious_keyword INTEGER,
            fraudulent INTEGER,
            spend_ratio REAL,
            is_spend_spike INTEGER,
            is_new_account INTEGER,
            is_low_history INTEGER,
            txn_hour INTEGER,
            is_night_txn INTEGER,
            is_weekend INTEGER,
            risk_flag_count INTEGER
        )
        """
    )
    conn.commit()
    out = df.rename(
        columns={
            "Transaction_ID": "transaction_id",
            "Customer_ID": "customer_id",
            "Transaction_Date": "transaction_date",
            "Transaction_Amount": "transaction_amount",
            "Merchant_Category": "merchant_category",
            "Payment_Method": "payment_method",
            "Device_Type": "device_type",
            "Location": "location",
            "Is_International": "is_international",
            "Previous_Transactions": "previous_transactions",
            "Average_Spend": "average_spend",
            "Account_Age_Days": "account_age_days",
            "Suspicious_Keyword": "suspicious_keyword",
            "Fraudulent": "fraudulent",
        }
    )[
        ["transaction_id",
        "customer_id",
        "transaction_date",
        "transaction_amount",
        "merchant_category",
        "payment_method",
        "device_type",
        "location",
        "is_international",
        "previous_transactions",
        "average_spend",
        "account_age_days",
        "suspicious_keyword",
        "fraudulent",
        "spend_ratio",
        "is_spend_spike",
        "is_new_account",
        "is_low_history",
        "txn_hour",
        "is_night_txn",
        "is_weekend",
        "risk_flag_count",
        ]
    ]
    out["transaction_date"] = out["transaction_date"].astype(str)
    out.to_sql("transactions", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()

def run_etl() -> pd.DataFrame:
    print("Loading raw data...")
    raw = load_raw()
    print(f"  {len(raw)} rows loaded")
    print("Cleaning...")
    cleaned = clean(raw)
    print(f"  {len(cleaned)} rows after cleaning")
    print("Engineering features...")
    featured = engineer_features(cleaned)
    print("Saving to SQLite...")
    save_to_sqlite(featured)
    print("Saving processed CSV...")
    featured.to_csv(PROCESSED_PATH, index=False)
    print("ETL process completed.")
    print(f"  DB: {DB_PATH}")
    print(f"  CSV: {PROCESSED_PATH}")
    return featured

if __name__ == "__main__":
    run_etl()
