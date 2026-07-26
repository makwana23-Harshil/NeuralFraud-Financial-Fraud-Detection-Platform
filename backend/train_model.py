import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed_transactions.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

CATEGORICAL_COLS = ["Merchant_Category", "Payment_Method", "Device_Type", "Location"]
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
TARGET_COL = "Fraudulent"

def load_processed() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH)
    return df

def encode_categoricals(df: pd.DataFrame, encoders: dict | None = None):
    df = df.copy()
    fit_mode = encoders is None
    if fit_mode:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if fit_mode:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[f"{col}_enc"] = df[col].astype(str).apply(
                lambda v: le.transform([v])[0] if v in known else -1
            )
    return df, encoders

def train():
    df = load_processed()
    df, encoders = encode_categoricals(df)
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # --- Supervised model: Random Forest ---
    print("Training RandomForestClassifier...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",  
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]

    print("\n--- Random Forest Results ---")
    print(classification_report(y_test, y_pred, digits=3))
    auc = roc_auc_score(y_test, y_proba)
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    # --- Second supervised model: Logistic Regression (simple, interpretable baseline) ---
    print("\nTraining LogisticRegression (baseline comparison model)...")
    logreg = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )
    logreg.fit(X_train_scaled, y_train)
    logreg_pred = logreg.predict(X_test_scaled)
    logreg_proba = logreg.predict_proba(X_test_scaled)[:, 1]
    logreg_auc = roc_auc_score(y_test, logreg_proba)

    print("\n--- Logistic Regression Results ---")
    print(classification_report(y_test, logreg_pred, digits=3))
    print(f"ROC-AUC: {logreg_auc:.4f}")
    print(
        f"\nModel comparison -> Random Forest ROC-AUC: {auc:.4f} | "
        f"Logistic Regression ROC-AUC: {logreg_auc:.4f}"
    )

    # --- Unsupervised model: Isolation Forest ---
    # Trained only on legitimate-looking behavior patterns (no label used),
    # contamination set close to the known fraud rate for calibration.
    print("\nTraining IsolationForest...")
    fraud_rate = y_train.mean()
    iso = IsolationForest(n_estimators=300,contamination=max(min(fraud_rate, 0.2), 0.01),random_state=42,n_jobs=-1,)
    iso.fit(X_train_scaled)

    # IsolationForest: -1 = anomaly, 1 = normal. Convert to fraud-style 1/0.
    iso_pred_test = (iso.predict(X_test_scaled) == -1).astype(int)
    print("\n--- Isolation Forest (unsupervised) vs true labels ---")
    print(classification_report(y_test, iso_pred_test, digits=3))

    # --- Feature importances (for SHAP-lite explainability on the dashboard) ---
    importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(
        ascending=False
    )
    print("\nTop feature importances:")
    print(importances.head(10))

    # --- Persist everything the API/dashboard need ---
    joblib.dump(rf, MODELS_DIR / "random_forest.joblib")
    joblib.dump(logreg, MODELS_DIR / "logistic_regression.joblib")
    joblib.dump(iso, MODELS_DIR / "isolation_forest.joblib")
    joblib.dump(scaler, MODELS_DIR / "scaler.joblib")
    joblib.dump(encoders, MODELS_DIR / "encoders.joblib")

    # Save held-out test predictions so the dashboard can render an honest
    # ROC curve / confusion matrix without re-training or leaking train data.
    test_results = pd.DataFrame(
        {"y_true": y_test.values,"rf_proba": y_proba,"rf_pred": y_pred,"logreg_proba": logreg_proba,
            "logreg_pred": logreg_pred,"iso_pred": iso_pred_test,
        }
    )
    test_results.to_csv(MODELS_DIR / "test_predictions.csv", index=False)

    metadata = {
        "feature_cols": FEATURE_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "rf_roc_auc": float(auc),
        "logreg_roc_auc": float(logreg_auc),
        "fraud_rate_train": float(fraud_rate),
        "top_features": importances.head(10).to_dict(),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }
    with open(MODELS_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nModels saved to {MODELS_DIR}")
    return rf, logreg, iso, scaler, encoders, metadata

if __name__ == "__main__":
    train()
