
# NeuralFraud — Financial Fraud Detection Platform

A hybrid (supervised + unsupervised) machine learning system for detecting fraudulent
financial transactions, with a FastAPI backend and a Power BI–style interactive dashboard.

Built as a portfolio project demonstrating: ETL pipeline design, behavioral feature
engineering, imbalanced-classification modeling, anomaly detection, REST API design,
and dashboard/data-visualization skills.
Deployed link:- https://neuralfraud-financial-fraud-detection-platform-brurmt8uyssvtgv.streamlit.app/
---

## Why "hybrid" detection?

Most simple fraud-detection demos train one classifier and stop there. This project
combines two complementary approaches, which is closer to how real fraud teams work:

| Model | Type | Catches |
|---|---|---|
| **Random Forest** | Supervised (learns from labeled fraud) | Known fraud patterns |
| **Isolation Forest** | Unsupervised (no labels needed) | Novel/unseen anomalies |

Their outputs are blended into a single **combined risk score**, so the system can flag
suspicious activity even when it doesn't match historical fraud patterns exactly.

---

## Project Structure

```
neuralfraud/
├── backend/
│   ├── etl.py              # Data cleaning + feature engineering + SQLite load
│   ├── train_model.py       # Trains RandomForest + IsolationForest, saves to /models
│   ├── main.py               # FastAPI app: /predict, /stats, /recent-alerts
│   └── stream_monitor.py     # Simulated real-time transaction scoring + alerts
├── dashboard/
│   └── app.py                 # Streamlit dashboard (Power BI-style UI)
├── data/
│   ├── raw_transactions.csv    # Your original dataset
│   ├── processed_transactions.csv  # After ETL + feature engineering
│   └── fraud_detection.db       # SQLite database
├── models/
│   ├── random_forest.joblib
│   ├── isolation_forest.joblib
│   ├── scaler.joblib
│   ├── encoders.joblib
│   └── metadata.json             # Training metrics, feature importances
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the ETL pipeline (cleans data, engineers features, loads SQLite)
python backend/etl.py

# 3. Train the models
python backend/train_model.py

# 4. Start the API (in one terminal)
uvicorn backend.main:app --reload --port 8000
# Docs available at http://127.0.0.1:8000/docs

# 5. Start the dashboard (in another terminal)
streamlit run dashboard/app.py

# 6. (Optional) Simulate a real-time transaction feed
python backend/stream_monitor.py --delay 0.2 --limit 50
```

---

## Feature Engineering

Beyond the raw columns, the ETL pipeline derives behavioral signals that are the actual
differentiators in the model (and the strongest talking points in an interview):

- **`spend_ratio`** — transaction amount vs. the customer's historical average spend
- **`is_spend_spike`** — flags transactions >3x a customer's normal spend
- **`is_new_account`** — accounts younger than 30 days (higher fraud risk)
- **`is_low_history`** — customers with fewer than 5 prior transactions
- **`txn_hour` / `is_night_txn`** — time-of-day risk (fraud clusters late at night)
- **`is_weekend`** — day-of-week pattern

## Results (on this dataset)

- 5,000 transactions, 9.64% fraud rate (realistic imbalance, not synthetic 50/50)
- Random Forest ROC-AUC: **0.858**
- Top predictive features: transaction hour, international flag, suspicious keyword,
  night-time transaction flag

## Model Notes / Honest Limitations

- The dataset is moderately imbalanced (~10% fraud); `class_weight="balanced"` is used
  in the Random Forest to compensate, but with more data a technique like SMOTE could
  improve minority-class recall further.
- Isolation Forest is unsupervised by design, so its precision on this dataset is lower
  than the supervised model — that's expected and is exactly why the two are blended
  rather than used alone.
- `stream_monitor.py` simulates a live feed for demo purposes; swapping it for a real
  Kafka/Spark Streaming consumer would be a natural "Future Enhancements" upgrade.

## Future Enhancements

- Real-time streaming via Kafka/Spark Structured Streaming
- Graph-based analysis to detect fraud rings across linked accounts
- SHAP-based per-transaction explainability
- Deploy API to Render/Railway free tier + dashboard to Streamlit Community Cloud
=======

>>>>>>> 9ee12c90ca112efd6546c1d6585df39d20f0564e
