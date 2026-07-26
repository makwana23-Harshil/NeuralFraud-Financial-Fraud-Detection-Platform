"""NeuralFraud — Live Transaction Risk Checker tab content."""

import pandas as pd
import streamlit as st

from common import POWERBI_RED, POWERBI_AMBER, POWERBI_GREEN, FEATURE_COLS

def render(df, rf_model, iso_model, scaler, encoders):
    locations = sorted(df["location"].unique().tolist())
    categories = sorted(df["merchant_category"].unique().tolist())
    payment_methods = sorted(df["payment_method"].unique().tolist())

    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:38px;font-weight:600;color:#252423;"> Live Transaction Risk Checker</h1>""",unsafe_allow_html=True,)
    st.caption("Score a hypothetical transaction using the trained models (same logic as the FastAPI /predict endpoint)")

    with st.form("scoring_form"):
        fc1, fc2, fc3, fc4 = st.columns(4)
        with fc1:
            amt = st.number_input("Transaction Amount", min_value=0.0, value=250.0, step=10.0)
            merchant = st.selectbox("Merchant Category", categories)
        with fc2:
            payment = st.selectbox("Payment Method", payment_methods)
            device = st.selectbox("Device Type", sorted(df["device_type"].unique().tolist()))
        with fc3:
            location = st.selectbox("Location", locations)
            is_intl = st.selectbox("International?", [0, 1], format_func=lambda x: "Yes" if x else "No")
        with fc4:
            avg_spend = st.number_input("Customer's Avg Spend", min_value=0.01, value=180.0, step=10.0)
            acct_age = st.number_input("Account Age (days)", min_value=0, value=400, step=10)

        prev_txns = st.slider("Previous Transactions", 0, 200, 45)
        suspicious_kw = st.checkbox("Suspicious keyword flagged?")
        hour = st.slider("Hour of Day", 0, 23, 14)
        submitted = st.form_submit_button("Score Transaction", type="primary")

    if submitted:
        spend_ratio = amt / max(avg_spend, 0.01)
        row = {
            "Transaction_Amount": amt,
            "Is_International": is_intl,
            "Previous_Transactions": prev_txns,
            "Average_Spend": avg_spend,
            "Account_Age_Days": acct_age,
            "Suspicious_Keyword": int(suspicious_kw),
            "spend_ratio": spend_ratio,
            "is_spend_spike": int(spend_ratio > 3),
            "is_new_account": int(acct_age < 30),
            "is_low_history": int(prev_txns < 5),
            "txn_hour": hour,
            "is_night_txn": int(0 <= hour < 5),
            "is_weekend": 0,
        }
        for col, value in [
            ("Merchant_Category", merchant), ("Payment_Method", payment),
            ("Device_Type", device), ("Location", location),
        ]:
            le = encoders[col]
            known = set(le.classes_)
            row[f"{col}_enc"] = int(le.transform([value])[0]) if value in known else -1

        X = pd.DataFrame([row])[FEATURE_COLS]
        rf_proba = float(rf_model.predict_proba(X)[0, 1])
        X_scaled = scaler.transform(X)
        iso_flag = int(iso_model.predict(X_scaled)[0] == -1)
        combined = 0.7 * rf_proba + 0.3 * iso_flag

        if combined >= 0.7:
            risk_level, color = "HIGH RISK", POWERBI_RED
        elif combined >= 0.4:
            risk_level, color = "MEDIUM RISK", POWERBI_AMBER
        else:
            risk_level, color = "LOW RISK", POWERBI_GREEN

        r1, r2, r3 = st.columns(3)
        with r1:
            st.metric("Random Forest Fraud Probability", f"{rf_proba*100:.1f}%")
        with r2:
            st.metric("Isolation Forest Anomaly Flag", "Yes" if iso_flag else "No")
        with r3:
            st.markdown(
                f"""
                <div class="kpi-card" style="border-left: 5px solid {color};">
                <div class="kpi-label">Combined Risk Level</div>
                <div class="kpi-value" style="color:{color}">{risk_level}</div>
                </div>""",
                unsafe_allow_html=True,
            )
