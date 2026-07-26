"""NeuralFraud — Fraud Alerts tab content."""

import plotly.express as px
import streamlit as st
from common import POWERBI_RED, transparent_axis_layout

def render(fdf):
    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:38px;font-weight:600;color:#252423;">Fraud Alerts</h1>""",unsafe_allow_html=True,)
    st.markdown("")
    fraud_df = fdf[fdf["fraudulent"] == 1]
    k1, k2 = st.columns(2)
    with k1:
        st.metric("Total Fraud Alerts (filtered)", f"{len(fraud_df):,}")
    with k2:
        st.metric("Total Value Flagged", f"${fraud_df['transaction_amount'].sum():,.0f}")

    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Daily Alert Volume</h1>""",unsafe_allow_html=True,)    
    daily_alerts = (fraud_df.set_index("transaction_date").resample("D").size().reset_index(name="alert_count"))
    fig = px.bar(daily_alerts, x="transaction_date", y="alert_count",labels={"alert_count": "Alerts", "transaction_date": ""})
    fig.update_traces(marker_color=POWERBI_RED)
    fig.update_layout(**transparent_axis_layout(), height=300)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Recent Fraud Alerts</h1>""",unsafe_allow_html=True,)    
    alerts = (fraud_df.sort_values("transaction_date", ascending=False).head(30)
        [["transaction_id", "customer_id", "transaction_date", "transaction_amount",
          "merchant_category", "location", "payment_method"]]
    )
    st.dataframe(alerts, width="stretch", hide_index=True)