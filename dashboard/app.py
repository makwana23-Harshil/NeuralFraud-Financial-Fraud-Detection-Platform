"""
NeuralFraud Dashboard — single-page app with a horizontal, centered top navbar.
Run:
    streamlit run dashboard/app.py
"""

import streamlit as st
from common import inject_css, load_data, load_models, apply_sidebar_filters, render_sidebar_header, render_sidebar_footer
from sections import overview, trends, model_insights, risk_checker, fraud_alerts, fraud_network

st.set_page_config(
    page_title="NeuralFraud | Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_css()

# ---------------- Style st.tabs() to look like a centered top navbar ----------------
st.markdown(
    """
    <style>
    div[data-baseweb="tab-list"] {
        justify-content: center;
        gap: 8px;
        border-bottom: 1px solid #E1DFDD;
        margin-bottom: 1rem;
    }
    div[data-baseweb="tab-list"] button {
        font-size: 16px;
        font-weight: 600;
        padding: 10px 20px;
    }
    div[data-baseweb="tab-list"] button[aria-selected="true"] {
        border-bottom: 3px solid #118DFF;
        color: #118DFF !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar: filters only ----------------
df = load_data()
rf_model, iso_model, scaler, encoders = load_models()

render_sidebar_header()
fdf = apply_sidebar_filters(df)
render_sidebar_footer()

# ---------------- Horizontal centered navbar ----------------
tab_home, tab_trends, tab_model, tab_network, tab_risk, tab_alerts = st.tabs(
    ["Overview", "Trends", "Model Insights", "Fraud Network", "Risk Checker", "Fraud Alerts"]
)

with tab_home:
    overview.render(df, fdf)

with tab_trends:
    trends.render(fdf)

with tab_model:
    model_insights.render(rf_model)

with tab_network:
    fraud_network.render(fdf)

with tab_risk:
    risk_checker.render(df, rf_model, iso_model, scaler, encoders)

with tab_alerts:
    fraud_alerts.render(fdf)
