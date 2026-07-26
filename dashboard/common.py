"""
NeuralFraud Dashboard — shared utilities
Imported by app.py and every file in dashboard/pages/.
Keeps styling, data loading, and model loading consistent across all pages.
"""

import sqlite3
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "data" / "fraud_detection.db"
MODELS_DIR = BASE_DIR / "models"

POWERBI_BLUE = "#118DFF"
POWERBI_DARK = "#0f0000"
POWERBI_GRAY = "#605E5C"
POWERBI_RED = "#D64550"
POWERBI_GREEN = "#3FB871"
POWERBI_AMBER = "#F2C811"
PALETTE = [POWERBI_BLUE, "#12239E", "#E66C37", "#6B007B", "#E044A7", "#744EC2", "#D9B300", "#D64550"]

FEATURE_COLS = [
    "Transaction_Amount", "Is_International", "Previous_Transactions", "Average_Spend",
    "Account_Age_Days", "Suspicious_Keyword", "spend_ratio", "is_spend_spike",
    "is_new_account", "is_low_history", "txn_hour", "is_night_txn", "is_weekend",
    "Merchant_Category_enc", "Payment_Method_enc", "Device_Type_enc", "Location_enc",
]

def inject_css():
    """Call once at the top of every page. Gradient background + larger sidebar type."""
    st.markdown(
        f"""
        <style>
        .stApp {{
            background: linear-gradient(180deg, #fff8f7 0%, #EAEFF8 100%) !important;
        }}
        section[data-testid="stSidebar"] {{
            background-color: #F3F2F1;
            border-right: 1px solid #E1DFDD;
        }}
        .block-container {{ padding-top: 1.5rem; padding-bottom: 2rem; }}

        h1, h2, h3 {{color: {POWERBI_DARK};font-family: 'Lobster';}}

        .kpi-card {{
            background-color: #FFFFFF;
            border: 3px solid #f9043e;
            border-radius: 6px;
            padding: 18px 20px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.06);
        }}
        .kpi-label {{
            font-size: 13px;
            color: {POWERBI_GRAY};
            font-weight: 650;
            text-transform: uppercase;
            letter-spacing: 0.4px;
        }}
        .kpi-value {{
            font-size: 30px;
            font-weight: 700;
            color: {POWERBI_DARK};
            margin-top: 4px;
        }}
        .kpi-delta-up {{ color: {POWERBI_RED}; font-size: 13px; font-weight: 600; }}
        .kpi-delta-down {{ color: {POWERBI_GREEN}; font-size: 13px; font-weight: 600; }}
        div[data-testid="stDataFrame"] {{ border: 1px solid #E1DFDD; border-radius: 6px; }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def render_sidebar_header():
    """Larger sidebar title + caption, matches the styling you added."""
    st.sidebar.markdown(
        '<p style="font-size: 22px; font-weight: bold; margin-bottom: 0px;">🛡️ NeuralFraud</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown(
        '<p style="font-size: 16px; color: #808495; margin-top: 0px;">Financial Fraud Detection Platform</p>',
        unsafe_allow_html=True,
    )
    st.sidebar.markdown("---")


def render_sidebar_filters_label():
    st.sidebar.markdown(
        '<span style="font-size: 25px; color: #808495 !important; font-weight: bold; margin-top: 0px;">Filters</span>',
        unsafe_allow_html=True,
    )


def render_sidebar_footer():
    st.sidebar.markdown("---")
    st.sidebar.caption("Models: RandomForest (supervised) + IsolationForest (anomaly detection)")
    st.sidebar.caption("Built with FastAPI · scikit-learn · Streamlit · SQLite")


def kpi_card(col, label, value, sublabel=None):
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{value}</div>
                {f'<div class="kpi-delta-up">{sublabel}</div>' if sublabel else ''}
            </div>
            """,
            unsafe_allow_html=True,
        )


def transparent_axis_layout():
    """Standard transparent-background layout w/ soft grid + readable axis text.
    Use for bar/line charts: fig.update_layout(**transparent_axis_layout())"""
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Segoe UI",
        margin=dict(l=10, r=10, t=10, b=10),
        xaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            title_font=dict(color="#31333F", size=14),
            tickfont=dict(color="#555555", size=12),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="rgba(0,0,0,0.1)",
            tickfont=dict(color="#31333F", size=12),
        ),
    )


def transparent_layout_no_grid():
    """For heatmaps/pies where axis gridlines don't apply."""
    return dict(
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font_family="Segoe UI",
        margin=dict(l=10, r=10, t=10, b=10),
    )


@st.cache_data(ttl=300)
def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM transactions", conn)
    conn.close()
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


@st.cache_resource
def load_models():
    rf = joblib.load(MODELS_DIR / "random_forest.joblib")
    iso = joblib.load(MODELS_DIR / "isolation_forest.joblib")
    scaler = joblib.load(MODELS_DIR / "scaler.joblib")
    encoders = joblib.load(MODELS_DIR / "encoders.joblib")
    return rf, iso, scaler, encoders


@st.cache_data(ttl=300)
def load_test_predictions() -> pd.DataFrame:
    return pd.read_csv(MODELS_DIR / "test_predictions.csv")


def apply_sidebar_filters(df: pd.DataFrame):
    """Renders the filter widgets + returns the filtered dataframe.
    Call this once on the Home page; other pages read st.session_state instead
    (see get_filtered_df) so filters don't visually duplicate on every page."""
    render_sidebar_filters_label()
    locations = sorted(df["location"].unique().tolist())
    sel_locations = st.sidebar.multiselect("Location", locations, default=locations)

    categories = sorted(df["merchant_category"].unique().tolist())
    sel_categories = st.sidebar.multiselect("Merchant Category", categories, default=categories)

    payment_methods = sorted(df["payment_method"].unique().tolist())
    sel_payment = st.sidebar.multiselect("Payment Method", payment_methods, default=payment_methods)

    date_min, date_max = df["transaction_date"].min(), df["transaction_date"].max()
    date_range = st.sidebar.date_input(
        "Date range", value=(date_min.date(), date_max.date()),
        min_value=date_min.date(), max_value=date_max.date(),
    )

    st.session_state["sel_locations"] = sel_locations
    st.session_state["sel_categories"] = sel_categories
    st.session_state["sel_payment"] = sel_payment
    st.session_state["date_range"] = date_range

    return filter_df(df, sel_locations, sel_categories, sel_payment, date_range)


def filter_df(df, sel_locations, sel_categories, sel_payment, date_range):
    mask = (
        df["location"].isin(sel_locations)
        & df["merchant_category"].isin(sel_categories)
        & df["payment_method"].isin(sel_payment)
    )
    if len(date_range) == 2:
        mask &= (df["transaction_date"].dt.date >= date_range[0]) & (
            df["transaction_date"].dt.date <= date_range[1]
        )
    return df[mask]


def get_filtered_df(df: pd.DataFrame):
    """Used on pages other than Home: re-applies the filters chosen on Home
    via session_state, without re-rendering the filter widgets."""
    sel_locations = st.session_state.get("sel_locations", sorted(df["location"].unique().tolist()))
    sel_categories = st.session_state.get("sel_categories", sorted(df["merchant_category"].unique().tolist()))
    sel_payment = st.session_state.get("sel_payment", sorted(df["payment_method"].unique().tolist()))
    date_min, date_max = df["transaction_date"].min(), df["transaction_date"].max()
    date_range = st.session_state.get("date_range", (date_min.date(), date_max.date()))
    return filter_df(df, sel_locations, sel_categories, sel_payment, date_range)
