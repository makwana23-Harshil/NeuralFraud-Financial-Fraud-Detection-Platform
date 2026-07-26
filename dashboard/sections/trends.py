"""NeuralFraud — Trends tab content."""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from common import POWERBI_BLUE, POWERBI_RED, POWERBI_GREEN, transparent_axis_layout

def render(fdf):
    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:38px;font-weight:600;color:#252423;">Trends</h1>""",unsafe_allow_html=True,)
    st.caption(f"Analyzing {len(fdf):,} filtered transactions.")
    st.markdown("")

    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Monthly Transaction Volume vs Fraud Rate</h1>""",unsafe_allow_html=True,)
    weekly = (
        fdf.set_index("transaction_date")
        .resample("W")
        .agg(total=("fraudulent", "count"), fraud=("fraudulent", "sum"))
        .reset_index()
    )
    weekly["fraud_rate_pct"] = (weekly["fraud"] / weekly["total"].replace(0, 1) * 100).round(2)

    fig_weekly = go.Figure()
    fig_weekly.add_trace(
        go.Bar(x=weekly["transaction_date"], y=weekly["total"], name="Total Transactions",
               marker_color=POWERBI_BLUE, opacity=0.5, yaxis="y1")
    )
    fig_weekly.add_trace(
        go.Scatter(x=weekly["transaction_date"], y=weekly["fraud_rate_pct"], name="Fraud Rate (%)",
                   line=dict(color=POWERBI_RED, width=3), yaxis="y2")
    )
    fig_weekly.update_layout(**transparent_axis_layout())
    fig_weekly.update_layout(
        height=380,
        yaxis=dict(title="Total Transactions", showgrid=True, gridcolor="rgba(0,0,0,0.1)"),
        yaxis2=dict(title="Fraud Rate (%)", overlaying="y", side="right", showgrid=False),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig_weekly, width="stretch")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Monthly Transaction Count</h1>""",unsafe_allow_html=True,)
        monthly = (
            fdf.set_index("transaction_date")
            .resample("ME")
            .agg(total=("fraudulent", "count"), fraud=("fraudulent", "sum"))
            .reset_index()
        )
        fig_monthly = go.Figure()
        fig_monthly.add_trace(go.Bar(x=monthly["transaction_date"], y=monthly["total"] - monthly["fraud"],
                                      name="Legitimate", marker_color=POWERBI_GREEN))
        fig_monthly.add_trace(go.Bar(x=monthly["transaction_date"], y=monthly["fraud"],
                                      name="Fraudulent", marker_color=POWERBI_RED))
        fig_monthly.update_layout(**transparent_axis_layout(), barmode="stack", height=340,
                                   legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        #st.plotly_chart(fig_monthly, use_container_width=True)
        st.plotly_chart(fig_monthly, width="stretch")

    with c2:
        st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Average Transaction Amount Over Time</h1>""",unsafe_allow_html=True,)
        amt_trend = (
            fdf.set_index("transaction_date")
            .resample("W")["transaction_amount"]
            .mean()
            .reset_index()
        )
        fig_amt = px.line(amt_trend, x="transaction_date", y="transaction_amount",labels={"transaction_amount": "Avg Amount ($)", "transaction_date": ""})
        fig_amt.update_traces(line=dict(color=POWERBI_BLUE, width=2))
        fig_amt.update_layout(**transparent_axis_layout(), height=340)
        st.plotly_chart(fig_amt, width="stretch")

    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Device Type Usage Over Time</h1>""",unsafe_allow_html=True,)
    device_trend = (
        fdf.groupby([fdf["transaction_date"].dt.to_period("W").dt.start_time, "device_type"])
        .size()
        .reset_index(name="count")
        .rename(columns={"transaction_date": "week"})
    )
    fig_device = px.area(
        device_trend, x="week", y="count", color="device_type",
        color_discrete_sequence=[POWERBI_BLUE, "#12239E", "#E66C37"],
        labels={"count": "Transactions", "week": ""},
    )
    fig_device.update_layout(**transparent_axis_layout(), height=340,
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_device, width="stretch")
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Transaction Amount Distribution — Fraud vs Legitimate</h1>""",unsafe_allow_html=True,)
        fdf_plot = fdf.copy()
        fdf_plot["label"] = fdf_plot["fraudulent"].map({0: "Legitimate", 1: "Fraudulent"})
        fig_dist = px.histogram(
            fdf_plot, x="transaction_amount", color="label", barmode="overlay", nbins=40,
            color_discrete_map={"Legitimate": POWERBI_GREEN, "Fraudulent": POWERBI_RED}, opacity=0.65,
            labels={"transaction_amount": "Transaction Amount ($)"},
        )
        fig_dist.update_layout(**transparent_axis_layout(), height=340,legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_dist, width="stretch")

    with c4:
        st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Fraud Rate by Account Age Bucket</h1>""",unsafe_allow_html=True,)
        bucketed = fdf.copy()
        bucketed["age_bucket"] = pd.cut(
            bucketed["account_age_days"],
            bins=[-1, 30, 180, 365, 730, 100000],
            labels=["<30d", "30-180d", "180-365d", "1-2yr", "2yr+"],
        )
        age_summary = (
            bucketed.groupby("age_bucket", observed=True)
            .agg(total=("fraudulent", "count"), fraud=("fraudulent", "sum"))
            .assign(fraud_rate=lambda d: d["fraud"] / d["total"].replace(0, 1) * 100)
            .reset_index()
        )
        fig_age = px.bar(age_summary, x="age_bucket", y="fraud_rate",
                          color="fraud_rate", color_continuous_scale=["#118DFF", "#D64550"],
                          labels={"fraud_rate": "Fraud Rate (%)", "age_bucket": "Account Age"})
        fig_age.update_layout(**transparent_axis_layout(), height=340, coloraxis_showscale=False)
        st.plotly_chart(fig_age, width="stretch")