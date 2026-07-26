"""NeuralFraud — Overview tab content."""

import plotly.express as px
import streamlit as st

from common import PALETTE, kpi_card, transparent_axis_layout, transparent_layout_no_grid


def render(df, fdf):
    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:38px;font-weight:600;color:#252423;">Fraud Detection Overview</h1>""",unsafe_allow_html=True,)
    st.caption(f"Showing {len(fdf):,} of {len(df):,} transactions")
    st.markdown("")

    total_txns = len(fdf)
    fraud_txns = int(fdf["fraudulent"].sum())
    fraud_rate = (fraud_txns / total_txns * 100) if total_txns else 0
    avg_amount = fdf["transaction_amount"].mean() if total_txns else 0
    fraud_value = fdf.loc[fdf["fraudulent"] == 1, "transaction_amount"].sum() if total_txns else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    kpi_card(k1, "Total Transactions", f"{total_txns:,}")
    kpi_card(k2, "Fraudulent Transactions", f"{fraud_txns:,}")
    kpi_card(k3, "Fraud Rate", f"{fraud_rate:.2f}%")
    kpi_card(k4, "Avg Transaction", f"${avg_amount:,.2f}")
    kpi_card(k5, "Fraud Value at Risk", f"${fraud_value:,.0f}")

    st.markdown("")
    st.markdown("")

    c1, c2 = st.columns([1, 1.3])

    with c1:
        st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Fraud Rate by Merchant Category</h1>""",unsafe_allow_html=True,)

        cat_summary = (
            fdf.groupby("merchant_category")
            .agg(total=("fraudulent", "count"), fraud=("fraudulent", "sum"))
            .assign(fraud_rate=lambda d: d["fraud"] / d["total"] * 100)
            .sort_values("fraud_rate", ascending=True)
            .reset_index()
        )
        fig = px.bar(
            cat_summary, x="fraud_rate", y="merchant_category", orientation="h",
            color="fraud_rate", color_continuous_scale=["#118DFF", "#D64550"],
            labels={"fraud_rate": "Fraud Rate (%)", "merchant_category": ""},
        )
        fig.update_layout(**transparent_axis_layout(), coloraxis_showscale=False, height=340)
        #st.plotly_chart(fig, use_container_width=True)
        st.plotly_chart(fig, width="stretch")

    with c2:
        st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Fraud Heatmap — Location vs Hour of Day</h1>""",unsafe_allow_html=True,)
        
        heat = (
            fdf[fdf["fraudulent"] == 1]
            .groupby(["location", "txn_hour"])
            .size()
            .reset_index(name="count")
        )
        heat_pivot = heat.pivot(index="location", columns="txn_hour", values="count").fillna(0)
        fig3 = px.imshow(
            heat_pivot, color_continuous_scale=["#FFFFFF", "#118DFF", "#D64550"],
            labels=dict(x="Hour of Day", y="Location", color="Fraud Count"), aspect="auto",
        )
        fig3.update_layout(**transparent_layout_no_grid(), height=340)
        #st.plotly_chart(fig3, use_container_width=True)
        st.plotly_chart(fig3, width="stretch")

    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Fraud by Payment Method</h1>""",unsafe_allow_html=True,)

    pm_summary = fdf[fdf["fraudulent"] == 1]["payment_method"].value_counts().reset_index()
    pm_summary.columns = ["payment_method", "count"]
    fig4 = px.pie(
        pm_summary, names="payment_method", values="count",
        color_discrete_sequence=PALETTE, hole=0.55,
    )
    fig4.update_layout(**transparent_layout_no_grid(), height=320, showlegend=True,
                        legend=dict(orientation="h", yanchor="top", y=-0.1))
    #st.plotly_chart(fig4, use_container_width=True)
    st.plotly_chart(fig4, width="stretch")