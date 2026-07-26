"""
NeuralFraud Dashboard — Fraud Network Analysis
Visualizes customer relationships, fraud rings, and suspicious transaction
networks using NetworkX + Plotly.
"""
import json
import networkx as nx
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from common import (MODELS_DIR, POWERBI_BLUE, POWERBI_RED,kpi_card, transparent_layout_no_grid,)
GRAPH_PATH = MODELS_DIR / "fraud_graph.json"
@st.cache_data(ttl=300)
def load_graph_json():
    if not GRAPH_PATH.exists():
        return None
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def render(fdf):
    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:38px;font-weight:600;color:#252423;">Fraud Network Analysis</h1>""",unsafe_allow_html=True,)
    st.caption("Relationship analysis between customers to identify organized ""fraud rings and suspicious transaction networks.")

    graph_json = load_graph_json()
    if graph_json is None:
        st.warning("No graph data found. Run `python backend/graph_fraud.py` after "
        "training the models to generate `models/fraud_graph.json`."
        )
        return

    nodes = graph_json["nodes"]
    edges = graph_json["edges"]
    rings = graph_json["rings"]

    if not nodes or not rings:
        st.info("No fraud network data available yet — not enough shared-incident links found.")
        return

    G = nx.Graph()
    for node in nodes:
        G.add_node(node)
    for edge in edges:
        G.add_edge(edge["source"], edge["target"], weight=edge.get("weight", 1))

    # ---------------- Summary + ring size filter ----------------
    left, right = st.columns([1, 3])
    with left:
        st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Network Summary</h1>""",unsafe_allow_html=True,)
        st.metric("Customers", len(nodes))
        st.metric("Connections", len(edges))
        st.metric("Fraud Rings", len(rings))
        largest_ring = max(rings, key=lambda x: x["size"])
        st.metric("Largest Ring", largest_ring["size"])
        min_ring = st.slider("Minimum Ring Size", 2, 10, graph_json.get("min_ring_size", 2), key="min_ring_slider")
    
    filtered_rings = [r for r in rings if r["size"] >= min_ring]
    # Build a set of all customers that belong to fraud rings
    ring_members = set()
    for ring in filtered_rings:
        ring_members.update(ring["members"])

    # ==========================================================
    # Fraud Ring Hierarchy (Sunburst)
    # ==========================================================
    with right:
        st.markdown("""<h1 style="text-align: center; font-family:'Segoe UI', sans-serif; font-size:28px; font-weight:600; color:#252423;">Fraud Ring Hierarchy</h1>""",unsafe_allow_html=True)
        sunburst_rows = []
        # Root Node
        total_fraud = sum(ring["total_fraud_value"]
        for ring in filtered_rings
        )
        sunburst_rows.append(
            {
                "id": "Fraud Network",
                "label": "Fraud Network",
                "parent": "",
                "value": total_fraud,
            }
        )

        # Ring + Customer Nodes
        for ring in filtered_rings:
            ring_id = ring["ring_id"]
            fraud_amount = ring["total_fraud_value"]
            members = ring["members"]

            # Ring
            sunburst_rows.append(
                {"id": ring_id,"label": ring_id,"parent": "Fraud Network","value": fraud_amount,}
            )
            # Customers
            member_value = max(
                fraud_amount / max(len(members), 1),1,
            )
            for customer in members:
                sunburst_rows.append(
                    {"id": customer,"label": customer,"parent": ring_id,"value": member_value,}
                )
        sunburst_df = pd.DataFrame(sunburst_rows)
        fig = go.Figure(
            go.Sunburst(
                ids=sunburst_df["id"],
                labels=sunburst_df["label"],
                parents=sunburst_df["parent"],
                values=sunburst_df["value"],
                branchvalues="total",
                insidetextorientation="radial",
                maxdepth=3,
                hovertemplate=
                "<b>%{label}</b><br>"
                "Fraud Amount: $%{value:,.2f}"
                "<extra></extra>",
            )
        )

        fig.update_layout(
            height=650,
            margin=dict(t=40,l=20,r=20,b=20,),
            paper_bgcolor="rgba(0,0,0,0)",   # Transparent outside the plot
            plot_bgcolor="rgba(0,0,0,0)",
        )

        st.plotly_chart(fig,use_container_width=True,)
        st.markdown("""<p style="text-align: center; color: gray; font-size: 14px;">
        Center → Fraud Network | Middle → Fraud Rings | Outer → Customers</p>""",unsafe_allow_html=True,)
    st.divider()

    # ---------------- Fraud ring summary (kept as table — exact figures matter here) ----------------
    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Fraud Ring Intelligence</h1>""",unsafe_allow_html=True,)
    if not filtered_rings:
        st.info("No fraud rings satisfy the selected filter.")
        return

    ring_df = pd.DataFrame(filtered_rings)
    ring_df["Customer IDs"] = ring_df["members"].apply(lambda x: ", ".join(x))
    ring_df = ring_df.sort_values("total_fraud_value", ascending=False)
    safe_edges = ring_df["total_edges"].replace(0, 1)
    ring_df["Risk Score"] = ((ring_df["total_fraud_value"] * ring_df["size"]) / safe_edges).round(2)

    ring_df.rename(
        columns={"ring_id": "Ring ID", "size": "Members", "total_edges": "Connections",
            "total_fraud_value": "Fraud Amount", "avg_fraud_value_per_member": "Average / Member",
        },
        inplace=True,
    )

    st.dataframe(
        ring_df[["Ring ID","Customer IDs", "Members", "Connections", "Fraud Amount", "Average / Member", "Risk Score"]],
        use_container_width=True, hide_index=True,
    )
    csv = ring_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download Fraud Ring Report",
        data=csv,
        file_name="fraud_ring_report.csv",
        mime="text/csv",
    )
    highest = ring_df.iloc[0]
    c1, c2, c3 = st.columns(3)
    kpi_card(c1, "Highest Risk Ring", highest["Ring ID"])
    kpi_card(c2, "Fraud Amount", f"${highest['Fraud Amount']:.2f}")
    kpi_card(c3, "Risk Score", f"{highest['Risk Score']}")
    st.divider()

    # ---------------- Fraud Ring Activity Trend (time-based) ----------------
    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Fraud Ring Activity Trend</h1>""",unsafe_allow_html=True,)
    st.caption(
        "Weekly fraud value over time, split between customers currently flagged as ring "
        "members vs. everyone else. Reflects the filters set on the Overview tab."
    )
    fraud_txns = fdf[fdf["fraudulent"] == 1].copy()
    if fraud_txns.empty:
        st.info("No fraudulent transactions in the current filtered range.")
    else:
        fraud_txns["ring_status"] = fraud_txns["customer_id"].apply(
            lambda c: "Ring Member" if c in ring_members else "Other Fraud"
        )
        weekly_trend = (
            fraud_txns.set_index("transaction_date")
            .groupby("ring_status")
            .resample("W")["transaction_amount"]
            .sum()
            .reset_index()
        )
        fig_trend = px.area(
            weekly_trend, x="transaction_date", y="transaction_amount", color="ring_status",
            color_discrete_map={"Ring Member": POWERBI_RED, "Other Fraud": POWERBI_BLUE},
            labels={"transaction_amount": "Fraud Value ($)", "transaction_date": "", "ring_status": ""},
        )
        fig_trend.update_layout(
            **transparent_layout_no_grid(), height=360,
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_trend, use_container_width=True)
    st.divider()

    # ---------------- Customer Connection Timeline ----------------
    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Customer Connection Timeline</h1>""",unsafe_allow_html=True,)
    # Customer options with payment method
    customer_options = (
        fdf[["customer_id", "payment_method"]]
        .drop_duplicates()
        .sort_values(["customer_id", "payment_method"])
    )
    option_map = {
        f"{row['customer_id']} | {row['payment_method']}": row["customer_id"]
        for _, row in customer_options.iterrows()
    }
    selected_labels = st.multiselect("Select Customer(s)",options=list(option_map.keys()),default=[],key="customer_timeline",)
    selected_customers = [option_map[label]
        for label in selected_labels
    ]
    if not selected_customers:
        st.info("Please select one or more customers to view their activity timeline.")
    else:
        customer_txns = fdf[fdf["customer_id"].isin(selected_customers)].copy()
        customer_txns["Month"] = pd.to_datetime(customer_txns["transaction_date"]).dt.to_period("M").astype(str)
        monthly_activity = (customer_txns.groupby(["customer_id", "Month"]).size().reset_index(name="Transactions"))
        fig = px.line(monthly_activity,x="Month",y="Transactions",color="customer_id",markers=True,title="Customer Transaction Activity Timeline",)
        fig.update_layout(xaxis_title="Month",yaxis_title="Number of Transactions",legend_title="Customer",hovermode="x unified",)
        st.plotly_chart(fig, width="stretch")
        st.markdown("### Summary")
    
        col1, col2, col3 = st.columns(3)
        col1.metric("Selected Customers",len(selected_customers),)
        col2.metric("Total Transactions",len(customer_txns),)
        col3.metric("Fraudulent Transactions",int(customer_txns["fraudulent"].sum()),)
        st.markdown("### Transaction History")
        history = customer_txns[[
            "customer_id","payment_method","transaction_date","transaction_id","transaction_amount","fraudulent",]].sort_values("transaction_date", ascending=False)
        st.dataframe(history,width="stretch",hide_index=True,)
        st.download_button(label="Download Customer History",data=history.to_csv(index=False).encode("utf-8"),file_name="customer_history.csv",mime="text/csv",)

    # ---------------- Overall network stats ----------------
    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Overall Network Statistics</h1>""",unsafe_allow_html=True,)
    s1, s2, s3, s4 = st.columns(4)
    kpi_card(s1, "Nodes", f"{G.number_of_nodes()}")
    kpi_card(s2, "Edges", f"{G.number_of_edges()}")
    kpi_card(s3, "Density", f"{nx.density(G):.3f}")
    kpi_card(s4, "Components", f"{nx.number_connected_components(G)}")
    st.divider()
    st.caption("NeuralFraud Network Analytics • Powered by NetworkX, Plotly and Streamlit")
