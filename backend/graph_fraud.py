"""
NeuralFraud — Graph-Based Fraud Ring Detection

Important honesty note: this dataset is card-present/card-not-present transaction
data (one row per transaction, no sender->receiver transfer structure). There is
no true account-linkage graph to build here — that would need a P2P/wire-transfer
dataset where money moves between named accounts.

Instead, this module builds a *realistic proxy* used in real fraud teams when
transfer data isn't available: it links customers who share suspicious activity
patterns (same location + same merchant category + same calendar week, all
fraudulent) under the working theory that a cluster of unrelated customers all
defrauded at the same place in the same short window points to a compromised
terminal, a coordinated skimming attack, or a shared bad actor — not coincidence.

Connected components in this graph are treated as candidate "fraud rings."
This is a heuristic, not a certainty — it's presented in the dashboard with
that caveat rather than as confirmed fraud rings.

Run:
    python backend/graph_fraud.py
"""

import json
from pathlib import Path
import networkx as nx
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
PROCESSED_PATH = BASE_DIR / "data" / "processed_transactions.csv"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)

MIN_RING_SIZE = 2  
def load_fraud_transactions() -> pd.DataFrame:
    df = pd.read_csv(PROCESSED_PATH, parse_dates=["Transaction_Date"])
    fraud = df[df["Fraudulent"] == 1].copy()
    fraud["txn_week"] = fraud["Transaction_Date"].dt.to_period("W").astype(str)
    return fraud

def build_graph(fraud_df: pd.DataFrame) -> nx.Graph:
    """
    Adds an edge between two customers if they both had a fraudulent transaction
    at the same Location + Merchant_Category + calendar week. Edge weight = number
    of such shared "incident groups" between the pair (higher = more suspicious).
    """
    G = nx.Graph()
    grouped = fraud_df.groupby(["Location", "Merchant_Category", "txn_week"])
    for (_location, _category, _week), group in grouped:
        customers = group["Customer_ID"].unique().tolist()
        if len(customers) < 2:
            continue  
        for cust in customers:
            G.add_node(cust)

        # Connect every pair of customers who shared this incident group
        for i in range(len(customers)):
            for j in range(i + 1, len(customers)):
                a, b = customers[i], customers[j]
                if G.has_edge(a, b):
                    G[a][b]["weight"] += 1
                    G[a][b]["shared_incidents"].append(
                        {"location": _location, "category": _category, "week": _week}
                    )
                else:
                    G.add_edge(a, b, weight=1,shared_incidents=[{"location": _location, "category": _category, "week": _week}],)
    return G

def extract_rings(G: nx.Graph, fraud_df: pd.DataFrame, min_size: int = MIN_RING_SIZE) -> list[dict]:
    """Connected components of size >= min_size are candidate fraud rings."""
    rings = []
    amount_by_customer = fraud_df.groupby("Customer_ID")["Transaction_Amount"].sum()

    for component in nx.connected_components(G):
        if len(component) < min_size:
            continue

        members = sorted(component)
        subgraph = G.subgraph(members)
        total_value = float(amount_by_customer.reindex(members).fillna(0).sum())
        total_edges = subgraph.number_of_edges()

        rings.append(
            {
                "ring_id": f"ring_{len(rings)+1}",
                "members": members,
                "size": len(members),
                "total_edges": total_edges,
                "total_fraud_value": round(total_value, 2),
                "avg_fraud_value_per_member": round(total_value / len(members), 2),
            }
        )
    rings.sort(key=lambda r: r["total_fraud_value"], reverse=True)
    return rings

def run():
    print("Loading fraudulent transactions...")
    fraud_df = load_fraud_transactions()
    print(f"  {len(fraud_df)} fraudulent transactions found")

    print("Building customer link graph (shared location + category + date)...")
    G = build_graph(fraud_df)
    print(f"  Graph has {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print(f"Extracting connected components with size >= {MIN_RING_SIZE} as candidate rings...")
    rings = extract_rings(G, fraud_df, MIN_RING_SIZE)
    print(f"  Found {len(rings)} candidate fraud ring(s)")

    # Save full graph (nodes + edges) for the dashboard network visualization
    graph_export = {
        "nodes": list(G.nodes()),
        "edges": [
            {"source": u, "target": v, "weight": d["weight"]}
            for u, v, d in G.edges(data=True)
        ],
        "rings": rings,
        "min_ring_size": MIN_RING_SIZE,
    }
    with open(MODELS_DIR / "fraud_graph.json", "w") as f:
        json.dump(graph_export, f, indent=2)

    print(f"\nSaved graph + {len(rings)} candidate ring(s) to {MODELS_DIR / 'fraud_graph.json'}")
    if rings:
        top = rings[0]
        print(
            f"Largest ring: {top['size']} customers, ${top['total_fraud_value']:,.2f} "
            f"total fraud value, {top['total_edges']} shared-incident links"
        )
    return graph_export

if __name__ == "__main__":
    run()
