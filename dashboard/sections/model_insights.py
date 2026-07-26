"""NeuralFraud — Model Insights tab content."""

import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, auc
from common import POWERBI_BLUE, MODELS_DIR, FEATURE_COLS, kpi_card, transparent_axis_layout, transparent_layout_no_grid, load_test_predictions

def render(rf_model):
    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:38px;font-weight:600;color:#252423;">Model Insights</h1>""",unsafe_allow_html=True,)
    st.caption("Evaluation metrics computed on the held-out test set at training time (backend/train_model.py)")
    st.markdown("")

    test_df = load_test_predictions()
    with open(MODELS_DIR / "metadata.json") as f:
        metadata = json.load(f)

    k1, k2, k3, k4 = st.columns(4)
    kpi_card(k1, "Random Forest ROC-AUC", f"{metadata['rf_roc_auc']:.3f}")
    kpi_card(k2, "Logistic Regression ROC-AUC", f"{metadata.get('logreg_roc_auc', 0):.3f}")
    kpi_card(k3, "Train Fraud Rate", f"{metadata['fraud_rate_train']*100:.2f}%")
    kpi_card(k4, "Test Set Size", f"{metadata['n_test']:,}")
    st.markdown("")

    if metadata.get("logreg_roc_auc", 0) > metadata["rf_roc_auc"]:
        st.info(f"On this dataset, **Logistic Regression** (ROC-AUC {metadata['logreg_roc_auc']:.3f}) "
            f"slightly outperforms **Random Forest** (ROC-AUC {metadata['rf_roc_auc']:.3f})."
        )

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">ROC Curve — Random Forest vs Logistic Regression</h1>""",unsafe_allow_html=True,)
        fpr, tpr, _ = roc_curve(test_df["y_true"], test_df["rf_proba"])
        roc_auc = auc(fpr, tpr)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"Random Forest (AUC={roc_auc:.3f})",line=dict(color=POWERBI_BLUE, width=3)))
        if "logreg_proba" in test_df.columns:
            fpr_lr, tpr_lr, _ = roc_curve(test_df["y_true"], test_df["logreg_proba"])
            auc_lr = auc(fpr_lr, tpr_lr)
            fig_roc.add_trace(go.Scatter(x=fpr_lr, y=tpr_lr, mode="lines", name=f"Logistic Regression (AUC={auc_lr:.3f})",
                                          line=dict(color="#E66C37", width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random baseline",
                                      line=dict(color="gray", width=1, dash="dash")))
        fig_roc.update_layout(**transparent_axis_layout(), height=360,
                               xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
                               legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fig_roc, use_container_width=True)

    with c2:
        st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Confusion Matrix (Random Forest, threshold=0.5)</h1>""",unsafe_allow_html=True,)
        cm = confusion_matrix(test_df["y_true"], test_df["rf_pred"])
        fig_cm = px.imshow(
            cm, text_auto=True, color_continuous_scale=["#FFFFFF", POWERBI_BLUE],
            labels=dict(x="Predicted", y="Actual", color="Count"),
            x=["Legitimate", "Fraud"], y=["Legitimate", "Fraud"],
        )
        fig_cm.update_layout(**transparent_layout_no_grid(), height=360)
        st.plotly_chart(fig_cm, use_container_width=True)
    
    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">What Drives Fraud Predictions (Random Forest Feature Importance)</h1>""",unsafe_allow_html=True,)
    importances = (pd.Series(rf_model.feature_importances_, index=FEATURE_COLS)
        .sort_values(ascending=True)
        .tail(12)
    )
    fig_imp = px.bar(x=importances.values, y=importances.index, orientation="h",
        color=importances.values, color_continuous_scale=["#E1DFDD", POWERBI_BLUE],
        labels={"x": "Importance", "y": ""},
    )
    fig_imp.update_layout(**transparent_axis_layout(), coloraxis_showscale=False, height=380)
    st.plotly_chart(fig_imp, use_container_width=True)
    st.markdown("""<h1 style="font-family:'Segoe UI', sans-serif;font-size:28px;font-weight:600;color:#252423;">Supervised vs Unsupervised Agreement</h1>""",unsafe_allow_html=True,)

    st.caption("How often the two models agree on the test set. Isolation Forest doesn't see labels during "
        "training, so lower agreement is expected — it's catching a different signal (statistical "
        "outliers) than the label-driven Random Forest.")
    agree = (test_df["rf_pred"] == test_df["iso_pred"]).mean() * 100
    both_flag = ((test_df["rf_pred"] == 1) & (test_df["iso_pred"] == 1)).sum()
    rf_only = ((test_df["rf_pred"] == 1) & (test_df["iso_pred"] == 0)).sum()
    iso_only = ((test_df["rf_pred"] == 0) & (test_df["iso_pred"] == 1)).sum()
    a1, a2, a3, a4 = st.columns(4)
    kpi_card(a1, "Agreement Rate", f"{agree:.1f}%")
    kpi_card(a2, "Flagged by Both", f"{both_flag}")
    kpi_card(a3, "RF Only", f"{rf_only}")
    kpi_card(a4, "Isolation Forest Only", f"{iso_only}")
