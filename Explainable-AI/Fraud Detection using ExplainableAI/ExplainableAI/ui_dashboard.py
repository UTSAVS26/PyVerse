import streamlit as st
import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import os

# Load model and explainer
@st.cache_resource
def load_model_and_explainer():
    model = joblib.load('fraud_model.pkl')
    explainer = shap.TreeExplainer(model)
    return model, explainer

model, explainer = load_model_and_explainer()

st.title("Fraud Detection Analyst Dashboard")

# Load feature names
try:
    X_test = pd.read_csv('X_test.csv')
    feature_names = X_test.columns.tolist()
    if len(feature_names) != 32:
        st.error(f"X_test.csv has {len(feature_names)} features, expected 32. Rerun data_prep.py.")
        st.stop()
    st.write("**Feature Names (for debugging):**", feature_names)
except FileNotFoundError:
    st.error("X_test.csv not found. Run data_prep.py first.")
    st.stop()

# Create sample transactions (5 rows from X_test.csv)
sample_df = X_test.iloc[:5]

# UI Controls
use_upload = st.checkbox("Upload a transaction CSV (must have 32 features)", value=False)
if use_upload:
    uploaded_file = st.file_uploader("Upload engineered transaction CSV")
    if uploaded_file is not None:
        txn_df = pd.read_csv(uploaded_file)
        if list(txn_df.columns) != feature_names:
            st.error(f"Invalid CSV: Columns don't match X_test.csv. Got {list(txn_df.columns)}.")
            st.stop()
        st.success(f"Uploaded {len(txn_df)} transactions!")
    else:
        st.info("No file uploaded. Using sample (5 transactions).")
        txn_df = sample_df
else:
    st.info("Using sample transactions (first 5 rows from X_test.csv).")
    txn_df = sample_df

# Compute scores and explanations
try:
    scores = model.predict_proba(txn_df)[:, 1]
    explanation = explainer(txn_df)  # Use Explanation object
except ValueError as e:
    st.error(f"Prediction failed: {e}. Ensure CSV has 32 features matching X_test.csv.")
    st.write("txn_df columns:", list(txn_df.columns))
    st.stop()

# Display results in a table
st.subheader("Transaction Analysis")
results = []
for i, (score, shap_val) in enumerate(zip(scores, explanation)):
    top_contribs = sorted(
        zip(feature_names, shap_val.values),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:3]
    summary = ", ".join([f"{feat} ({val:.2f})" for feat, val in top_contribs if val > 0])
    action = "Escalate to analyst" if score >= 0.6 else "Soft-action (2FA)" if score >= 0.3 else "Auto-clear"
    results.append({
        "Txn ID": i+1,
        "Fraud Score": f"{score:.2%}",
        "Top Contributors": summary or "None (low risk)",
        "Action": action
    })
results_df = pd.DataFrame(results)
st.dataframe(results_df.style.highlight_max(subset=["Fraud Score"], color='lightcoral'))

# SHAP Waterfall for selected transaction
st.subheader("Detailed Explanation for a Transaction")
txn_id = st.selectbox("Select Transaction ID for SHAP Plot", options=list(range(1, len(txn_df)+1)))
if txn_id:
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation[txn_id-1], max_display=10)
    st.pyplot(fig)

# Counterfactuals (simplified: reduce Amount)
st.subheader("Counterfactual Suggestions")
cf_results = []
for i in range(len(txn_df)):
    if 'Amount' in feature_names:
        cf_df = txn_df.iloc[[i]].copy()
        orig_amount = cf_df['Amount'].iloc[0]
        cf_df['Amount'] *= 0.3
        cf_df['LogAmount'] = np.log1p(cf_df['Amount'])
        try:
            cf_score = model.predict_proba(cf_df)[:, 1][0]
            cf_results.append({
                "Txn ID": i+1,
                "Original Amount": f"${orig_amount:.0f}",
                "New Amount": f"${cf_df['Amount'].iloc[0]:.0f}",
                "New Score": f"{cf_score:.2%}"
            })
        except ValueError as e:
            cf_results.append({"Txn ID": i+1, "Note": f"Counterfactual failed: {e}"})
    else:
        cf_results.append({"Txn ID": i+1, "Note": "Amount feature missing"})
st.dataframe(pd.DataFrame(cf_results))

# Transaction Timeline (dummy)
st.subheader("Recent Transaction Timeline (Sample)")
timeline_data = pd.DataFrame({
    'Time': [f"Now - Txn {i+1}" for i in range(len(txn_df))] + ['-1h', '-2h'],
    'Amount': list(txn_df['Amount']) + [50, 100],
    'Score': list(scores) + [0.1, 0.05],
    'Explanation': [results[i]["Top Contributors"] for i in range(len(txn_df))] + ['Normal', 'Normal']
})
st.dataframe(timeline_data)

# Feedback
st.subheader("Feedback Loop")
for i in range(len(txn_df)):
    st.write(f"Feedback for Txn {i+1}")
    feedback = st.selectbox(f"Is Txn {i+1} fraud?", ["Legitimate (False Positive)", "Fraud"], key=f"fb_{i}")
    if st.button(f"Submit Feedback for Txn {i+1}", key=f"btn_{i}"):
        with open('feedback.log', 'a') as f:
            f.write(f"{txn_df.iloc[[i]].to_json(orient='records')}|{scores[i]}|{feedback}\n")
        st.success(f"Feedback for Txn {i+1} logged!")

# API Test
st.subheader("Test API Endpoint (Batch)")
if st.button("Call Scoring API"):
    try:
        response = requests.post("http://127.0.0.1:8000/score_batch",
                                json={"transactions": txn_df.values.tolist()})
        st.json(response.json())
    except Exception as e:
        st.error(f"API error: {e}. Ensure FastAPI is running on port 8000.")