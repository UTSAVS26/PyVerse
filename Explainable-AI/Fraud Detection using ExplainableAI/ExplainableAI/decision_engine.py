import joblib
import shap
import pandas as pd
import numpy as np

model = joblib.load('fraud_model.pkl')
explainer = shap.TreeExplainer(model)

def decide_actions(txn_df):
    scores = model.predict_proba(txn_df)[:, 1]
    explanation = explainer(txn_df)  # Use Explanation object
    actions = []
    for i, score in enumerate(scores):
        top_contrib = max(explanation[i].values, key=abs)  # Highest impact feature
        if score >= 0.9:
            actions.append(f"Txn {i+1}: Auto-block")
        elif 0.6 <= score < 0.9:
            if top_contrib < 0.2:
                actions.append(f"Txn {i+1}: Soft-action (2FA)")
            else:
                actions.append(f"Txn {i+1}: Escalate to analyst")
        elif 0.3 <= score < 0.6:
            actions.append(f"Txn {i+1}: Low-priority review")
        else:
            actions.append(f"Txn {i+1}: Auto-clear")
    return actions, scores, explanation

# Test
sample_txn = pd.read_csv('X_test.csv').iloc[:5]
actions, scores, explanation = decide_actions(sample_txn)
for action, score in zip(actions, scores):
    print(f"{action}, Score: {score:.2%}")