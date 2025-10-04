from fastapi import FastAPI
import joblib
import shap
import pandas as pd
from pydantic import BaseModel
from typing import List

app = FastAPI()
model = joblib.load('fraud_model.pkl')
explainer = shap.TreeExplainer(model)
feature_names = pd.read_csv('X_test.csv').columns.tolist()

class TransactionBatch(BaseModel):
    transactions: List[List[float]]

@app.post("/score_batch")
def score_batch(txn: TransactionBatch):
    df = pd.DataFrame(txn.transactions, columns=feature_names)
    if df.shape[1] != 32:
        return {"error": f"Expected 32 features, got {df.shape[1]}"}
    scores = model.predict_proba(df)[:, 1]
    explanation = explainer(df)
    results = []
    for i, (score, shap_val) in enumerate(zip(scores, explanation)):
        top_contribs = sorted(
            zip(feature_names, shap_val.values),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        summary = f"Txn {i+1}: Top contributors: " + ", ".join([f"{feat} ({val:.2f})" for feat, val in top_contribs])
        action = "Escalate" if score > 0.5 else "Clear"
        results.append({"txn_id": i+1, "score": score, "explanation": summary, "action": action})
    return {"results": results}