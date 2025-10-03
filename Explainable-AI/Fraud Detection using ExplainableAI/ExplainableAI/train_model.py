import pandas as pd
import xgboost as xgb
from sklearn.metrics import precision_recall_curve, auc, roc_auc_score
import joblib

# Load data
X_train = pd.read_csv('X_train.csv').values
y_train = pd.read_csv('y_train.csv').values.ravel()
X_test = pd.read_csv('X_test.csv').values
y_test = pd.read_csv('y_test.csv').values.ravel()

# Train
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='aucpr', scale_pos_weight=1)
model.fit(X_train, y_train)

# Evaluate
y_pred_proba = model.predict_proba(X_test)[:, 1]
pr_auc = auc(*precision_recall_curve(y_test, y_pred_proba)[:2])
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"PR-AUC: {pr_auc:.4f}, ROC-AUC: {roc_auc:.4f}")

# Save
joblib.dump(model, 'fraud_model.pkl')
print("Model trained and saved.")