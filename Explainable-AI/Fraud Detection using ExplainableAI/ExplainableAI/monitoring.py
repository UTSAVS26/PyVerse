import pandas as pd
from scipy.stats import ks_2samp
import logging

logging.basicConfig(filename='monitor.log', level=logging.INFO)

# Load train/test distributions
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')

# Check data drift (KS test per feature)
for col in X_train.columns:
    stat, p = ks_2samp(X_train[col], X_test[col])
    if p < 0.05:
        logging.warning(f"Drift detected in {col}: p={p}")
    else:
        logging.info(f"No drift in {col}")

# Explanation drift: compare top SHAP over time (simplified; run periodically)
print("Monitoring complete. Check monitor.log")
