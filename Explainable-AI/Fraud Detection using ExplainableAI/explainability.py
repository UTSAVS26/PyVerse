import shap
import dice_ml
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and data
model = joblib.load('fraud_model.pkl')
X_test = pd.read_csv('X_test.csv')
feature_names = X_test.columns.tolist()

# SHAP Explainer
explainer = shap.TreeExplainer(model)
# Use explainer() to get Explanation object, not just shap_values
explanation = explainer(X_test.iloc[:5])  # First 5 rows

# Save SHAP waterfall plots for each transaction
for i in range(len(explanation)):
    fig, ax = plt.subplots()
    shap.plots.waterfall(explanation[i], max_display=10)
    plt.savefig(f'shap_waterfall_{i}.png')
    plt.close()

# DiCE Counterfactuals
data = dice_ml.Data(
    dataframe=pd.concat([X_test.iloc[:5], pd.read_csv('y_test.csv').iloc[:5]], axis=1),
    continuous_features=feature_names,
    outcome_name='Class'
)
ml_model = dice_ml.Model(model=model, backend='sklearn')
expl = dice_ml.Dice(data, ml_model)
cf = expl.generate_counterfactuals(X_test.iloc[:5], total_CFs=1, desired_class="opposite")
cf_df = cf.final_cfs_df
print("Counterfactuals:", cf_df)

# Human-readable summaries
def get_summaries(explanation, instances):
    summaries = []
    for i in range(len(instances)):
        top_features = sorted(
            zip(feature_names, explanation[i].values),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:3]
        summary = f"Txn {i+1}: Flagged due to " + ", ".join([f"{feat} ({val:.2f})" for feat, val in top_features if val > 0])
        summaries.append(summary)
    return summaries

summaries = get_summaries(explanation, X_test.iloc[:5])
for s in summaries:
    print(s)

print("SHAP plots and counterfactuals generated.")