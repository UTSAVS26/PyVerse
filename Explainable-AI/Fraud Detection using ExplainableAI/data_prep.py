import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.under_sampling import RandomUnderSampler

# Load dataset (Kaggle: https://www.kaggle.com/mlg-ulb/creditcardfraud)
df = pd.read_csv('/Users/parthavikurugundla/ExplainableAI/creditcard.csv')

# Engineer features
df['LogAmount'] = np.log1p(df['Amount'])
df['pseudo_user'] = np.random.randint(0, 1000, len(df))  # Fake user ID
df.sort_values(['pseudo_user', 'Time'], inplace=True)
df['txns_last_1h'] = df.groupby('pseudo_user')['Time'].transform(
    lambda x: (x.diff().shift(-1) < 3600).rolling(window=100).sum()
).fillna(0)
df['avg_amount_30d'] = df.groupby('pseudo_user')['Amount'].transform(
    lambda x: x.rolling(window=30, min_periods=1).mean()
).fillna(0)

# Features and target
X = df.drop(['Class', 'Time', 'pseudo_user'], axis=1)  # 32 features
y = df['Class']

# Handle imbalance
rus = RandomUnderSampler()
X_res, y_res = rus.fit_resample(X, y)

# Scale
scaler = StandardScaler()
X_res = scaler.fit_transform(X_res)

# Split (time-based)
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, shuffle=False)

# Save
pd.DataFrame(X_train, columns=X.columns).to_csv('X_train.csv', index=False)
pd.DataFrame(y_train, columns=['Class']).to_csv('y_train.csv', index=False)
pd.DataFrame(X_test, columns=X.columns).to_csv('X_test.csv', index=False)
pd.DataFrame(y_test, columns=['Class']).to_csv('y_test.csv', index=False)
print("Data prepared: X_train.csv, X_test.csv have 32 features.")