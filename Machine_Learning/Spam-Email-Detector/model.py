
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import os
# 1. Load dataset


csv_path = os.path.join(os.path.dirname(__file__), "synthetic_sms_data.csv")
try:
   df = pd.read_csv(csv_path)
   print("Dataset loaded successfully.")
   print(f"Dataset shape: {df.shape}")
    print(df.head())
except FileNotFoundError:
    print(f"Error: Could not find {csv_path}")
    exit(1)
except Exception as e:
   print(f"Error loading dataset: {e}")
    exit(1)
# Validate required columns
required_columns = ['text', 'spam']
missing_columns = [col for col in required_columns if col not in df.columns]
if missing_columns:
   print(f"Error: Missing required columns: {missing_columns}")
   print(f"Available columns: {list(df.columns)}")
   exit(1)

# 2. Train-test split
X = df['text']
y = df['spam']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Create pipeline: TF-IDF + Logistic Regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('model', LogisticRegression(max_iter=1000))
])

# 4. Train the model
pipeline.fit(X_train, y_train)

# 5. Predict
y_pred = pipeline.predict(X_test)

# 6. Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print("\n Evaluation Metrics:")
print(f" Accuracy:  {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall:    {recall:.4f}")
print(f" F1 Score:  {f1:.4f}")

print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Ham', 'Spam']))

# 7. Confusion Matrix Plot
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(' Confusion Matrix')
plt.tight_layout()
plt.show()
