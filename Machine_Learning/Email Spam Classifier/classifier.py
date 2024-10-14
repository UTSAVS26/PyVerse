import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("./spam.csv")

# Convert category to binary (spam=1, ham=0)
df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)

# Split the dataset into features and target variable
X = df['Message']
y = df['spam']

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline that combines CountVectorizer and MultinomialNB
clf = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('nb', MultinomialNB())
])

# Fit the model
clf.fit(X_train, y_train)

# Evaluate the model
accuracy = clf.score(X_test, y_test)
print(f"Model accuracy: {accuracy:.4f}")

# Example emails to classify
emails = [
    'Hey mohan, can we get together to watch the football game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Don’t miss this reward!'
]

# Predict using the pipeline
predictions = clf.predict(emails)

# Print predictions
for email, prediction in zip(emails, predictions):
    label = 'spam' if prediction == 1 else 'ham'
    print(f"Email: '{email}' => Prediction: {label}")
