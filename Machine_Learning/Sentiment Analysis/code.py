import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Set up the Streamlit app
st.title("Sentiment Analysis using Naive Bayes")

# Load Dataset
df = pd.read_csv(r'C:\Users\Ananya\OneDrive\Documents\GitHub\PyVerse\Machine_Learning\Sentiment Analysis\test.csv', encoding='ISO-8859-1')


# Select relevant columns for sentiment analysis
df = df[['text', 'sentiment']]

# Remove rows with missing 'text' or 'sentiment' values
df = df.dropna(subset=['text', 'sentiment'])


# Separate features and target variable
X = df['text']
y = df['sentiment']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Vectorize the text data using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)

# Initialize and train the Naive Bayes model
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test_vectorized)

# Evaluate the model's performance
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)

# Create performance metrics DataFrame
performance_metrics = {
    'Label': [],
    'Precision': [],
    'Recall': [],
    'F1-Score': []
}

for label in report.keys():
    if label != 'accuracy':
        performance_metrics['Label'].append(label)
        performance_metrics['Precision'].append(report[label]['precision'])
        performance_metrics['Recall'].append(report[label]['recall'])
        performance_metrics['F1-Score'].append(report[label]['f1-score'])

performance_df = pd.DataFrame(performance_metrics)

# Display results in Streamlit
st.subheader("Model Performance")
st.write(f"Accuracy: {accuracy:.2f}")
st.subheader("Classification Report")
st.write(performance_df)

# Create attractive graphs
st.subheader("Performance Metrics Visualization")
fig, ax = plt.subplots()
sns.set_palette("pastel")
performance_df.set_index('Label').plot(kind='bar', ax=ax)
plt.title("Model Performance Metrics")
plt.xticks(rotation=0)
plt.ylabel("Scores")
plt.grid(axis='y')

# Adding gradient colors
plt.gca().set_facecolor('#f0f0f0')
st.pyplot(fig)

# Generate a word cloud from the text data
st.subheader("Word Cloud Visualization")
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(" ".join(X))
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
st.pyplot(plt)

