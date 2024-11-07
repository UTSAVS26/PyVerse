import pandas as pd
import requests
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk

# Download VADER lexicon for sentiment analysis
nltk.download('vader_lexicon')

# Step 1: Fetching data from World Bank API
def fetch_world_bank_data(indicator, countries):
    url = f"http://api.worldbank.org/v2/country/{countries}/indicator/{indicator}?format=json&date=2000:2024"
    try:
        response = requests.get(url)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        if len(data) < 2:
            raise ValueError(f"No data found for indicator: {indicator}")
        return pd.json_normalize(data[1])
    except Exception as e:
        print(f"Error fetching data for {indicator}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame on error

# Step 2: Data Collection
gdp_data = fetch_world_bank_data("NY.GDP.MKTP.CD", "IN;BR;ZA")
unemployment_data = fetch_world_bank_data("SL.UEM.TOTL.ZS", "IN;BR;ZA")
inflation_data = fetch_world_bank_data("FP.CPI.TOTL", "IN;BR;ZA")
exports_data = fetch_world_bank_data("NE.EXP.GOODS.CD", "IN;BR;ZA")
current_account_data = fetch_world_bank_data("BN.CAB.XOKA.CD", "IN;BR;ZA")

# Step 3: Data Processing
def process_data(df, value_col, rename_cols):
    if df.empty:
        print(f"No data to process for {rename_cols[1]}.")
        return pd.DataFrame(columns=['date', 'Country', rename_cols[1]])
    
    df['date'] = pd.to_datetime(df['date'])
    return df[['date', 'country.value', value_col]].rename(columns={'country.value': rename_cols[0], value_col: rename_cols[1]})

gdp_data = process_data(gdp_data, 'value', ['Country', 'GDP'])
unemployment_data = process_data(unemployment_data, 'value', ['Country', 'Unemployment Rate'])
inflation_data = process_data(inflation_data, 'value', ['Country', 'Inflation Rate'])
exports_data = process_data(exports_data, 'value', ['Country', 'Exports'])
current_account_data = process_data(current_account_data, 'value', ['Country', 'Current Account Balance'])

# Debugging checks
print("GDP Data:")
print(gdp_data.head())  # Check GDP data
print("NaN values in GDP Data:", gdp_data.isnull().sum())

# Merge datasets
merged_data = gdp_data.merge(unemployment_data, on=['date', 'Country'], how='outer') \
                        .merge(inflation_data, on=['date', 'Country'], how='outer') \
                        .merge(exports_data, on=['date', 'Country'], how='outer') \
                        .merge(current_account_data, on=['date', 'Country'], how='outer')

# Debugging check for merged data
print("Merged Data:")
print(merged_data.head())  # Check merged data
print("NaN values in Merged Data:", merged_data.isnull().sum())

# Step 4: Data Visualization
# GDP Growth Visualization
if not gdp_data.empty:
    fig_gdp = px.line(merged_data, x='date', y='GDP', color='Country', title='GDP Growth Over Time')
    fig_gdp.show()
else:
    print("No GDP data available for visualization.")

# Unemployment Rate Visualization
if not unemployment_data.empty:
    fig_unemployment = px.line(merged_data, x='date', y='Unemployment Rate', color='Country', title='Unemployment Rate Over Time')
    fig_unemployment.show()

# Inflation Rate Visualization
if not inflation_data.empty:
    fig_inflation = px.line(merged_data, x='date', y='Inflation Rate', color='Country', title='Inflation Rate Over Time')
    fig_inflation.show()

# Correlation Heatmap
correlation_columns = ['GDP', 'Unemployment Rate', 'Inflation Rate', 'Exports', 'Current Account Balance']
if not merged_data[correlation_columns].empty:
    correlation_matrix = merged_data[correlation_columns].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap of Economic Indicators')
    plt.show()
else:
    print("Not enough data available for correlation analysis.")

# Step 5: Sentiment Analysis on Economic News
def analyze_sentiment(news_articles):
    sia = SentimentIntensityAnalyzer()
    sentiments = []
    for article in news_articles:
        sentiment = sia.polarity_scores(article)
        sentiments.append(sentiment)
    return pd.DataFrame(sentiments)

# Example news articles (replace with actual news data)
news_articles = [
    "India's economy is projected to grow despite global challenges.",
    "Rising unemployment rates are a cause for concern in Brazil.",
    "South Africa's inflation rate is increasing at an alarming rate."
]

sentiment_df = analyze_sentiment(news_articles)
print("Sentiment Analysis of News Articles:")
print(sentiment_df)

# Step 6: Exporting data to CSV
merged_data.to_csv('economic_analysis_data.csv', index=False)
print("Data has been exported to economic_analysis_data.csv")
