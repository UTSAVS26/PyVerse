from textblob import TextBlob
import pandas as pd
# from data_collection import scrape_news

def analyze_sentiment(news_df):
    def analyze(text):
        return TextBlob(text).sentiment.polarity
    
    news_df['sentiment'] = news_df['Content'].apply(analyze)     
    positive_news_df = news_df[news_df['sentiment'] > 0.2]
    positive_news_df.to_csv('positive_news_filtered.csv', index=False)
    return positive_news_df   

# analyze_sentiment(scrape_news())