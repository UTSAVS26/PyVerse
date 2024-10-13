from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import pandas as pd
import base64
import io


def generate_wordcloud(positive_news_df):
    text = " ".join(positive_news_df['Content'].tolist())
    
    wordcloud = WordCloud(width=800, height=400, stopwords=STOPWORDS.update(['S','said','t']), 
                          background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')

    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()
    return plot_url