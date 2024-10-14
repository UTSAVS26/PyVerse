from flask import Flask, render_template, request, jsonify
import json
import os
from data_collection import scrape_news
from visualization import generate_wordcloud
from sentiment_analysis import analyze_sentiment

app = Flask(__name__)

# File to store bookmarks
BOOKMARK_FILE = 'bookmarks.json'

# Load bookmarks from file
def load_bookmarks():
    if os.path.exists(BOOKMARK_FILE):
        with open(BOOKMARK_FILE, 'r') as f:
            bookmarks = json.load(f)
            return [int(bookmark) for bookmark in bookmarks]
    return []

# Save bookmarks to file
def save_bookmarks(bookmarks):
    with open(BOOKMARK_FILE, 'w') as f:
        json.dump(bookmarks, f)



news_df = scrape_news()
positive_news_df = analyze_sentiment(news_df)
wordcloud_url = generate_wordcloud(positive_news_df)
articles = positive_news_df.to_dict(orient='records')

@app.route('/')
def home():
    return render_template('index.html', articles=articles, wordcloud_url=wordcloud_url)


@app.route('/bookmark/<article_id>', methods=['POST'])
def bookmark(article_id):
    bookmarks = load_bookmarks()
    if int(article_id) not in bookmarks:
        bookmarks.append(int(article_id))                         ########## added int(articleid)
        save_bookmarks(bookmarks)
        return jsonify(success=True)
    return jsonify(success=False)



@app.route('/removebookmark/<article_id>', methods=['POST'])
def removebookmark(article_id):
    bookmarks = load_bookmarks()
    if int(article_id) in bookmarks:
        bookmarks.remove(int(article_id))
        save_bookmarks(bookmarks)
        # return jsonify(success=True)
        '''reload the page'''
        bookmarks = load_bookmarks()
        bookmarked_articles = [article for article in articles if article['id'] in bookmarks]
        return jsonify(success=True), render_template('bookmarks.html', articles=bookmarked_articles)
        
    return jsonify(success=False)



@app.route('/bookmarks')
def list_bookmarks():
    bookmarks = load_bookmarks()
    bookmarked_articles = [article for article in articles if article['id'] in bookmarks]
    return render_template('bookmarks.html', articles=bookmarked_articles)

if __name__ == '__main__':
    app.run(debug=True)