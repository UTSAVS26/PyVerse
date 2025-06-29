import os
import sys
import json
from datetime import datetime, timedelta
from newspaper import Article
from newsapi import NewsApiClient

current_dir = os.path.dirname(os.path.abspath(__file__))
app_dir = os.path.join(current_dir, "..", "app")
sys.path.append(app_dir)

from summarizer import summarize_text

API_KEY = os.getenv("NEWS_API")
if not API_KEY:
    raise ValueError("❌ NEWS_API is not set in environment variables")

client = NewsApiClient(api_key=API_KEY)

def fetch_news():
    from_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")

    response = client.get_everything(
        q="technology OR politics OR world",
        from_param=from_date,
        to=to_date,
        language="en",
        sort_by="relevancy",
        page_size=50,
    )

    return response.get("articles", [])

def extract_full_article(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except:
        return ""

def main():
    articles = fetch_news()
    results = []
    logs = []

    for art in articles:
        url = art.get("url")
        full_text = extract_full_article(url)

        if not full_text or len(full_text.split()) < 100:
            continue

        summary = summarize_text(full_text)

        results.append({
            "title": art.get("title"),
            "source": art.get("source", {}).get("name"),
            "publishedAt": art.get("publishedAt"),
            "url": url,
            "summary": summary
        })

        logs.append({
            "title": art.get("title"),
            "url": url,
            "word_count": len(full_text.split()),
            "summary_length": len(summary.split())
        })

        if len(results) >= 30:
            break

    os.makedirs("app", exist_ok=True)
    os.makedirs("backend", exist_ok=True)

    with open("app/summaries.json", "w") as f:
        json.dump(results, f, indent=2)

    with open("backend/fetch_log.json", "w") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "article_count": len(results),
            "logs": logs
        }, f, indent=2)

if __name__ == "__main__":
    main()