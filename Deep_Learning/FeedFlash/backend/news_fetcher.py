import os
import json
from datetime import datetime, timedelta
from newspaper import Article
from newsapi import NewsApiClient
from app.summarizer import summarize_text

API_KEY = os.getenv("NEWS_API")
if not API_KEY:
    raise ValueError("❌ NEWS_API is not set in environment variables")

client = NewsApiClient(api_key=API_KEY)

def fetch_news(q, from_date, to_date, page_size=50):
    """Fetch news articles for a given query and date range."""
    try:
        response = client.get_everything(
            q=q,
            from_param=from_date,
            to=to_date,
            language="en",
            sort_by="relevancy",
            page_size=page_size,
        )
        return response.get("articles", [])
    except Exception as e:
        print(f"❌ Error fetching news: {e}")
        return []

def extract_full_article(url):
    """Extract full text content from a news article URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception as e:
        print(f"⚠️ Failed to extract article from {url}: {e}")
        return ""

def main():
    print("🚀 Starting news fetching and summarization...")

    from_date = (datetime.utcnow() - timedelta(days=1)).strftime("%Y-%m-%d")
    to_date = datetime.utcnow().strftime("%Y-%m-%d")

    # Set target counts: 20 international, 10 Indian, no genre restrictions
    target_counts = {"international": 20, "india": 10}
    queries = {
        "international": "news OR world OR global OR politics OR economy OR technology",
        "india": "India OR Indian OR Bharat OR Desh"
    }
    results = []
    logs = []

    for category, count in target_counts.items():
        collected = 0
        attempts = 0
        while collected < count and attempts < 10:
            articles = fetch_news(
                q=queries[category],
                from_date=from_date,
                to_date=to_date,
                page_size=50
            )
            for art in articles:
                if collected >= count:
                    break
                url = art.get("url")
                if not url:
                    continue
                full_text = extract_full_article(url)
                word_count = len(full_text.split())
                if not full_text or word_count < 150 or word_count > 2500:
                    continue
                prompt_text = "Summarize the following news article: " + full_text
                results.append({
                    "title": art.get("title"),
                    "source": art.get("source", {}).get("name"),
                    "publishedAt": art.get("publishedAt"),
                    "url": url,
                    "prompt_text": prompt_text,
                    "category": category
                })
                collected += 1
                if collected % 5 == 0:
                    print(f"✅ Collected {collected} {category} articles so far.")
            attempts += 1

    print(f"✅ Finished collecting {len(results)} articles. Starting summarization...")

    summaries = []
    for idx, art in enumerate(results, 1):
        try:
            summary = summarize_text(art["prompt_text"])
        except Exception as e:
            print(f"⚠️ Failed to summarize article {art['url']}: {e}")
            continue
        summaries.append({
            "title": art["title"],
            "source": art["source"],
            "publishedAt": art["publishedAt"],
            "url": art["url"],
            "summary": summary,
            "category": art["category"]
        })
        logs.append({
            "title": art["title"],
            "url": art["url"],
            "word_count": len(art["prompt_text"].split()),
            "summary_length": len(summary.split()),
            "category": art["category"]
        })
        if idx % 5 == 0:
            print(f"📝 Summarized {idx} articles so far.")

    print(f"✅ Successfully summarized {len(summaries)} articles.")

    os.makedirs("app", exist_ok=True)
    os.makedirs("backend", exist_ok=True)

    with open("app/summaries.json", "w", encoding="utf-8") as f:
        json.dump(summaries, f, indent=2)

    with open("backend/fetch_log.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "article_count": len(summaries),
            "logs": logs
        }, f, indent=2)

    print("💾 Results saved to app/summaries.json and backend/fetch_log.json")

if __name__ == "__main__":
    main()