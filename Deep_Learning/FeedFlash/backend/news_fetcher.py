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
        return None

def chunk_text(text, min_words=450, max_words=500):
    """
    Chunk text into pieces for summarization.
    Each chunk will have up to max_words (default 500) and at least min_words (default 450) if possible.
    """
    words = text.split()
    chunks = []
    start = 0
    n = len(words)
    while start < n:
        end = min(start + max_words, n)
        if end - start < min_words and end < n:
            end = min(start + min_words, n)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end
    return chunks

def summarize_long_article(full_text):
    """
    Summarizes a long article by:
    1. Splitting into chunks,
    2. Summarizing each chunk,
    3. Summarizing the concatenated chunk summaries for coherence.
    """
    chunks = chunk_text(full_text)
    chunk_summaries = []
    for chunk in chunks:
        try:
            summary = summarize_text("Summarize the following news article: " + chunk)
        except Exception as e:
            print(f"⚠️ Failed to summarize chunk: {e}")
            summary = ""
        chunk_summaries.append(summary)
    combined = " ".join(chunk_summaries)
    try:
        final_summary = summarize_text("Summarize the following combined summaries: " + combined)
    except Exception as e:
        print(f"⚠️ Failed to summarize combined summary: {e}")
        final_summary = combined  # fallback
    return final_summary

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
                if not full_text:
                    continue
                word_count = len(full_text.split())
                # Strictly enforce word count between 200 and 1500
                if word_count < 200 or word_count > 1500:
                    print(f"⚠️ Skipping article ({word_count} words) not in 200-1500 word range: {url}")
                    continue

                # Summarize long articles with chunking and recursive summarization
                if word_count > 450:
                    combined_summary = summarize_long_article(full_text)
                else:
                    try:
                        combined_summary = summarize_text("Summarize the following news article: " + full_text)
                    except Exception as e:
                        print(f"⚠️ Failed to summarize article: {e}")
                        combined_summary = "Summary not available due to error."

                results.append({
                    "title": art.get("title"),
                    "source": art.get("source", {}).get("name"),
                    "publishedAt": art.get("publishedAt"),
                    "url": url,
                    "prompt_text": full_text,
                    "summary": combined_summary,
                    "category": category
                })
                collected += 1
                if collected % 5 == 0:
                    print(f"✅ Collected {collected} {category} articles so far.")
            attempts += 1

        if collected < count:
            print(f"⚠️ Only collected {collected}/{count} {category} articles after {attempts} attempts")

    print(f"✅ Finished collecting {len(results)} articles. Summaries generated.")

    # Save summaries and logs
    os.makedirs("app", exist_ok=True)
    os.makedirs("backend", exist_ok=True)

    with open("app/summaries.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    with open("backend/fetch_log.json", "w", encoding="utf-8") as f:
        json.dump({
            "timestamp": datetime.utcnow().isoformat(),
            "article_count": len(results),
            "logs": logs
        }, f, indent=2)

    print("💾 Results saved to app/summaries.json and backend/fetch_log.json")

if __name__ == "__main__":
    main()