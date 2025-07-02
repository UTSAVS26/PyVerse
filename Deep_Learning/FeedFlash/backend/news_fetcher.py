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

def fetch_news():  
    """Fetch news articles from the last 24 hours."""  

    try:  
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
    """Main function to fetch, process, and save news summaries."""  
    
    print("🚀 Starting news fetching and summarization...")  

    articles = fetch_news()  
    if not articles:  
        print("❌ No articles fetched. Exiting.")  
        return  

    results = []  
    logs = []  

    print(f"📰 Processing {len(articles)} articles...")  

    for art in articles:  
        url = art.get("url")  
        if not url:  
            continue  

        full_text = extract_full_article(url)  

        if not full_text or len(full_text.split()) < 100:  
            continue  

        try:  
            summary = summarize_text(full_text)  
        except Exception as e:  
            print(f"⚠️ Failed to summarize article {url}: {e}")  
            continue  

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

    print(f"✅ Successfully processed {len(results)} articles")  

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