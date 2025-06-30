import json
import gradio as gr
import os  

def load_articles(file_path="app/summaries.json"):  
    """Load articles from JSON file with error handling."""  
    try:  
        if not os.path.exists(file_path):  
            return []  
        with open(file_path, "r", encoding="utf-8") as f:  
            articles = json.load(f)  
            if not isinstance(articles, list):  
                raise ValueError("Articles data must be a list")  
            return articles  
    except (json.JSONDecodeError, IOError) as e:  
        print(f"Error loading articles: {e}")  
        return []  

articles = load_articles()

def create_card(index):
    article = articles[index]
    title = article.get("title", "Untitled")
    author = ", ".join(article.get("authors", [])) or "Unknown Author"
    source = article.get("source", "Unknown Source")
    summary = article.get("summary", "Summary not available.")
    url = article.get("url", "")

    headline_display = f"**{title}**\n_{author} - {source}_"

    return headline_display, url, summary

def build_ui():  
    if not articles:  
        with gr.Blocks(title="FeedFlash - Summarized News") as demo:  
            gr.Markdown(  
                "# 🗞️ FeedFlash\n"  
                "⚠️ No news articles available. Please check the data source."  
            )  
        return demo  

    with gr.Blocks(title="FeedFlash - Summarized News") as demo:  
        gr.Markdown("# 🗞️ FeedFlash\nSummarized News Updates Every Few Hours")  

        with gr.Row():  
            with gr.Column(scale=1):  
                headlines = [f"{i+1}. {a['title']}" for i, a in enumerate(articles)]  
                selector = gr.Radio(headlines, label="📰 Top Headlines", value=headlines[0])  
            
            with gr.Column(scale=2):  
                title_box = gr.Markdown()  
                link_box = gr.Textbox(label="🔗 Source URL", interactive=False)  
                summary_box = gr.Textbox(label="📝 Summary", lines=8, interactive=False)  

        def update_ui(title):  
            if not title or not isinstance(title, str):  
                return "Error: Invalid title", "", "No summary available"  
            
            try:  
                parts = title.split(". ", 1)  
                if len(parts) < 2:  
                    return "Error: Invalid title format", "", "No summary available"  
                
                index = int(parts[0]) - 1  
                if index < 0 or index >= len(articles):  
                    return "Error: Article not found", "", "No summary available"  
                
                return create_card(index)  
            except (ValueError, IndexError) as e:  
                return f"Error: {str(e)}", "", "No summary available" 

        selector.change(  
            fn=update_ui,  
            inputs=selector,  
            outputs=[title_box, link_box, summary_box],  
        )  

    return demo  

if __name__ == "__main__":  
    app = build_ui()  
    app.launch()