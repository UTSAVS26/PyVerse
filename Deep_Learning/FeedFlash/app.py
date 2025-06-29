import json
import gradio as gr

with open("app/summaries.json", "r", encoding="utf-8") as f:
    articles = json.load(f)

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
            index = int(title.split(". ")[0]) - 1
            return create_card(index)

        selector.change(fn=update_ui,
                        inputs=selector,
                        outputs=[title_box, link_box, summary_box])

    return demo

if __name__ == "__main__":
    app = build_ui()
    app.launch()