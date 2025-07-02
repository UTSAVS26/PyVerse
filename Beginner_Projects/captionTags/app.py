from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=gemini_api_key)
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    caption = ""
    hashtags = []
    if request.method == "POST":
        description = request.form.get("description")
        mood = request.form.get("mood")
        prompt = f"Write a short, catchy, and {mood.lower()} Instagram caption based on the following post description: \"{description}\". Make it engaging and suitable for social media."
        try:
            model = genai.GenerativeModel('gemini-1.5-flash')
            response = model.generate_content(prompt, generation_config={"temperature": 0.8})
            match = re.search(r"\*\*Option 1.*?>\s*(.+?)(?=\n\*\*Option|\Z)", response.text.strip(), re.DOTALL)
            if match:
                caption = match.group(1).strip()
            else:
                caption = response.text.strip().split('\n')[0]
            # Remove hashtags from the caption (any word starting with #)
            caption = ' '.join(word for word in caption.split() if not word.startswith('#'))

            # Hashtag generation
            hashtag_prompt = f"Generate 5 to 6 relevant, popular Instagram hashtags (without # symbol) for a post with this description: '{description}' and mood: '{mood}'. Separate each hashtag with a comma. Do not include any caption, only the hashtags."
            hashtag_response = model.generate_content(hashtag_prompt, generation_config={"temperature": 0.8})
            hashtags_raw = hashtag_response.text.strip()
            # Split by comma, clean up, and add #
            hashtags = [f"#{tag.strip().replace(' ', '')}" for tag in hashtags_raw.split(',') if tag.strip()]
        except Exception as e:
            caption = f"Error: {str(e)}"
            hashtags = []
    return render_template("index.html", caption=caption, hashtags=hashtags)

if __name__ == "__main__":
    app.run(debug=True)