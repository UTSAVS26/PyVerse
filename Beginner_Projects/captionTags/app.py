from flask import Flask, render_template, request
import os
from dotenv import load_dotenv
import google.generativeai as genai
import re

load_dotenv()
gemini_api_key = os.getenv("GOOGLE_API_KEY")

if not gemini_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is required")
genai.configure(api_key=gemini_api_key)
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def index():
    caption = ""
    hashtags = []
    error = None  # <-- Add error message

    if request.method == "POST":
        description = request.form.get("description", "").strip()
        mood = request.form.get("mood", "").strip()

        # Validation
        if not description:
            error = "Please enter a post description."
        elif not mood:
            error = "Please select a mood."
        else:
            prompt = f"Write a short, catchy, and {mood.lower()} Instagram caption based on the following post description: \"{description}\". Make it engaging and suitable for social media."
            try:
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(prompt, generation_config={"temperature": 0.8})
                match = re.search(r"\*\*Option 1.*?>\s*(.+?)(?=\n\*\*Option|\Z)", response.text.strip(), re.DOTALL)
                if match:
                    caption = match.group(1).strip()
                else:
                    caption = response.text.strip().split('\n')[0]
                caption = ' '.join(word for word in caption.split() if not word.startswith('#'))

                # Hashtag generation
                hashtag_prompt = f"Generate 5 to 6 relevant, popular Instagram hashtags (without # symbol) for a post with this description: '{description}' and mood: '{mood}'. Separate each hashtag with a comma. Do not include any caption, only the hashtags."
                hashtag_response = model.generate_content(hashtag_prompt, generation_config={"temperature": 0.8})
                hashtags_raw = hashtag_response.text.strip()
                hashtags = []
                for tag in hashtags_raw.split(','):
                   clean_tag = re.sub(r'[^a-zA-Z0-9_]', '', tag.strip())
                   if clean_tag:
                      hashtags.append(f"#{clean_tag}")
                      
            except Exception as e:
                error = f"Error: {str(e)}"
                caption = ""
                hashtags = []

    return render_template("index.html", caption=caption, hashtags=hashtags, error=error)


if __name__ == "__main__":
    debug_mode = os.getenv("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug_mode)
   