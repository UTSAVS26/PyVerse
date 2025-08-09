from flask import Flask, request, render_template
from sentiment_analysis import sentiment_analyzer
import json

app = Flask(__name__)

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    text = request.form.get('textToAnalyze')
    
    if not text or not text.strip():
        return {
            "error": "Text input is required",
            "label": "ERROR",
            "score": 0.0
        }
    
    try:
        # Call the sentiment analyzer function
        result = sentiment_analyzer(text)
        
        # Parse the JSON response
        sentiment_data = json.loads(result)
        
        # Extract sentiment information
        # The response format may vary, adjust based on actual response structure
        if 'documentSentiment' in sentiment_data:
            sentiment = sentiment_data['documentSentiment']
            label = sentiment.get('label', 'NEUTRAL')
            score = sentiment.get('score', 0.0)
        else:
            # Fallback parsing - adjust based on actual response structure
            label = "NEUTRAL"
            score = 0.0
        
        return {
            "label": label,
            "score": abs(score)  # Ensure positive score for frontend
        }
    
    except Exception as e:
        return {
            "error": f"Analysis failed: {str(e)}",
            "label": "ERROR",
            "score": 0.0
        }

if __name__ == "__main__":
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
