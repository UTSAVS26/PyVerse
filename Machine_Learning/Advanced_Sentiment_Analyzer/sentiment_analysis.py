import requests  # Import the requests library to handle HTTP requests
import json
import re
import os
from collections import Counter

def simple_sentiment_fallback(text):
    """Simple rule-based sentiment analysis fallback"""
    # Convert to lowercase for analysis
    text_lower = text.lower()
    
    # Define positive and negative words
    positive_words = ['good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 'awesome', 
                     'love', 'like', 'happy', 'pleased', 'satisfied', 'perfect', 'brilliant',
                     'outstanding', 'superb', 'magnificent', 'remarkable', 'incredible', 'best']
    
    negative_words = ['bad', 'terrible', 'awful', 'horrible', 'hate', 'dislike', 'angry', 'disappointed',
                     'frustrated', 'annoyed', 'disgusted', 'furious', 'irritated', 'upset', 'worst',
                     'pathetic', 'useless', 'worthless', 'ridiculous', 'stupid', 'fail', 'failed']
    
    # Tokenize text and count whole-word occurrences
    tokens = re.findall(r"\b\w+\b", text_lower)
    freq = Counter(tokens)
    positive_count = sum(freq[w] for w in positive_words)
    negative_count = sum(freq[w] for w in negative_words)
    
    # Determine sentiment
    if positive_count > negative_count:
        label = "POSITIVE"
        score = min(0.9, 0.5 + (positive_count - negative_count) * 0.1)
    elif negative_count > positive_count:
        label = "NEGATIVE" 
        score = min(0.9, 0.5 + (negative_count - positive_count) * 0.1)
    else:
        label = "NEUTRAL"
        score = 0.5
    
    return {
        "documentSentiment": {
            "label": label,
            "score": score
        }
    }

def sentiment_analyzer(text_to_analyse):  # Define a function named sentiment_analyzer that takes a string input (text_to_analyse)
    # Input validation
    if not isinstance(text_to_analyse, str) or not text_to_analyse.strip():
        return json.dumps({
            "documentSentiment": {
                "label": "NEUTRAL",
                "score": 0.5
            }
        })
    
    # Make API configuration externally configurable
    url = os.getenv('SENTIMENT_API_URL', 'https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict')  # URL of the sentiment analysis service
    model_id = os.getenv('SENTIMENT_MODEL_ID', 'sentiment_aggregated-bert-workflow_lang_multi_stock')
    myobj = { "raw_document": { "text": text_to_analyse } }  # Create a dictionary with the text to be analyzed
    header = {"grpc-metadata-mm-model-id": model_id}  # Set the headers required for the API request
    
    try:
        response = requests.post(url, json=myobj, headers=header, timeout=5)  # Send a POST request to the API with the text and headers
        if response.status_code == 200:
            # Validate JSON and expected keys
            try:
                data = response.json()
            except ValueError:
                data = None
            if isinstance(data, dict) and isinstance(data.get("documentSentiment"), dict):
                doc = data["documentSentiment"]
                label = doc.get("label")
                score = doc.get("score")
                # Normalize label variants from providers
                norm_map = {
                    "positive": "POSITIVE",
                    "sent_positive": "POSITIVE",
                    "pos": "POSITIVE",
                    "negative": "NEGATIVE",
                    "sent_negative": "NEGATIVE",
                    "neg": "NEGATIVE",
                    "neutral": "NEUTRAL",
                    "sent_neutral": "NEUTRAL",
                }
                normalized = norm_map.get(str(label).strip().lower()) if isinstance(label, str) else None
                if normalized and isinstance(score, (int, float)) and 0.0 <= float(score) <= 1.0:
                    doc["label"] = normalized
                    return json.dumps(data)
            # Malformed or unexpected success body — use fallback
            return json.dumps(simple_sentiment_fallback(text_to_analyse))
        else:
            # Fall back on non-200 status codes
            fallback_result = simple_sentiment_fallback(text_to_analyse)
            return json.dumps(fallback_result)
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        # If the service is unavailable, use fallback sentiment analysis
        fallback_result = simple_sentiment_fallback(text_to_analyse)
        return json.dumps(fallback_result)
