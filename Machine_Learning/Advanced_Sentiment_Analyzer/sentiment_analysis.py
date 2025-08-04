import requests  # Import the requests library to handle HTTP requests
import json
import re

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
    
    # Count positive and negative words
    positive_count = sum(1 for word in positive_words if word in text_lower)
    negative_count = sum(1 for word in negative_words if word in text_lower)
    
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
    url = 'https://sn-watson-sentiment-bert.labs.skills.network/v1/watson.runtime.nlp.v1/NlpService/SentimentPredict'  # URL of the sentiment analysis service
    myobj = { "raw_document": { "text": text_to_analyse } }  # Create a dictionary with the text to be analyzed
    header = {"grpc-metadata-mm-model-id": "sentiment_aggregated-bert-workflow_lang_multi_stock"}  # Set the headers required for the API request
    
    try:
        response = requests.post(url, json = myobj, headers=header, timeout=10)  # Send a POST request to the API with the text and headers
        return response.text  # Return the response text from the API
    except (requests.exceptions.RequestException, requests.exceptions.Timeout):
        # If the service is unavailable, use fallback sentiment analysis
        fallback_result = simple_sentiment_fallback(text_to_analyse)
        return json.dumps(fallback_result)  # Return as JSON string
