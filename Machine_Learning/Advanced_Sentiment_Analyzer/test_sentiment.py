#!/usr/bin/env python3
"""
Test script for the sentiment analysis application
"""

from sentiment_analysis import sentiment_analyzer
import json

def test_sentiment_analysis():
    """Test the sentiment analyzer with various inputs"""
    
    test_cases = [
        ("I love this product! It's amazing!", "POSITIVE"),
        ("This is terrible and awful!", "NEGATIVE"),
        ("The weather is okay today.", "NEUTRAL"),
        ("I hate this so much!", "NEGATIVE"),
        ("Fantastic work! Great job!", "POSITIVE"),
        ("This is just normal.", "NEUTRAL")
    ]
    
    print("Testing Sentiment Analysis:")
    print("-" * 50)
    
    for text, expected in test_cases:
        result = sentiment_analyzer(text)
        parsed = json.loads(result)
        
        if 'documentSentiment' in parsed:
            sentiment = parsed['documentSentiment']
            label = sentiment['label']
            score = sentiment['score']
            
            status = "✓" if label == expected else "✗"
            print(f"{status} Text: {text}")
            print(f"  Expected: {expected}, Got: {label}, Score: {score:.2f}")
            print()
        else:
            print(f"✗ Error parsing result for: {text}")
            print(f"  Result: {result}")
            print()

if __name__ == "__main__":
    test_sentiment_analysis()
