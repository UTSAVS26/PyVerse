#!/usr/bin/env python3
"""
Test script for the sentiment analysis application
"""

import unittest
import json
from sentiment_analysis import sentiment_analyzer

class TestSentimentAnalysis(unittest.TestCase):
    """Test cases for sentiment analyzer"""
    
    def setUp(self):
        """Set up test cases"""
        self.test_cases = [
            ("I love this product! It's amazing!", "POSITIVE"),
            ("This is terrible and awful!", "NEGATIVE"),
            ("The weather is okay today.", "NEUTRAL"),
            ("I hate this so much!", "NEGATIVE"),
            ("Fantastic work! Great job!", "POSITIVE"),
            ("This is just normal.", "NEUTRAL")
        ]
    
    def test_sentiment_analyzer_response_format(self):
        """Test that sentiment analyzer returns valid JSON"""
        for text, _ in self.test_cases:
            with self.subTest(text=text):
                result = sentiment_analyzer(text)
                try:
                    parsed = json.loads(result)
                    self.assertIsInstance(parsed, dict)
                except json.JSONDecodeError:
                    self.fail(f"Invalid JSON returned for text: {text}")
    
    def test_sentiment_analysis_results(self):
        """Test sentiment analysis accuracy"""
        for text, expected in self.test_cases:
            with self.subTest(text=text, expected=expected):
                result = sentiment_analyzer(text)
                parsed = json.loads(result)
                
                if 'documentSentiment' in parsed:
                    sentiment = parsed['documentSentiment']
                    label = sentiment.get('label')
                    score = sentiment.get('score')
                    
                    self.assertIsNotNone(label, f"No label found for: {text}")
                    self.assertIsNotNone(score, f"No score found for: {text}")
                    self.assertEqual(label, expected, f"Expected {expected}, got {label} for: {text}")
                else:
                    self.fail(f"documentSentiment not found in result for: {text}")

if __name__ == "__main__":
    unittest.main()
