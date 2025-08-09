#!/usr/bin/env python3
"""
Test script for the sentiment analysis application
"""

import unittest
import json
import requests
from sentiment_analysis import sentiment_analyzer
from unittest.mock import patch, Mock

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
    
    @patch("sentiment_analysis.requests.post")
    def test_sentiment_analyzer_response_format(self, mock_post):
        """Test that sentiment analyzer returns valid JSON"""
        mock_post.return_value = Mock(
            status_code=200,
            json=Mock(return_value={"documentSentiment": {"label": "neutral", "score": 0.5}})
        )
        for text, _ in self.test_cases:
            with self.subTest(text=text):
                result = sentiment_analyzer(text)
                try:
                    parsed = json.loads(result)
                    self.assertIsInstance(parsed, dict)
                except json.JSONDecodeError:
                    self.fail(f"Invalid JSON returned for text: {text}")
    
    @patch("sentiment_analysis.requests.post")
    def test_sentiment_analysis_results(self, mock_post):
        """Test sentiment analysis accuracy"""
        def _mock_json(text):
            t = text.lower()
            if any(w in t for w in ["love", "amazing", "fantastic", "great"]):
                return {"documentSentiment": {"label": "positive", "score": 0.9}}
            if any(w in t for w in ["terrible", "awful", "hate"]):
                return {"documentSentiment": {"label": "negative", "score": 0.8}}
            return {"documentSentiment": {"label": "neutral", "score": 0.5}}
        def side_effect(url=None, json=None, headers=None, timeout=None):
            return Mock(status_code=200, json=Mock(return_value=_mock_json(json["raw_document"]["text"])))
        mock_post.side_effect = side_effect
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
                    self.assertIsInstance(score, (int, float), f"Score must be numeric for: {text}")
                    self.assertGreaterEqual(score, 0.0, f"Score must be >= 0 for: {text}")
                    self.assertLessEqual(score, 1.0, f"Score must be <= 1 for: {text}")
                    self.assertEqual(label, expected, f"Expected {expected}, got {label} for: {text}")
                else:
                    self.fail(f"documentSentiment not found in result for: {text}")

    @patch("sentiment_analysis.requests.post")
    def test_fallback_on_non_200(self, mock_post):
        """Test fallback when API returns non-200 status"""
        mock_post.return_value = Mock(status_code=500, text='{"error":"upstream"}')
        result = sentiment_analyzer("Any text")
        parsed = json.loads(result)
        self.assertIn("documentSentiment", parsed)
        self.assertIn(parsed["documentSentiment"]["label"], {"POSITIVE", "NEGATIVE", "NEUTRAL"})

    @patch("sentiment_analysis.requests.post", side_effect=requests.exceptions.RequestException("timeout"))
    def test_fallback_on_exception(self, _):
        """Test fallback when API request raises exception"""
        result = sentiment_analyzer("Any text")
        parsed = json.loads(result)
        self.assertIn("documentSentiment", parsed)
        self.assertIn(parsed["documentSentiment"]["label"], {"POSITIVE", "NEGATIVE", "NEUTRAL"})

    @patch("sentiment_analysis.requests.post")
    def test_fallback_on_malformed_json(self, mock_post):
        """Test fallback when API returns malformed JSON"""
        mock_post.return_value = Mock(status_code=200, json=Mock(side_effect=ValueError("Invalid JSON")))
        result = sentiment_analyzer("Any text")
        parsed = json.loads(result)
        self.assertIn("documentSentiment", parsed)
        self.assertIn(parsed["documentSentiment"]["label"], {"POSITIVE", "NEGATIVE", "NEUTRAL"})

    @patch("sentiment_analysis.requests.post")
    def test_fallback_on_missing_keys(self, mock_post):
        """Test fallback when API response is missing expected keys"""
        mock_post.return_value = Mock(status_code=200, json=Mock(return_value={"unexpected": "format"}))
        result = sentiment_analyzer("Any text")
        parsed = json.loads(result)
        self.assertIn("documentSentiment", parsed)
        self.assertIn(parsed["documentSentiment"]["label"], {"POSITIVE", "NEGATIVE", "NEUTRAL"})

    def test_empty_input_validation(self):
        """Test input validation for empty and whitespace-only inputs"""
        test_inputs = ["", "   ", "\t", "\n", None]
        
        for test_input in test_inputs:
            with self.subTest(input=test_input):
                result = sentiment_analyzer(test_input)
                parsed = json.loads(result)
                
                self.assertIn("documentSentiment", parsed)
                self.assertEqual(parsed["documentSentiment"]["label"], "NEUTRAL")
                self.assertEqual(parsed["documentSentiment"]["score"], 0.5)

    def test_non_string_inputs(self):
        """Test handling of non-string inputs"""
        test_inputs = [123, [], {}, True, False]
        
        for test_input in test_inputs:
            with self.subTest(input=test_input):
                result = sentiment_analyzer(test_input)
                parsed = json.loads(result)
                self.assertIn("documentSentiment", parsed)
                self.assertEqual(parsed["documentSentiment"]["label"], "NEUTRAL")
                self.assertEqual(parsed["documentSentiment"]["score"], 0.5)

if __name__ == "__main__":
    unittest.main()
