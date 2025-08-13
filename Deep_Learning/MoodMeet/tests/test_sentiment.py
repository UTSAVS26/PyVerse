"""
Comprehensive Test Suite for MoodMeet

Tests all components including sentiment analysis, clustering, keyword extraction, and visualization.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os
from typing import Dict, List

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules to test
from data.uploader import TextInputHandler, TranscriptProcessor
from analysis.sentiment_analyzer import SentimentAnalyzer, SentimentTrendAnalyzer
from analysis.mood_clustering import MoodClusterer, TopicAnalyzer
from analysis.keyword_extractor import KeywordExtractor, PhraseExtractor
from visualization.mood_timeline import MoodTimelineVisualizer
from visualization.heatmap_generator import HeatmapGenerator


class TestDataUploader:
    """Test data uploader functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.text_handler = TextInputHandler()
        self.processor = TranscriptProcessor()
        
        self.sample_text = """
Alice: We're falling behind schedule.
Bob: Let's regroup and finish the draft today.
Carol: Honestly, I'm feeling a bit burned out.
David: I think we can make it work if we focus.
Alice: That sounds like a good plan.
        """
    
    def test_parse_transcript(self):
        """Test transcript parsing."""
        entries = self.processor.parse_transcript(self.sample_text)
        
        assert len(entries) == 5
        assert entries[0].speaker == "Alice"
        assert entries[0].text == "We're falling behind schedule."
        assert entries[1].speaker == "Bob"
        assert entries[1].text == "Let's regroup and finish the draft today."
    
    def test_validate_transcript(self):
        """Test transcript validation."""
        entries = self.processor.parse_transcript(self.sample_text)
        is_valid, errors = self.processor.validate_transcript(entries)
        
        assert is_valid == True
        assert len(errors) == 0
    
    def test_to_dataframe(self):
        """Test DataFrame conversion."""
        entries = self.processor.parse_transcript(self.sample_text)
        df = self.processor.to_dataframe(entries)
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 5
        assert 'speaker' in df.columns
        assert 'text' in df.columns
        assert 'text_length' in df.columns
    
    def test_get_speaker_stats(self):
        """Test speaker statistics."""
        entries = self.processor.parse_transcript(self.sample_text)
        stats = self.processor.get_speaker_stats(entries)
        
        assert 'Alice' in stats
        assert 'Bob' in stats
        assert 'Carol' in stats
        assert 'David' in stats
        assert stats['Alice']['message_count'] == 2
    
    def test_process_text_input(self):
        """Test complete text input processing."""
        df, speaker_stats, is_valid, errors = self.text_handler.process_text_input(self.sample_text)
        
        assert is_valid == True
        assert len(errors) == 0
        assert isinstance(df, pd.DataFrame)
        assert isinstance(speaker_stats, dict)
        assert len(df) == 5


class TestSentimentAnalyzer:
    """Test sentiment analysis functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = SentimentAnalyzer(model_type="vader")
        self.trend_analyzer = SentimentTrendAnalyzer()
        
        self.test_texts = [
            "I love this project!",
            "This is terrible.",
            "The meeting was okay.",
            "I'm feeling a bit burned out.",
            "Let's make this work together!"
        ]
    
    def test_analyze_text(self):
        """Test single text sentiment analysis."""
        result = self.analyzer.analyze_text("I love this project!")
        
        assert hasattr(result, 'text')
        assert hasattr(result, 'polarity')
        assert hasattr(result, 'sentiment_label')
        assert result.text == "I love this project!"
        assert result.polarity > 0  # Should be positive
    
    def test_analyze_dataframe(self):
        """Test DataFrame sentiment analysis."""
        df = pd.DataFrame({'text': self.test_texts})
        results_df = self.analyzer.analyze_dataframe(df)
        
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) == 5
        assert 'polarity' in results_df.columns
        assert 'sentiment_label' in results_df.columns
    
    def test_get_sentiment_summary(self):
        """Test sentiment summary generation."""
        df = pd.DataFrame({'text': self.test_texts})
        results_df = self.analyzer.analyze_dataframe(df)
        summary = self.analyzer.get_sentiment_summary(results_df)
        
        assert isinstance(summary, dict)
        assert 'total_messages' in summary
        assert 'avg_polarity' in summary
        assert summary['total_messages'] == 5
    
    def test_analyze_trend(self):
        """Test trend analysis."""
        df = pd.DataFrame({
            'polarity': [-0.2, 0.3, -0.4, 0.1, 0.5]
        })
        trend_df = self.trend_analyzer.analyze_trend(df)
        
        assert isinstance(trend_df, pd.DataFrame)
        assert 'moving_avg' in trend_df.columns
        assert 'trend' in trend_df.columns
    
    def test_get_trend_summary(self):
        """Test trend summary generation."""
        df = pd.DataFrame({
            'polarity': [-0.2, 0.3, -0.4, 0.1, 0.5]
        })
        trend_df = self.trend_analyzer.analyze_trend(df)
        summary = self.trend_analyzer.get_trend_summary(trend_df)
        
        assert isinstance(summary, dict)
        assert 'trend_direction' in summary
        assert 'trend_magnitude' in summary


class TestMoodClustering:
    """Test mood clustering functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.clusterer = MoodClusterer(method="kmeans", n_clusters=3)
        self.topic_analyzer = TopicAnalyzer()
        
        self.test_texts = [
            "We're falling behind schedule.",
            "Let's regroup and finish the draft today.",
            "I'm feeling a bit burned out.",
            "I think we can make it work if we focus.",
            "That sounds like a good plan.",
            "The deadline is approaching fast.",
            "We need to prioritize our tasks.",
            "I'm confident we can deliver on time.",
            "Let's break this down into smaller tasks.",
            "The team is working well together."
        ]
        
        self.test_sentiments = [-0.2, 0.3, -0.4, 0.1, 0.5, -0.3, 0.2, 0.4, 0.1, 0.6]
    
    def test_fit_clustering(self):
        """Test clustering functionality."""
        cluster_results = self.clusterer.fit(self.test_texts, self.test_sentiments)
        
        assert isinstance(cluster_results, list)
        assert len(cluster_results) > 0
        
        for result in cluster_results:
            assert hasattr(result, 'cluster_id')
            assert hasattr(result, 'texts')
            assert hasattr(result, 'keywords')
            assert hasattr(result, 'sentiment_avg')
            assert hasattr(result, 'size')
    
    def test_get_clustering_summary(self):
        """Test clustering summary generation."""
        cluster_results = self.clusterer.fit(self.test_texts, self.test_sentiments)
        summary = self.clusterer.get_clustering_summary(cluster_results)
        
        assert isinstance(summary, dict)
        assert 'total_clusters' in summary
        assert 'total_texts' in summary
        assert 'clusters' in summary
    
    def test_extract_topics(self):
        """Test topic extraction."""
        topics = self.topic_analyzer.extract_topics(self.test_texts, n_topics=3)
        
        assert isinstance(topics, list)
        assert len(topics) > 0
        
        for topic in topics:
            assert 'topic_id' in topic
            assert 'keywords' in topic
            assert 'weight' in topic
    
    def test_get_topic_summary(self):
        """Test topic summary generation."""
        topics = self.topic_analyzer.extract_topics(self.test_texts, n_topics=3)
        summary = self.topic_analyzer.get_topic_summary(topics)
        
        assert isinstance(summary, dict)
        assert 'total_topics' in summary
        assert 'topics' in summary


class TestKeywordExtractor:
    """Test keyword extraction functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = KeywordExtractor(method="tfidf")
        self.phrase_extractor = PhraseExtractor()
        
        self.test_texts = [
            "We're falling behind schedule.",
            "Let's regroup and finish the draft today.",
            "I'm feeling a bit burned out.",
            "I think we can make it work if we focus.",
            "That sounds like a good plan.",
            "The deadline is approaching fast.",
            "We need to prioritize our tasks.",
            "I'm confident we can deliver on time."
        ]
    
    def test_extract_keywords(self):
        """Test keyword extraction."""
        keywords = self.extractor.extract_keywords(self.test_texts, max_keywords=10)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        
        for keyword in keywords:
            assert hasattr(keyword, 'keyword')
            assert hasattr(keyword, 'score')
            assert hasattr(keyword, 'method')
    
    def test_get_keyword_summary(self):
        """Test keyword summary generation."""
        keywords = self.extractor.extract_keywords(self.test_texts, max_keywords=10)
        summary = self.extractor.get_keyword_summary(keywords)
        
        assert isinstance(summary, dict)
        assert 'total_keywords' in summary
        assert 'methods_used' in summary
        assert 'top_keywords' in summary
    
    def test_extract_phrases(self):
        """Test phrase extraction."""
        phrases = self.phrase_extractor.extract_phrases(self.test_texts)
        
        assert isinstance(phrases, list)
        
        for phrase in phrases:
            assert 'phrase' in phrase
            assert 'frequency' in phrase
            assert 'length' in phrase


class TestVisualization:
    """Test visualization functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.timeline_viz = MoodTimelineVisualizer()
        self.heatmap_gen = HeatmapGenerator()
        
        self.test_data = {
            'text': [
                "We're falling behind schedule.",
                "Let's regroup and finish the draft today.",
                "I'm feeling a bit burned out.",
                "I think we can make it work if we focus.",
                "That sounds like a good plan."
            ],
            'speaker': ['Alice', 'Bob', 'Carol', 'David', 'Alice'],
            'polarity': [-0.2, 0.3, -0.4, 0.1, 0.5],
            'sentiment_label': ['negative', 'positive', 'negative', 'positive', 'positive']
        }
        
        self.test_df = pd.DataFrame(self.test_data)
    
    def test_create_sentiment_timeline(self):
        """Test timeline chart creation."""
        fig = self.timeline_viz.create_sentiment_timeline(self.test_df, speaker_column='speaker')
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
    
    def test_create_speaker_sentiment_chart(self):
        """Test speaker comparison chart creation."""
        fig = self.timeline_viz.create_speaker_sentiment_chart(self.test_df, speaker_column='speaker')
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
    
    def test_create_sentiment_distribution(self):
        """Test sentiment distribution chart creation."""
        fig = self.timeline_viz.create_sentiment_distribution(self.test_df)
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
    
    def test_create_moving_average_chart(self):
        """Test moving average chart creation."""
        fig = self.timeline_viz.create_moving_average_chart(self.test_df)
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
    
    def test_create_sentiment_heatmap(self):
        """Test sentiment heatmap creation."""
        fig = self.heatmap_gen.create_sentiment_heatmap(self.test_df, speaker_column='speaker')
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
    
    def test_create_emotion_heatmap(self):
        """Test emotion heatmap creation."""
        fig = self.heatmap_gen.create_emotion_heatmap(self.test_df, speaker_column='speaker')
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')
    
    def test_create_sentiment_intensity_heatmap(self):
        """Test sentiment intensity heatmap creation."""
        fig = self.heatmap_gen.create_sentiment_intensity_heatmap(self.test_df, speaker_column='speaker')
        
        assert fig is not None
        assert hasattr(fig, 'data')
        assert hasattr(fig, 'layout')


class TestIntegration:
    """Test integration between components."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.text_handler = TextInputHandler()
        self.sentiment_analyzer = SentimentAnalyzer(model_type="vader")
        self.clusterer = MoodClusterer(method="kmeans", n_clusters=3)
        self.keyword_extractor = KeywordExtractor(method="tfidf")
        
        self.sample_text = """
Alice: We're falling behind schedule.
Bob: Let's regroup and finish the draft today.
Carol: Honestly, I'm feeling a bit burned out.
David: I think we can make it work if we focus.
Alice: That sounds like a good plan.
        """
    
    def test_end_to_end_analysis(self):
        """Test complete end-to-end analysis pipeline."""
        # Process text input
        df, speaker_stats, is_valid, errors = self.text_handler.process_text_input(self.sample_text)
        
        assert is_valid == True
        assert len(errors) == 0
        
        # Perform sentiment analysis
        sentiment_results = self.sentiment_analyzer.analyze_dataframe(df)
        sentiment_summary = self.sentiment_analyzer.get_sentiment_summary(sentiment_results)
        
        assert isinstance(sentiment_results, pd.DataFrame)
        assert isinstance(sentiment_summary, dict)
        
        # Perform clustering
        texts = df['text'].tolist()
        sentiments = sentiment_results['polarity'].tolist()
        cluster_results = self.clusterer.fit(texts, sentiments)
        
        assert isinstance(cluster_results, list)
        assert len(cluster_results) > 0
        
        # Extract keywords
        keywords = self.keyword_extractor.extract_keywords(texts, max_keywords=10)
        
        assert isinstance(keywords, list)
        assert len(keywords) > 0
    
    def test_data_consistency(self):
        """Test data consistency across components."""
        df, speaker_stats, is_valid, errors = self.text_handler.process_text_input(self.sample_text)
        sentiment_results = self.sentiment_analyzer.analyze_dataframe(df)
        
        # Check that all DataFrames have the same number of rows
        assert len(df) == len(sentiment_results)
        
        # Check that speaker statistics match the data
        assert len(speaker_stats) == df['speaker'].nunique()
        
        # Check that sentiment results have expected columns
        expected_columns = ['polarity', 'sentiment_label']
        for col in expected_columns:
            assert col in sentiment_results.columns


class TestErrorHandling:
    """Test error handling and edge cases."""
    
    def test_empty_text_input(self):
        """Test handling of empty text input."""
        text_handler = TextInputHandler()
        df, speaker_stats, is_valid, errors = text_handler.process_text_input("")
        
        assert is_valid == False
        assert len(errors) > 0
    
    def test_invalid_transcript_format(self):
        """Test handling of invalid transcript format."""
        text_handler = TextInputHandler()
        df, speaker_stats, is_valid, errors = text_handler.process_text_input("Just some random text without speakers")
        
        # Should still be valid but with anonymous speaker
        assert is_valid == True
    
    def test_sentiment_analyzer_with_empty_df(self):
        """Test sentiment analyzer with empty DataFrame."""
        analyzer = SentimentAnalyzer(model_type="vader")
        df = pd.DataFrame()
        results_df = analyzer.analyze_dataframe(df)
        summary = analyzer.get_sentiment_summary(results_df)
        
        assert isinstance(results_df, pd.DataFrame)
        assert isinstance(summary, dict)
    
    def test_clustering_with_single_text(self):
        """Test clustering with single text."""
        clusterer = MoodClusterer(method="kmeans", n_clusters=3)
        results = clusterer.fit(["Single text"], [0.0])
        
        # Should handle gracefully
        assert isinstance(results, list)


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 