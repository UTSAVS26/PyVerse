"""
Sentiment Analyzer Module for MoodMeet

Provides multiple sentiment analysis approaches including VADER, TextBlob, and transformer-based models.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
import logging

# Sentiment analysis libraries
VADER_AVAILABLE = False
TEXTBLOB_AVAILABLE = False
TRANSFORMER_AVAILABLE = False

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    VADER_AVAILABLE = True
except ImportError:
    logging.warning("VADER sentiment analyzer not available")

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    logging.warning("TextBlob sentiment analyzer not available")

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    import torch
    TRANSFORMER_AVAILABLE = True
except ImportError:
    logging.warning("Transformers library not available")


@dataclass
class SentimentResult:
    """Represents sentiment analysis results for a text."""
    text: str
    polarity: float  # -1 to 1
    subjectivity: Optional[float] = None  # 0 to 1 (TextBlob only)
    compound_score: Optional[float] = None  # VADER compound score
    sentiment_label: str = "neutral"  # positive, negative, neutral
    confidence: Optional[float] = None  # Transformer model confidence


class SentimentAnalyzer:
    """Multi-model sentiment analyzer."""
    
    def __init__(self, model_type: str = "vader"):
        """
        Initialize sentiment analyzer.
        
        Args:
            model_type: Type of model to use ('vader', 'textblob', 'transformer')
        """
        self.model_type = model_type
        self.analyzers = {}
        
        # Initialize VADER
        if VADER_AVAILABLE and model_type in ["vader", "ensemble"]:
            self.analyzers['vader'] = SentimentIntensityAnalyzer()
        
        # Initialize TextBlob (no initialization needed)
        if TEXTBLOB_AVAILABLE and model_type in ["textblob", "ensemble"]:
            self.analyzers['textblob'] = None  # TextBlob doesn't need initialization
        
        # Initialize Transformer model
        if TRANSFORMER_AVAILABLE and model_type in ["transformer", "ensemble"]:
            try:
                self.analyzers['transformer'] = pipeline(
                    "sentiment-analysis",
                    model="cardiffnlp/twitter-roberta-base-sentiment",
                    return_all_scores=True
                )
            except Exception as e:
                logging.warning(f"Failed to load transformer model: {e}")
                # Don't modify the global variable, just skip this analyzer
                pass
        
        if not self.analyzers:
            raise ValueError(f"No sentiment analyzers available for model type: {model_type}")
    
    def analyze_text(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            SentimentResult object
        """
        if self.model_type == "ensemble":
            return self._ensemble_analyze(text)
        elif self.model_type == "vader":
            return self._vader_analyze(text)
        elif self.model_type == "textblob":
            return self._textblob_analyze(text)
        elif self.model_type == "transformer":
            return self._transformer_analyze(text)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _vader_analyze(self, text: str) -> SentimentResult:
        """Analyze using VADER."""
        if 'vader' not in self.analyzers:
            raise ValueError("VADER analyzer not available")
        
        scores = self.analyzers['vader'].polarity_scores(text)
        
        # Determine sentiment label
        if scores['compound'] >= 0.05:
            sentiment_label = "positive"
        elif scores['compound'] <= -0.05:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return SentimentResult(
            text=text,
            polarity=scores['compound'],
            compound_score=scores['compound'],
            sentiment_label=sentiment_label
        )
    
    def _textblob_analyze(self, text: str) -> SentimentResult:
        """Analyze using TextBlob."""
        if 'textblob' not in self.analyzers:
            raise ValueError("TextBlob analyzer not available")
        
        blob = TextBlob(text)
        
        # Determine sentiment label
        if blob.sentiment.polarity > 0:
            sentiment_label = "positive"
        elif blob.sentiment.polarity < 0:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        return SentimentResult(
            text=text,
            polarity=blob.sentiment.polarity,
            subjectivity=blob.sentiment.subjectivity,
            sentiment_label=sentiment_label
        )
    
    def _transformer_analyze(self, text: str) -> SentimentResult:
        """Analyze using transformer model."""
        if 'transformer' not in self.analyzers:
            raise ValueError("Transformer analyzer not available")
        
        try:
            results = self.analyzers['transformer'](text)
            
            # Get the highest scoring sentiment
            best_result = max(results[0], key=lambda x: x['score'])
            
            # Map labels to polarity
            label_to_polarity = {
                'positive': 1.0,
                'negative': -1.0,
                'neutral': 0.0
            }
            
            polarity = label_to_polarity.get(best_result['label'], 0.0)
            
            return SentimentResult(
                text=text,
                polarity=polarity,
                sentiment_label=best_result['label'],
                confidence=best_result['score']
            )
        except Exception as e:
            logging.error(f"Transformer analysis failed: {e}")
            # Fallback to neutral
            return SentimentResult(
                text=text,
                polarity=0.0,
                sentiment_label="neutral"
            )
    
    def _ensemble_analyze(self, text: str) -> SentimentResult:
        """Analyze using ensemble of available models."""
        results = []
        
        # Collect results from all available analyzers
        if 'vader' in self.analyzers:
            results.append(self._vader_analyze(text))
        
        if 'textblob' in self.analyzers:
            results.append(self._textblob_analyze(text))
        
        if 'transformer' in self.analyzers:
            results.append(self._transformer_analyze(text))
        
        if not results:
            raise ValueError("No analyzers available for ensemble")
        
        # Average the polarities
        avg_polarity = np.mean([r.polarity for r in results])
        
        # Determine sentiment label based on average
        if avg_polarity > 0.1:
            sentiment_label = "positive"
        elif avg_polarity < -0.1:
            sentiment_label = "negative"
        else:
            sentiment_label = "neutral"
        
        # Calculate confidence as standard deviation (lower is more confident)
        polarities = [r.polarity for r in results]
        confidence = 1.0 - min(np.std(polarities), 1.0) if len(polarities) > 1 else 0.8
        
        return SentimentResult(
            text=text,
            polarity=avg_polarity,
            sentiment_label=sentiment_label,
            confidence=confidence
        )
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text
            
        Returns:
            DataFrame with sentiment analysis results
        """
        results = []
        
        for idx, row in df.iterrows():
            text = row[text_column]
            sentiment_result = self.analyze_text(text)
            
            results.append({
                'index': idx,
                'text': text,
                'polarity': sentiment_result.polarity,
                'subjectivity': sentiment_result.subjectivity,
                'compound_score': sentiment_result.compound_score,
                'sentiment_label': sentiment_result.sentiment_label,
                'confidence': sentiment_result.confidence
            })
        
        return pd.DataFrame(results)
    
    def get_sentiment_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary statistics for sentiment analysis.
        
        Args:
            df: DataFrame with sentiment analysis results
            
        Returns:
            Dictionary with summary statistics
        """
        if df.empty:
            return {}
        
        # Basic statistics
        summary = {
            'total_messages': len(df),
            'avg_polarity': df['polarity'].mean(),
            'std_polarity': df['polarity'].std(),
            'min_polarity': df['polarity'].min(),
            'max_polarity': df['polarity'].max()
        }
        
        # Sentiment distribution
        if 'sentiment_label' in df.columns:
            sentiment_counts = df['sentiment_label'].value_counts()
            summary['sentiment_distribution'] = sentiment_counts.to_dict()
            
            # Calculate positive ratio
            positive_count = sentiment_counts.get('positive', 0)
            summary['positive_ratio'] = positive_count / len(df)
        
        # Subjectivity (if available)
        if 'subjectivity' in df.columns and not df['subjectivity'].isna().all():
            summary['avg_subjectivity'] = df['subjectivity'].mean()
        
        # Confidence (if available)
        if 'confidence' in df.columns and not df['confidence'].isna().all():
            summary['avg_confidence'] = df['confidence'].mean()
        
        # Most extreme sentiments
        if not df.empty:
            most_positive = df.loc[df['polarity'].idxmax()]
            most_negative = df.loc[df['polarity'].idxmin()]
            
            summary['most_positive_text'] = most_positive['text']
            summary['most_positive_score'] = most_positive['polarity']
            summary['most_negative_text'] = most_negative['text']
            summary['most_negative_score'] = most_negative['polarity']
        
        return summary


class SentimentTrendAnalyzer:
    """Analyzes sentiment trends over time or sequence."""
    
    def __init__(self, window_size: int = 5):
        """
        Initialize trend analyzer.
        
        Args:
            window_size: Size of moving average window
        """
        self.window_size = window_size
    
    def analyze_trend(self, df: pd.DataFrame, polarity_column: str = 'polarity') -> pd.DataFrame:
        """
        Analyze sentiment trends using moving averages.
        
        Args:
            df: DataFrame with sentiment data
            polarity_column: Column name for polarity scores
            
        Returns:
            DataFrame with trend analysis
        """
        df_copy = df.copy()
        
        # Calculate moving average
        df_copy['moving_avg'] = df_copy[polarity_column].rolling(
            window=self.window_size, center=True
        ).mean()
        
        # Calculate trend direction
        df_copy['trend'] = df_copy['moving_avg'].diff()
        
        # Determine overall trend
        if len(df_copy) > 1:
            overall_trend = df_copy['moving_avg'].iloc[-1] - df_copy['moving_avg'].iloc[0]
            df_copy['overall_trend'] = overall_trend
        else:
            df_copy['overall_trend'] = 0
        
        return df_copy
    
    def get_trend_summary(self, df: pd.DataFrame) -> Dict:
        """
        Get summary of sentiment trends.
        
        Args:
            df: DataFrame with trend analysis
            
        Returns:
            Dictionary with trend summary
        """
        if df.empty:
            return {}
        
        summary = {
            'trend_direction': 'stable',
            'trend_magnitude': 0.0,
            'volatility': df['polarity'].std() if 'polarity' in df.columns else 0.0
        }
        
        if 'overall_trend' in df.columns and len(df) > 1:
            trend = df['overall_trend'].iloc[0]
            summary['trend_magnitude'] = abs(trend)
            
            if trend > 0.1:
                summary['trend_direction'] = 'improving'
            elif trend < -0.1:
                summary['trend_direction'] = 'declining'
        
        return summary


# Example usage and testing
if __name__ == "__main__":
    # Test sentiment analyzer
    analyzer = SentimentAnalyzer(model_type="vader")
    
    test_texts = [
        "I love this project!",
        "This is terrible.",
        "The meeting was okay.",
        "I'm feeling a bit burned out.",
        "Let's make this work together!"
    ]
    
    print("Testing sentiment analysis:")
    for text in test_texts:
        result = analyzer.analyze_text(text)
        print(f"'{text}' -> {result.sentiment_label} ({result.polarity:.3f})")
    
    # Test with DataFrame
    df = pd.DataFrame({'text': test_texts})
    results_df = analyzer.analyze_dataframe(df)
    print("\nDataFrame results:")
    print(results_df)
    
    # Test summary
    summary = analyzer.get_sentiment_summary(results_df)
    print("\nSummary:")
    print(summary) 