"""
Transformer Sentiment Analysis Model for MoodMeet

Provides advanced transformer-based sentiment analysis using HuggingFace models.
"""

import pandas as pd
from typing import Dict, List, Tuple
from dataclasses import dataclass
import logging
import torch
from transformers import (
    pipeline, 
    AutoTokenizer, 
    AutoModelForSequenceClassification,
)
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


@dataclass
class TransformerResult:
    """Represents transformer sentiment analysis result."""
    text: str
    sentiment_label: str
    confidence: float
    polarity: float
    model_name: str
    processing_time: float


class TransformerSentimentAnalyzer:
    """Advanced transformer-based sentiment analyzer."""

    def __init__(self, model_name: str = "cardiffnlp/twitter-roberta-base-sentiment"):
        """
        Initialize transformer sentiment analyzer.

        Args:
            model_name: HuggingFace model name
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.pipeline = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        try:
            self._load_model()
        except Exception as e:
            logging.error(f"Failed to load transformer model: {e}")
            raise

    def _load_model(self):
        """Load the transformer model and tokenizer."""
        try:
            # Load tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name)

            # Move to device
            self.model.to(self.device)

            # Create pipeline
            self.pipeline = pipeline(
                "sentiment-analysis",
                model=self.model,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1,
                return_all_scores=True
            )

            logging.info(f"Successfully loaded transformer model: {self.model_name}")

        except Exception as e:
            logging.error(f"Error loading model {self.model_name}: {e}")
            raise
    def _map_label_to_polarity(self, label: str) -> float:
        """
        Map model-specific labels to a normalized polarity in [-1.0, 1.0].
        Handles:
        - positive/negative/neutral (any case)
        - LABEL_0/LABEL_1/LABEL_2 (common 3-class)
        - "1 star"..."5 stars" (nlptown 5-class)
        """
        if not label:
            return 0.0
        norm = label.strip().lower()
        base = {
            'positive': 1.0,
            'negative': -1.0,
            'neutral': 0.0,
            'label_0': -1.0,
            'label_1': 0.0,
            'label_2': 1.0,
        }
        if norm in base:
            return base[norm]
        import re
        m = re.match(r'([1-5])\s*star', norm)
        if m:
            stars = int(m.group(1))
            # scale: 1→-1.0, 3→0.0, 5→+1.0
            return (stars - 3) / 2.0
        return 0.0

    def analyze_text(self, text: str) -> TransformerResult:
        """
        Analyze sentiment of a single text.
        
        Args:
            text: Text to analyze
            
        Returns:
            TransformerResult object
        """
        if not self.pipeline:
            raise ValueError("Model not loaded")
        
        import time
        start_time = time.time()
        
        try:
            # Get sentiment scores
            results = self.pipeline(text)
            
            # Find the highest scoring sentiment
            best_result = max(results[0], key=lambda x: x['score'])
            
            # Compute polarity via the centralized helper
            polarity = self._map_label_to_polarity(best_result['label'])
            
            processing_time = time.time() - start_time
            # … rest of method …
            
            return TransformerResult(
                text=text,
                sentiment_label=best_result['label'],
                confidence=best_result['score'],
                polarity=polarity,
                model_name=self.model_name,
                processing_time=processing_time
            )
            
        except Exception as e:
            logging.error(f"Error analyzing text: {e}")
            # Return neutral result on error
            return TransformerResult(
                text=text,
                sentiment_label="neutral",
                confidence=0.0,
                polarity=0.0,
                model_name=self.model_name,
                processing_time=time.time() - start_time
            )
    
    def analyze_batch(self, texts: List[str], batch_size: int = 8) -> List[TransformerResult]:
        """
        Analyze sentiment of multiple texts in batches.
        
        Args:
            texts: List of texts to analyze
            batch_size: Batch size for processing
            
        Returns:
            List of TransformerResult objects
        """
        if not self.pipeline:
            raise ValueError("Model not loaded")
        
        results = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            try:
                batch_results = self.pipeline(batch_texts)
                
                for j, text in enumerate(batch_texts):
                    if j < len(batch_results):
                        result = batch_results[j]
                        best_result = max(result, key=lambda x: x['score'])
                        
                        polarity = self._map_label_to_polarity(best_result['label'])
                        transformer_result = TransformerResult(
                            text=text,
                            sentiment_label=best_result['label'],
                            confidence=best_result['score'],
                            polarity=polarity,
                            model_name=self.model_name,
                            processing_time=0.0  # Batch processing time not tracked individually
                        )
                        
                        results.append(transformer_result)
                    else:
                        # Fallback for missing results
                        results.append(TransformerResult(
                            text=text,
                            sentiment_label="neutral",
                            confidence=0.0,
                            polarity=0.0,
                            model_name=self.model_name,
                            processing_time=0.0
                        ))
                        
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
                # Add neutral results for failed batch
                for text in batch_texts:
                    results.append(TransformerResult(
                        text=text,
                        sentiment_label="neutral",
                        confidence=0.0,
                        polarity=0.0,
                        model_name=self.model_name,
                        processing_time=0.0
                    ))
        
        return results
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str = 'text') -> pd.DataFrame:
        """
        Analyze sentiment for all texts in a DataFrame.
        
        Args:
            df: DataFrame containing text data
            text_column: Name of the column containing text
            
        Returns:
            DataFrame with transformer analysis results
        """
        if df.empty:
            return pd.DataFrame()
        
        texts = df[text_column].tolist()
        results = self.analyze_batch(texts)
        
        # Convert to DataFrame
        result_data = []
        for result in results:
            result_data.append({
                'text': result.text,
                'transformer_sentiment': result.sentiment_label,
                'transformer_confidence': result.confidence,
                'transformer_polarity': result.polarity,
                'processing_time': result.processing_time
            })
        
        return pd.DataFrame(result_data)
    
    def get_model_info(self) -> Dict:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary with model information
        """
        return {
            'model_name': self.model_name,
            'device': self.device,
            'model_loaded': self.model is not None,
            'pipeline_loaded': self.pipeline is not None
        }
    
    def evaluate_model(self, test_data: List[Tuple[str, str]]) -> Dict:
        """
        Evaluate model performance on test data.
        
        Args:
            test_data: List of (text, expected_label) tuples
            
        Returns:
            Dictionary with evaluation metrics
        """
        if not test_data:
            return {}
        
        texts = [item[0] for item in test_data]
        expected_labels = [item[1] for item in test_data]
        
        # Get predictions
        results = self.analyze_batch(texts)
        predicted_labels = [result.sentiment_label for result in results]
        
        # Calculate metrics
        try:
            report = classification_report(expected_labels, predicted_labels, output_dict=True)
            conf_matrix = confusion_matrix(expected_labels, predicted_labels)
            
            evaluation = {
                'accuracy': report.get('accuracy', 0.0),
                'precision': report.get('weighted avg', {}).get('precision', 0.0),
                'recall': report.get('weighted avg', {}).get('recall', 0.0),
                'f1_score': report.get('weighted avg', {}).get('f1-score', 0.0),
                'confusion_matrix': conf_matrix.tolist(),
                'detailed_report': report
            }
            
            return evaluation
            
        except Exception as e:
            logging.error(f"Error evaluating model: {e}")
            return {}


class ModelManager:
    """Manages multiple transformer models."""
    
    def __init__(self):
        self.models = {}
        self.available_models = {
            'twitter-roberta': 'cardiffnlp/twitter-roberta-base-sentiment',
            'distilbert': 'distilbert-base-uncased-finetuned-sst-2-english',
            'bert-base': 'nlptown/bert-base-multilingual-uncased-sentiment'
        }
    
    def load_model(self, model_key: str) -> TransformerSentimentAnalyzer:
        """
        Load a specific model.
        
        Args:
            model_key: Key for the model to load
            
        Returns:
            TransformerSentimentAnalyzer instance
        """
        if model_key not in self.available_models:
            raise ValueError(f"Unknown model key: {model_key}")
        
        if model_key not in self.models:
            model_name = self.available_models[model_key]
            self.models[model_key] = TransformerSentimentAnalyzer(model_name)
        
        return self.models[model_key]
    
    def get_available_models(self) -> List[str]:
        """Get list of available model keys."""
        return list(self.available_models.keys())
    
    def compare_models(self, texts: List[str], model_keys: List[str]) -> pd.DataFrame:
        """
        Compare multiple models on the same texts.
        
        Args:
            texts: List of texts to analyze
            model_keys: List of model keys to compare
            
        Returns:
            DataFrame with comparison results
        """
        results = []
        
        for model_key in model_keys:
            try:
                model = self.load_model(model_key)
                model_results = model.analyze_batch(texts)
                
                for i, result in enumerate(model_results):
                    results.append({
                        'text': texts[i],
                        'model': model_key,
                        'sentiment': result.sentiment_label,
                        'confidence': result.confidence,
                        'polarity': result.polarity
                    })
                    
            except Exception as e:
                logging.error(f"Error with model {model_key}: {e}")
                # Add neutral results for failed model
                for text in texts:
                    results.append({
                        'text': text,
                        'model': model_key,
                        'sentiment': 'neutral',
                        'confidence': 0.0,
                        'polarity': 0.0
                    })
        
        return pd.DataFrame(results)


# Example usage and testing
if __name__ == "__main__":
    # Test transformer analyzer
    try:
        analyzer = TransformerSentimentAnalyzer()
        
        test_texts = [
            "I love this project!",
            "This is terrible.",
            "The meeting was okay.",
            "I'm feeling a bit burned out.",
            "Let's make this work together!"
        ]
        
        print("Testing transformer sentiment analysis:")
        for text in test_texts:
            result = analyzer.analyze_text(text)
            print(f"'{text}' -> {result.sentiment_label} ({result.confidence:.3f})")
        
        # Test batch processing
        print("\nTesting batch processing:")
        batch_results = analyzer.analyze_batch(test_texts)
        for result in batch_results:
            print(f"'{result.text}' -> {result.sentiment_label}")
        
        # Test DataFrame processing
        df = pd.DataFrame({'text': test_texts})
        results_df = analyzer.analyze_dataframe(df)
        print("\nDataFrame results:")
        print(results_df)
        
        # Test model manager
        print("\nTesting model manager:")
        manager = ModelManager()
        available_models = manager.get_available_models()
        print(f"Available models: {available_models}")
        
    except Exception as e:
        print(f"Error testing transformer analyzer: {e}")
        print("This is expected if transformers library is not available.") 