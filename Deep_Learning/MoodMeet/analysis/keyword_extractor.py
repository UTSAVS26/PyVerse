"""
Keyword Extractor Module for MoodMeet

Provides keyword extraction using RAKE, YAKE, and TF-IDF methods.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import re
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Import keyword extraction libraries
try:
    import yake
    YAKE_AVAILABLE = True
except ImportError:
    YAKE_AVAILABLE = False

try:
    from rake_nltk import Rake
    RAKE_AVAILABLE = True
except ImportError:
    RAKE_AVAILABLE = False


@dataclass
class KeywordResult:
    """Represents a keyword extraction result."""
    keyword: str
    score: float
    method: str
    frequency: int = 1


class KeywordExtractor:
    """Extracts keywords from text using multiple methods."""
    
    def __init__(self, method: str = "tfidf"):
        """
        Initialize keyword extractor.
        
        Args:
            method: Extraction method ('tfidf', 'rake', 'yake', 'ensemble')
        """
        self.method = method
        self.vectorizer = None
        self.stop_words = self._get_stop_words()
        
    def _get_stop_words(self) -> set:
        """Get common stop words."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should',
            'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us',
            'them', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was', 'were',
            'very', 'just', 'really', 'quite', 'rather', 'too', 'so', 'much', 'many',
            'few', 'several', 'some', 'any', 'all', 'each', 'every', 'no', 'not',
            'can', 'cannot', 'could', 'would', 'should', 'may', 'might', 'must'
        }
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters but keep spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def _extract_tfidf_keywords(self, texts: List[str], max_keywords: int = 20) -> List[KeywordResult]:
        """Extract keywords using TF-IDF."""
        if not texts:
            return []
        
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Create TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=1,
            max_df=0.8
        )
        
        # Fit and transform
        tfidf_matrix = self.vectorizer.fit_transform(cleaned_texts)
        feature_names = self.vectorizer.get_feature_names_out()
        
        # Get average TF-IDF scores across all documents
        avg_scores = np.mean(tfidf_matrix.toarray(), axis=0)
        
        # Create keyword results
        keywords = []
        for i, score in enumerate(avg_scores):
            if score > 0:
                keyword = feature_names[i]
                keywords.append(KeywordResult(
                    keyword=keyword,
                    score=score,
                    method="tfidf"
                ))
        
        # Sort by score and return top keywords
        keywords.sort(key=lambda x: x.score, reverse=True)
        return keywords[:max_keywords]
    
    def _extract_rake_keywords(self, texts: List[str], max_keywords: int = 20) -> List[KeywordResult]:
        """Extract keywords using RAKE."""
        if not RAKE_AVAILABLE:
            return []
        
        if not texts:
            return []
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Initialize RAKE
        rake = Rake()
        rake.extract_keywords_from_text(combined_text)
        
        # Get keyword scores
        keyword_scores = rake.get_ranked_phrases_with_scores()
        
        # Convert to KeywordResult objects
        keywords = []
        for score, keyword in keyword_scores[:max_keywords]:
            keywords.append(KeywordResult(
                keyword=keyword,
                score=score,
                method="rake"
            ))
        
        return keywords
    
    def _extract_yake_keywords(self, texts: List[str], max_keywords: int = 20) -> List[KeywordResult]:
        """Extract keywords using YAKE."""
        if not YAKE_AVAILABLE:
            return []
        
        if not texts:
            return []
        
        # Combine all texts
        combined_text = ' '.join(texts)
        
        # Initialize YAKE
        kw_extractor = yake.KeywordExtractor(
            lan="en",
            n=1,  # Unigrams
            dedupLim=0.9,
            dedupFunc='seqm',
            windowsSize=3,
            top=max_keywords
        )
        
        # Extract keywords
        keywords_list = kw_extractor.extract_keywords(combined_text)
        
        # Convert to KeywordResult objects
        keywords = []
        for keyword, score in keywords_list:
            keywords.append(KeywordResult(
                keyword=keyword,
                score=score,
                method="yake"
            ))
        
        return keywords
    
    def _extract_ensemble_keywords(self, texts: List[str], max_keywords: int = 20) -> List[KeywordResult]:
        """Extract keywords using ensemble of methods."""
        all_keywords = []
        
        # Collect keywords from all available methods
        if self.method in ["tfidf", "ensemble"]:
            tfidf_keywords = self._extract_tfidf_keywords(texts, max_keywords)
            all_keywords.extend(tfidf_keywords)
        
        if RAKE_AVAILABLE and self.method in ["rake", "ensemble"]:
            rake_keywords = self._extract_rake_keywords(texts, max_keywords)
            all_keywords.extend(rake_keywords)
        
        if YAKE_AVAILABLE and self.method in ["yake", "ensemble"]:
            yake_keywords = self._extract_yake_keywords(texts, max_keywords)
            all_keywords.extend(yake_keywords)
        
        if not all_keywords:
            return []
        
        # Combine and rank keywords
        keyword_scores = {}
        for keyword_result in all_keywords:
            keyword = keyword_result.keyword.lower()
            if keyword not in keyword_scores:
                keyword_scores[keyword] = {
                    'score': 0,
                    'count': 0,
                    'methods': set()
                }
            
            keyword_scores[keyword]['score'] += keyword_result.score
            keyword_scores[keyword]['count'] += 1
            keyword_scores[keyword]['methods'].add(keyword_result.method)
        
        # Calculate ensemble scores
        ensemble_keywords = []
        for keyword, data in keyword_scores.items():
            # Average score across methods
            avg_score = data['score'] / data['count']
            # Bonus for methods agreement
            method_bonus = len(data['methods']) * 0.1
            final_score = avg_score + method_bonus
            
            ensemble_keywords.append(KeywordResult(
                keyword=keyword,
                score=final_score,
                method="ensemble",
                frequency=data['count']
            ))
        
        # Sort by score and return top keywords
        ensemble_keywords.sort(key=lambda x: x.score, reverse=True)
        return ensemble_keywords[:max_keywords]
    
    def extract_keywords(self, texts: List[str], max_keywords: int = 20) -> List[KeywordResult]:
        """
        Extract keywords from texts.
        
        Args:
            texts: List of texts to analyze
            max_keywords: Maximum number of keywords to return
            
        Returns:
            List of KeywordResult objects
        """
        if not texts:
            return []
        
        if self.method == "tfidf":
            return self._extract_tfidf_keywords(texts, max_keywords)
        elif self.method == "rake":
            return self._extract_rake_keywords(texts, max_keywords)
        elif self.method == "yake":
            return self._extract_yake_keywords(texts, max_keywords)
        elif self.method == "ensemble":
            return self._extract_ensemble_keywords(texts, max_keywords)
        else:
            raise ValueError(f"Unknown extraction method: {self.method}")
    
    def get_keyword_summary(self, keywords: List[KeywordResult]) -> Dict:
        """
        Get summary of keyword extraction results.
        
        Args:
            keywords: List of KeywordResult objects
            
        Returns:
            Dictionary with keyword summary
        """
        if not keywords:
            return {}
        
        summary = {
            'total_keywords': len(keywords),
            'methods_used': list(set(kw.method for kw in keywords)),
            'top_keywords': [],
            'keyword_categories': {}
        }
        
        # Top keywords
        for keyword in keywords[:10]:
            summary['top_keywords'].append({
                'keyword': keyword.keyword,
                'score': keyword.score,
                'method': keyword.method,
                'frequency': keyword.frequency
            })
        
        # Categorize keywords by sentiment
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'fantastic', 
                         'love', 'like', 'enjoy', 'happy', 'excited', 'confident', 'success'}
        negative_words = {'bad', 'terrible', 'awful', 'hate', 'dislike', 'sad', 'angry', 
                         'frustrated', 'burned', 'out', 'stress', 'pressure', 'deadline'}
        
        for keyword in keywords:
            word = keyword.keyword.lower()
            if word in positive_words:
                category = 'positive'
            elif word in negative_words:
                category = 'negative'
            else:
                category = 'neutral'
            
            if category not in summary['keyword_categories']:
                summary['keyword_categories'][category] = []
            
            summary['keyword_categories'][category].append({
                'keyword': keyword.keyword,
                'score': keyword.score
            })
        
        return summary


class PhraseExtractor:
    """Extracts meaningful phrases from text."""
    
    def __init__(self):
        self.stop_words = self._get_stop_words()
    
    def _get_stop_words(self) -> set:
        """Get common stop words."""
        return {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 
            'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 
            'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'
        }
    
    def extract_phrases(self, texts: List[str], min_length: int = 2, max_length: int = 4) -> List[Dict]:
        """
        Extract meaningful phrases from texts.
        
        Args:
            texts: List of texts to analyze
            min_length: Minimum phrase length
            max_length: Maximum phrase length
            
        Returns:
            List of phrase dictionaries
        """
        if not texts:
            return []
        
        # Combine all texts
        combined_text = ' '.join(texts).lower()
        
        # Extract n-grams
        words = combined_text.split()
        phrases = []
        
        for n in range(min_length, max_length + 1):
            for i in range(len(words) - n + 1):
                phrase = ' '.join(words[i:i+n])
                
                # Filter out phrases with too many stop words
                phrase_words = phrase.split()
                stop_word_count = sum(1 for word in phrase_words if word in self.stop_words)
                
                if stop_word_count < len(phrase_words) * 0.5:  # Less than 50% stop words
                    phrases.append(phrase)
        
        # Count phrase frequencies
        phrase_counts = Counter(phrases)
        
        # Convert to result format
        results = []
        for phrase, count in phrase_counts.most_common(50):
            if count > 1:  # Only phrases that appear more than once
                results.append({
                    'phrase': phrase,
                    'frequency': count,
                    'length': len(phrase.split())
                })
        
        return results


# Example usage and testing
if __name__ == "__main__":
    # Test keyword extraction
    texts = [
        "We're falling behind schedule.",
        "Let's regroup and finish the draft today.",
        "I'm feeling a bit burned out.",
        "I think we can make it work if we focus.",
        "That sounds like a good plan.",
        "The deadline is approaching fast.",
        "We need to prioritize our tasks.",
        "I'm confident we can deliver on time."
    ]
    
    # Test TF-IDF extraction
    extractor = KeywordExtractor(method="tfidf")
    keywords = extractor.extract_keywords(texts, max_keywords=10)
    
    print("TF-IDF Keywords:")
    for keyword in keywords:
        print(f"'{keyword.keyword}': {keyword.score:.4f}")
    
    print("\nKeyword Summary:")
    summary = extractor.get_keyword_summary(keywords)
    print(f"Total keywords: {summary['total_keywords']}")
    print(f"Methods used: {summary['methods_used']}")
    
    # Test phrase extraction
    phrase_extractor = PhraseExtractor()
    phrases = phrase_extractor.extract_phrases(texts)
    
    print("\nExtracted Phrases:")
    for phrase in phrases[:10]:
        print(f"'{phrase['phrase']}': {phrase['frequency']} times") 