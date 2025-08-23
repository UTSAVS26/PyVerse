"""
Keyword Extraction Module

Extracts keywords and key concepts from text using various NLP techniques.
"""

import re
from typing import List, Dict, Tuple, Optional
import logging
from collections import Counter
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from rake_nltk import Rake
import spacy

logger = logging.getLogger(__name__)


class KeywordExtractor:
    """Extracts keywords and key concepts from text using multiple methods."""
    
    def __init__(self, use_spacy: bool = True, use_keybert: bool = False):
        """
        Initialize the keyword extractor.
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP
            use_keybert: Whether to use KeyBERT for keyword extraction
        """
        self.use_spacy = use_spacy
        self.use_keybert = use_keybert
        
        # Download required NLTK data
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
            
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        # Initialize spaCy if requested
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        
        # Initialize KeyBERT if requested
        if self.use_keybert:
            try:
                from keybert import KeyBERT
                self.keybert_model = KeyBERT()
            except ImportError:
                logger.warning("KeyBERT not available. Install with: pip install keybert")
                self.use_keybert = False
        
        # Initialize RAKE
        self.rake = Rake()
        
        # Get stopwords
        self.stop_words = set(stopwords.words('english'))
        
        # Add custom stopwords for academic text
        academic_stopwords = {
            'chapter', 'section', 'figure', 'table', 'page', 'pages',
            'example', 'examples', 'note', 'notes', 'see', 'also',
            'et', 'al', 'etc', 'ie', 'eg', 'cf', 'ibid', 'op', 'cit'
        }
        self.stop_words.update(academic_stopwords)
    
    def extract_keywords_tfidf(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords using TF-IDF approach.
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        if not text:
            return []
        
        # Tokenize and clean
        words = word_tokenize(text.lower())
        words = [word for word in words if word.isalpha() and word not in self.stop_words]
        
        # Count word frequencies
        word_freq = Counter(words)
        
        # Calculate TF-IDF-like scores (simplified)
        total_words = len(words)
        keyword_scores = []
        
        for word, freq in word_freq.items():
            if len(word) > 2:  # Filter out very short words
                # Simple TF-IDF approximation
                tf = freq / total_words
                # Bonus for longer words (often more specific)
                length_bonus = min(len(word) / 10, 1.0)
                score = tf * (1 + length_bonus)
                keyword_scores.append((word, score))
        
        # Sort by score and return top_k
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return keyword_scores[:top_k]
    
    def extract_keywords_rake(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords using RAKE algorithm.
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        if not text:
            return []
        
        self.rake.extract_keywords_from_text(text)
        keywords = self.rake.get_ranked_phrases_with_scores()
        
        # Convert to our format and filter
        keyword_scores = []
        for score, phrase in keywords:
            if len(phrase.split()) <= 3:  # Limit to 3-word phrases
                keyword_scores.append((phrase, score))
        
        return keyword_scores[:top_k]
    
    def extract_keywords_spacy(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords using spaCy NLP features.
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        if not text or not self.use_spacy:
            return []
        
        doc = self.nlp(text)
        keyword_scores = []
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ['PERSON', 'ORG', 'GPE', 'PRODUCT', 'EVENT', 'WORK_OF_ART']:
                keyword_scores.append((ent.text, 2.0))  # High score for entities
        
        # Extract noun phrases
        for chunk in doc.noun_chunks:
            if len(chunk.text.split()) <= 3:  # Limit phrase length
                # Score based on POS tags
                score = 1.0
                for token in chunk:
                    if token.pos_ in ['NOUN', 'PROPN']:
                        score += 0.5
                    elif token.pos_ == 'ADJ':
                        score += 0.3
                keyword_scores.append((chunk.text, score))
        
        # Extract important individual words
        for token in doc:
            if (token.pos_ in ['NOUN', 'PROPN', 'ADJ'] and 
                not token.is_stop and 
                len(token.text) > 2):
                score = 1.0
                if token.pos_ == 'PROPN':
                    score += 0.5
                keyword_scores.append((token.text, score))
        
        # Sort by score and return top_k
        keyword_scores.sort(key=lambda x: x[1], reverse=True)
        return keyword_scores[:top_k]
    
    def extract_keywords_keybert(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords using KeyBERT.
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        if not text or not self.use_keybert:
            return []
        
        try:
            keywords = self.keybert_model.extract_keywords(text, 
                                                         keyphrase_ngram_range=(1, 3),
                                                         stop_words='english',
                                                         use_maxsum=True,
                                                         nr_candidates=top_k * 2,
                                                         top_n=top_k)
            return [(keyword, score) for keyword, score in keywords]
        except Exception as e:
            logger.warning(f"KeyBERT extraction failed: {e}")
            return []
    
    def extract_keywords_combined(self, text: str, top_k: int = 20) -> List[Tuple[str, float]]:
        """
        Extract keywords using multiple methods and combine results.
        
        Args:
            text: Input text
            top_k: Number of top keywords to return
            
        Returns:
            List of (keyword, score) tuples
        """
        all_keywords = {}
        
        # Extract using different methods
        methods = [
            self.extract_keywords_tfidf,
            self.extract_keywords_rake
        ]
        
        if self.use_spacy:
            methods.append(self.extract_keywords_spacy)
        
        if self.use_keybert:
            methods.append(self.extract_keywords_keybert)
        
        # Collect keywords from all methods
        for method in methods:
            try:
                keywords = method(text, top_k)
                for keyword, score in keywords:
                    keyword_lower = keyword.lower().strip()
                    if keyword_lower not in all_keywords:
                        all_keywords[keyword_lower] = []
                    all_keywords[keyword_lower].append(score)
            except Exception as e:
                logger.warning(f"Method {method.__name__} failed: {e}")
        
        # Combine scores (average across methods)
        combined_keywords = []
        for keyword, scores in all_keywords.items():
            avg_score = sum(scores) / len(scores)
            combined_keywords.append((keyword, avg_score))
        
        # Sort by score and return top_k
        combined_keywords.sort(key=lambda x: x[1], reverse=True)
        return combined_keywords[:top_k]
    
    def extract_key_concepts(self, text: str, min_length: int = 3) -> List[str]:
        """
        Extract key concepts (longer phrases) from text.
        
        Args:
            text: Input text
            min_length: Minimum word length for concepts
            
        Returns:
            List of key concepts
        """
        if not text:
            return []
        
        # Use RAKE for concept extraction
        self.rake.extract_keywords_from_text(text)
        concepts = self.rake.get_ranked_phrases_with_scores()
        
        # Filter by length and score
        key_concepts = []
        for score, concept in concepts:
            words = concept.split()
            if len(words) >= min_length and score > 1.0:
                key_concepts.append(concept)
        
        return key_concepts[:10]  # Return top 10 concepts
    
    def get_context_for_keyword(self, text: str, keyword: str, context_size: int = 100) -> List[str]:
        """
        Get context sentences for a specific keyword.
        
        Args:
            text: Input text
            keyword: Keyword to find context for
            context_size: Number of characters around keyword
            
        Returns:
            List of context sentences
        """
        if not text or not keyword:
            return []
        
        sentences = sent_tokenize(text)
        context_sentences = []
        
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                context_sentences.append(sentence)
        
        return context_sentences
