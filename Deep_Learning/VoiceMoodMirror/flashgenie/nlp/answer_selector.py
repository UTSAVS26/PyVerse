"""
Answer Selector Module

Selects appropriate answers for generated questions from source text.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
import logging
from nltk.tokenize import sent_tokenize
import spacy

logger = logging.getLogger(__name__)


class AnswerSelector:
    """Selects appropriate answers for questions from source text."""
    
    def __init__(self, use_spacy: bool = True):
        """
        Initialize the answer selector.
        
        Args:
            use_spacy: Whether to use spaCy for advanced NLP
        """
        self.use_spacy = use_spacy
        
        # Initialize spaCy if requested
        if self.use_spacy:
            try:
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                logger.warning("spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.use_spacy = False
        
        # Answer extraction patterns
        self.answer_patterns = {
            'definition': [
                r'is\s+(?:a|an)\s+([^.]*)',
                r'defined\s+as\s+([^.]*)',
                r'refers\s+to\s+([^.]*)',
                r'means\s+([^.]*)'
            ],
            'function': [
                r'function\s+is\s+([^.]*)',
                r'purpose\s+is\s+([^.]*)',
                r'used\s+to\s+([^.]*)',
                r'responsible\s+for\s+([^.]*)'
            ],
            'process': [
                r'process\s+involves\s+([^.]*)',
                r'steps\s+include\s+([^.]*)',
                r'procedure\s+is\s+([^.]*)'
            ]
        }
    
    def find_answer_for_question(self, question: str, text: str, max_length: int = 200) -> str:
        """
        Find the best answer for a given question from the text.
        
        Args:
            question: The question to find an answer for
            text: Source text to search in
            max_length: Maximum length of the answer
            
        Returns:
            Best matching answer
        """
        if not question or not text:
            return ""
        
        # Extract key terms from question
        key_terms = self._extract_key_terms_from_question(question)
        
        if not key_terms:
            return ""
        
        # Find relevant sentences
        relevant_sentences = self._find_relevant_sentences(text, key_terms)
        
        if not relevant_sentences:
            return ""
        
        # Select the best answer
        best_answer = self._select_best_answer(question, relevant_sentences, max_length)
        
        return best_answer
    
    def find_answers_for_questions(self, questions: List[str], text: str, max_length: int = 200) -> List[str]:
        """
        Find answers for multiple questions.
        
        Args:
            questions: List of questions
            text: Source text
            max_length: Maximum length of each answer
            
        Returns:
            List of answers corresponding to questions
        """
        answers = []
        
        for question in questions:
            answer = self.find_answer_for_question(question, text, max_length)
            answers.append(answer)
        
        return answers
    
    def extract_definitions(self, text: str, keywords: List[str]) -> Dict[str, str]:
        """
        Extract definitions for keywords from text.
        
        Args:
            text: Source text
            keywords: List of keywords to find definitions for
            
        Returns:
            Dictionary mapping keywords to their definitions
        """
        definitions = {}
        
        for keyword in keywords:
            definition = self._extract_definition_for_keyword(text, keyword)
            if definition:
                definitions[keyword] = definition
        
        return definitions
    
    def _extract_key_terms_from_question(self, question: str) -> List[str]:
        """Extract key terms from a question."""
        if not question:
            return []
        
        # Remove question words and common words
        question_lower = question.lower()
        question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'is', 'are', 'does', 'do', 'the', 'a', 'an']
        
        for word in question_words:
            question_lower = question_lower.replace(word, '')
        
        # Extract potential key terms (capitalized words, phrases, and important nouns)
        words = question.split()
        key_terms = []
        
        for i, word in enumerate(words):
            # Clean the word (remove punctuation)
            clean_word = word.strip('?.,!;:')
            
            # Skip question words even if they're capitalized
            if clean_word.lower() in question_words:
                continue
            
            # Look for capitalized words
            if clean_word[0].isupper() and len(clean_word) > 2:
                # Check if it's part of a phrase
                phrase = clean_word
                j = i + 1
                while j < len(words) and words[j][0].isupper():
                    phrase += " " + words[j].strip('?.,!;:')
                    j += 1
                
                key_terms.append(phrase)
            # Also look for important nouns (words that are not question words)
            elif clean_word.lower() not in question_words and len(clean_word) > 2:
                key_terms.append(clean_word)
        
        return key_terms
    
    def _find_relevant_sentences(self, text: str, key_terms: List[str]) -> List[str]:
        """Find sentences that contain the key terms."""
        sentences = sent_tokenize(text)
        relevant_sentences = []
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            
            # Check if any key term appears in the sentence
            for term in key_terms:
                if term.lower() in sentence_lower:
                    relevant_sentences.append(sentence)
                    break
        
        return relevant_sentences
    
    def _select_best_answer(self, question: str, sentences: List[str], max_length: int) -> str:
        """Select the best answer from a list of sentences."""
        if not sentences:
            return ""
        
        # Score sentences based on relevance
        scored_sentences = []
        
        for sentence in sentences:
            score = self._calculate_relevance_score(question, sentence)
            scored_sentences.append((sentence, score))
        
        # Sort by score and select the best
        scored_sentences.sort(key=lambda x: x[1], reverse=True)
        
        best_sentence = scored_sentences[0][0]
        
        # Truncate if too long
        if len(best_sentence) > max_length:
            # Try to truncate at sentence boundary
            truncated = best_sentence[:max_length]
            last_period = truncated.rfind('.')
            if last_period > max_length * 0.7:  # If we can find a good break point
                best_sentence = truncated[:last_period + 1]
            else:
                best_sentence = truncated + "..."
        
        return best_sentence.strip()
    
    def _calculate_relevance_score(self, question: str, sentence: str) -> float:
        """Calculate relevance score between question and sentence."""
        if not question or not sentence:
            return 0.0
        
        score = 0.0
        
        # Extract key terms from question
        question_terms = set(question.lower().split())
        sentence_terms = set(sentence.lower().split())
        
        # Calculate term overlap
        overlap = len(question_terms.intersection(sentence_terms))
        score += overlap * 0.5
        
        # Bonus for exact phrase matches
        question_lower = question.lower()
        sentence_lower = sentence.lower()
        
        # Look for exact matches of key terms
        key_terms = self._extract_key_terms_from_question(question)
        for term in key_terms:
            if term.lower() in sentence_lower:
                score += 5.0  # Much higher weight for key term matches
        
        # Bonus for definition patterns
        if self._contains_definition_pattern(sentence):
            score += 1.0
        
        # Penalty for very long sentences
        if len(sentence.split()) > 50:
            score -= 0.5
        
        return score
    
    def _contains_definition_pattern(self, sentence: str) -> bool:
        """Check if sentence contains definition patterns."""
        sentence_lower = sentence.lower()
        
        definition_indicators = [
            'is a', 'is an', 'is the', 'defined as', 'refers to', 'means',
            'consists of', 'composed of', 'characterized by'
        ]
        
        return any(indicator in sentence_lower for indicator in definition_indicators)
    
    def _extract_definition_for_keyword(self, text: str, keyword: str) -> str:
        """Extract definition for a specific keyword."""
        if not text or not keyword:
            return ""
        
        # Find sentences containing the keyword
        sentences = sent_tokenize(text)
        relevant_sentences = []
        
        for sentence in sentences:
            if keyword.lower() in sentence.lower():
                relevant_sentences.append(sentence)
        
        if not relevant_sentences:
            return ""
        
        # Look for definition patterns
        for sentence in relevant_sentences:
            definition = self._extract_definition_from_sentence(sentence, keyword)
            if definition:
                return definition
        
        # If no definition pattern found, return the first relevant sentence
        return relevant_sentences[0]
    
    def _extract_definition_from_sentence(self, sentence: str, keyword: str) -> str:
        """Extract definition from a sentence using patterns."""
        sentence_lower = sentence.lower()
        keyword_lower = keyword.lower()
        
        # Look for definition patterns
        for pattern_type, patterns in self.answer_patterns.items():
            for pattern in patterns:
                # Create pattern that includes the keyword
                full_pattern = rf"{keyword_lower}\s+{pattern}"
                match = re.search(full_pattern, sentence_lower)
                
                if match:
                    # Extract the definition part
                    start = match.start()
                    end = sentence.find('.', start)
                    if end == -1:
                        end = len(sentence)
                    
                    definition = sentence[start:end].strip()
                    return definition
        
        return ""
    
    def generate_summary_answer(self, question: str, relevant_text: str, max_length: int = 150) -> str:
        """
        Generate a summary answer for a question.
        
        Args:
            question: The question
            relevant_text: Relevant text to summarize
            max_length: Maximum length of the answer
            
        Returns:
            Summarized answer
        """
        if not question or not relevant_text:
            return ""
        
        # Extract key information based on question type
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['what is', 'define', 'meaning']):
            # Look for definition
            return self._extract_definition_answer(relevant_text, max_length)
        elif any(word in question_lower for word in ['how', 'process', 'steps']):
            # Look for process/steps
            return self._extract_process_answer(relevant_text, max_length)
        elif any(word in question_lower for word in ['why', 'purpose', 'function']):
            # Look for purpose/function
            return self._extract_purpose_answer(relevant_text, max_length)
        else:
            # Generic summary
            return self._extract_generic_answer(relevant_text, max_length)
    
    def _extract_definition_answer(self, text: str, max_length: int) -> str:
        """Extract definition-style answer."""
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            if self._contains_definition_pattern(sentence):
                if len(sentence) <= max_length:
                    return sentence
                else:
                    return sentence[:max_length] + "..."
        
        # Fallback to first sentence
        if sentences:
            return sentences[0][:max_length] + "..." if len(sentences[0]) > max_length else sentences[0]
        
        return ""
    
    def _extract_process_answer(self, text: str, max_length: int) -> str:
        """Extract process/steps answer."""
        # Look for numbered or bulleted lists
        lines = text.split('\n')
        
        for line in lines:
            if re.match(r'^\d+\.', line) or re.match(r'^[-*]', line):
                if len(line) <= max_length:
                    return line
                else:
                    return line[:max_length] + "..."
        
        # Fallback to definition extraction
        return self._extract_definition_answer(text, max_length)
    
    def _extract_purpose_answer(self, text: str, max_length: int) -> str:
        """Extract purpose/function answer."""
        sentences = sent_tokenize(text)
        
        for sentence in sentences:
            sentence_lower = sentence.lower()
            if any(word in sentence_lower for word in ['purpose', 'function', 'used to', 'responsible for']):
                if len(sentence) <= max_length:
                    return sentence
                else:
                    return sentence[:max_length] + "..."
        
        # Fallback to definition extraction
        return self._extract_definition_answer(text, max_length)
    
    def _extract_generic_answer(self, text: str, max_length: int) -> str:
        """Extract generic answer."""
        sentences = sent_tokenize(text)
        
        if sentences:
            first_sentence = sentences[0]
            if len(first_sentence) <= max_length:
                return first_sentence
            else:
                return first_sentence[:max_length] + "..."
        
        return ""
