"""
Question Generation Module

Generates questions from text using transformers and rule-based approaches.
"""

import re
import random
from typing import List, Dict, Any, Optional, Tuple
import logging
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch

logger = logging.getLogger(__name__)


class QuestionGenerator:
    """Generates questions from text using multiple approaches."""
    
    def __init__(self, use_transformers: bool = True, model_name: str = "t5-base"):
        """
        Initialize the question generator.
        
        Args:
            use_transformers: Whether to use transformer models
            model_name: Name of the transformer model to use
        """
        self.use_transformers = use_transformers
        self.model_name = model_name
        
        # Initialize transformer model if requested
        if self.use_transformers:
            try:
                self._initialize_transformers()
            except Exception as e:
                logger.warning(f"Failed to initialize transformers: {e}")
                self.use_transformers = False
        
        # Question templates for rule-based generation
        self.question_templates = {
            'what_is': [
                "What is {concept}?",
                "Define {concept}.",
                "What does {concept} mean?",
                "Explain {concept}."
            ],
            'how_does': [
                "How does {concept} work?",
                "How is {concept} implemented?",
                "What is the process of {concept}?",
                "How do you {concept}?"
            ],
            'why_does': [
                "Why does {concept} happen?",
                "What causes {concept}?",
                "Why is {concept} important?",
                "What is the purpose of {concept}?"
            ],
            'fill_blank': [
                "{concept} is composed of ____.",
                "The main function of {concept} is ____.",
                "{concept} occurs when ____.",
                "____ is essential for {concept}."
            ]
        }
        
        # Common question starters
        self.question_starters = [
            "What is", "How does", "Why does", "When does", "Where does",
            "Which", "Who", "What are", "How are", "Why are"
        ]
    
    def _initialize_transformers(self):
        """Initialize transformer models for question generation."""
        try:
            # Use a smaller model for faster processing
            model_name = "google/flan-t5-small"  # Smaller alternative
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            # Create question generation pipeline
            self.qg_pipeline = pipeline(
                "text2text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=64,
                do_sample=True,
                temperature=0.7
            )
            
            logger.info(f"Initialized transformer model: {model_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize transformers: {e}")
            self.use_transformers = False
    
    def generate_questions_transformers(self, text: str, num_questions: int = 5) -> List[str]:
        """
        Generate questions using transformer models.
        
        Args:
            text: Input text
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        if not self.use_transformers or not text:
            return []
        
        try:
            # Prepare input for question generation
            input_text = f"Generate a question: {text[:500]}"  # Limit input length
            
            questions = []
            for _ in range(num_questions):
                try:
                    result = self.qg_pipeline(input_text, max_length=64, do_sample=True)
                    question = result[0]['generated_text'].strip()
                    
                    # Clean and validate question
                    if self._is_valid_question(question):
                        questions.append(question)
                        
                except Exception as e:
                    logger.warning(f"Failed to generate question: {e}")
                    continue
            
            return questions[:num_questions]
            
        except Exception as e:
            logger.error(f"Transformer question generation failed: {e}")
            return []
    
    def generate_questions_rule_based(self, text: str, keywords: List[str], num_questions: int = 5) -> List[str]:
        """
        Generate questions using rule-based templates.
        
        Args:
            text: Input text
            keywords: List of keywords to use in questions
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        if not text or not keywords:
            return []
        
        questions = []
        used_keywords = set()
        
        # Generate different types of questions
        question_types = list(self.question_templates.keys())
        
        for _ in range(num_questions):
            # Select a random question type
            q_type = random.choice(question_types)
            template = random.choice(self.question_templates[q_type])
            
            # Find an unused keyword
            available_keywords = [k for k in keywords if k not in used_keywords]
            if not available_keywords:
                available_keywords = keywords  # Reuse if all used
            
            keyword = random.choice(available_keywords)
            used_keywords.add(keyword)
            
            # Generate question
            question = template.format(concept=keyword)
            questions.append(question)
        
        return questions
    
    def generate_fill_blank_questions(self, text: str, keywords: List[str], num_questions: int = 3) -> List[Tuple[str, str]]:
        """
        Generate fill-in-the-blank questions.
        
        Args:
            text: Input text
            keywords: List of keywords to use
            num_questions: Number of questions to generate
            
        Returns:
            List of (question, answer) tuples
        """
        if not text or not keywords:
            return []
        
        fill_blank_questions = []
        
        # Find sentences containing keywords
        sentences = self._extract_sentences_with_keywords(text, keywords)
        
        for sentence in sentences[:num_questions]:
            # Find a keyword in the sentence
            for keyword in keywords:
                if keyword.lower() in sentence.lower():
                    # Create fill-in-the-blank question
                    question = sentence.replace(keyword, "____")
                    answer = keyword
                    
                    fill_blank_questions.append((question, answer))
                    break
        
        return fill_blank_questions
    
    def generate_questions_from_sentences(self, sentences: List[str], num_questions: int = 5) -> List[str]:
        """
        Generate questions from a list of sentences.
        
        Args:
            sentences: List of sentences
            num_questions: Number of questions to generate
            
        Returns:
            List of generated questions
        """
        if not sentences:
            return []
        
        questions = []
        
        for sentence in sentences[:num_questions]:
            # Extract key concepts from sentence
            concepts = self._extract_concepts_from_sentence(sentence)
            
            if concepts:
                concept = random.choice(concepts)
                question_type = random.choice(['what_is', 'how_does', 'why_does'])
                template = random.choice(self.question_templates[question_type])
                
                question = template.format(concept=concept)
                questions.append(question)
        
        return questions
    
    def _extract_sentences_with_keywords(self, text: str, keywords: List[str]) -> List[str]:
        """Extract sentences that contain any of the keywords."""
        import nltk
        from nltk.tokenize import sent_tokenize
        
        sentences = sent_tokenize(text)
        relevant_sentences = []
        
        for sentence in sentences:
            for keyword in keywords:
                if keyword.lower() in sentence.lower():
                    relevant_sentences.append(sentence)
                    break
        
        return relevant_sentences
    
    def _extract_concepts_from_sentence(self, sentence: str) -> List[str]:
        """Extract key concepts from a sentence."""
        # Simple approach: extract noun phrases
        words = sentence.split()
        concepts = []
        
        for i, word in enumerate(words):
            # Look for capitalized words or phrases
            if word[0].isupper() and len(word) > 2:
                # Check if it's part of a phrase
                phrase = word
                j = i + 1
                while j < len(words) and words[j][0].isupper():
                    phrase += " " + words[j]
                    j += 1
                
                if len(phrase.split()) <= 3:  # Limit phrase length
                    concepts.append(phrase)
        
        return concepts
    
    def _is_valid_question(self, question: str) -> bool:
        """Check if a generated question is valid."""
        if not question or len(question) < 10:
            return False
        
        # Check if it starts with a question word
        question_lower = question.lower()
        starts_with_question_word = any(question_lower.startswith(starter.lower()) 
                                      for starter in self.question_starters)
        
        # Check if it ends with question mark
        ends_with_question_mark = question.strip().endswith('?')
        
        # Check if it contains common question indicators
        has_question_indicators = any(word in question_lower for word in 
                                    ['what', 'how', 'why', 'when', 'where', 'which', 'who'])
        
        return starts_with_question_word or ends_with_question_mark or has_question_indicators
    
    def generate_questions_combined(self, text: str, keywords: List[str], num_questions: int = 10) -> List[str]:
        """
        Generate questions using multiple approaches.
        
        Args:
            text: Input text
            keywords: List of keywords
            num_questions: Total number of questions to generate
            
        Returns:
            List of generated questions
        """
        all_questions = []
        
        # Generate transformer-based questions
        if self.use_transformers:
            transformer_questions = self.generate_questions_transformers(text, num_questions // 2)
            all_questions.extend(transformer_questions)
        
        # Generate rule-based questions
        rule_questions = self.generate_questions_rule_based(text, keywords, num_questions // 2)
        all_questions.extend(rule_questions)
        
        # Remove duplicates and return
        unique_questions = list(set(all_questions))
        return unique_questions[:num_questions]
