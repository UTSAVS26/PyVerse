#!/usr/bin/env python3
"""
Debug script for AnswerSelector
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flashgenie.nlp.answer_selector import AnswerSelector

def debug_answer_selector():
    """Debug the AnswerSelector functionality."""
    
    # Sample text and question from the test
    sample_text = """
    The mitochondria is the powerhouse of the cell. 
    It generates ATP through cellular respiration.
    DNA is composed of nucleotides containing bases, phosphate, and a sugar.
    Photosynthesis is the process by which plants convert sunlight into energy.
    """
    
    question = "What is mitochondria?"
    
    print("Sample text:")
    print(sample_text)
    print("\nQuestion:", question)
    
    # Create selector
    selector = AnswerSelector(use_spacy=False)
    
    # Debug key term extraction
    key_terms = selector._extract_key_terms_from_question(question)
    print(f"\nExtracted key terms: {key_terms}")
    
    # Debug the extraction process
    question_lower = question.lower()
    question_words = ['what', 'how', 'why', 'when', 'where', 'which', 'who', 'is', 'are', 'does', 'do', 'the', 'a', 'an']
    
    for word in question_words:
        question_lower = question_lower.replace(word, '')
    
    print(f"\nQuestion after removing question words: '{question_lower}'")
    
    words = question.split()
    print(f"\nWords in question: {words}")
    
    # Debug sentence finding
    from nltk.tokenize import sent_tokenize
    sentences = sent_tokenize(sample_text)
    print(f"\nAll sentences: {sentences}")
    
    relevant_sentences = selector._find_relevant_sentences(sample_text, key_terms)
    print(f"\nRelevant sentences: {relevant_sentences}")
    
    # Debug answer selection
    if relevant_sentences:
        best_answer = selector._select_best_answer(question, relevant_sentences, max_length=200)
        print(f"\nBest answer: '{best_answer}'")
    else:
        print("\nNo relevant sentences found!")
    
    # Test the full pipeline
    result = selector.find_answer_for_question(question, sample_text)
    print(f"\nFinal result: '{result}'")

if __name__ == "__main__":
    debug_answer_selector()
