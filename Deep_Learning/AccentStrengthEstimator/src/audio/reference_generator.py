"""
Reference audio generation for native English pronunciation.
"""

import os
import numpy as np
from typing import List, Dict
import json


class ReferenceGenerator:
    """Generates reference audio and phoneme data for native English pronunciation."""
    
    def __init__(self, sample_rate: int = 22050):
        self.sample_rate = sample_rate
        self.reference_data = {}
        
    def load_reference_phrases(self, file_path: str) -> List[str]:
        """Load reference phrases from a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                phrases = [line.strip() for line in f if line.strip()]
            return phrases
        except Exception as e:
            print(f"Error loading reference phrases: {e}")
            return []
    
    def generate_phonemes(self, text: str) -> List[str]:
        """Generate phoneme sequence from text."""
        # Simplified phoneme generation
        phoneme_map = {
            'th': 'θ', 'ch': 'tʃ', 'sh': 'ʃ', 'ng': 'ŋ',
            'a': 'æ', 'e': 'ɛ', 'i': 'ɪ', 'o': 'oʊ', 'u': 'ʌ'
        }
        
        phonemes = []
        text_lower = text.lower()
        i = 0
        while i < len(text_lower):
            if i + 1 < len(text_lower):
                digraph = text_lower[i:i+2]
                if digraph in phoneme_map:
                    phonemes.append(phoneme_map[digraph])
                    i += 2
                    continue
            
            char = text_lower[i]
            if char in phoneme_map:
                phonemes.append(phoneme_map[char])
            elif char.isalpha():
                phonemes.append(char)
            i += 1
        
        return phonemes
    
    def create_reference_data(self, phrases: List[str]) -> Dict[str, Dict]:
        """Create reference data for all phrases."""
        reference_data = {}
        
        for i, phrase in enumerate(phrases):
            try:
                phonemes = self.generate_phonemes(phrase)
                
                reference_data[f"phrase_{i+1}"] = {
                    'text': phrase,
                    'phonemes': phonemes,
                    'phoneme_count': len(phonemes),
                    'word_count': len(phrase.split()),
                    'difficulty_level': self._assess_difficulty(phrase, phonemes)
                }
                
            except Exception as e:
                print(f"Error processing phrase {i+1}: {e}")
                continue
        
        self.reference_data = reference_data
        return reference_data
    
    def _assess_difficulty(self, text: str, phonemes: List[str]) -> str:
        """Assess the difficulty level of a phrase."""
        difficult_phonemes = ['θ', 'ð', 'ʃ', 'ʒ', 'ŋ', 'r']
        difficult_count = sum(1 for p in phonemes if p in difficult_phonemes)
        word_count = len(text.split())
        
        if word_count <= 3 and difficult_count <= 1:
            return 'easy'
        elif word_count <= 6 and difficult_count <= 3:
            return 'medium'
        else:
            return 'hard'
    
    def save_reference_data(self, file_path: str) -> bool:
        """Save reference data to a JSON file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(self.reference_data, f, indent=2, ensure_ascii=False)
            return True
        except Exception as e:
            print(f"Error saving reference data: {e}")
            return False
    
    def load_reference_data(self, file_path: str) -> bool:
        """Load reference data from a JSON file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                self.reference_data = json.load(f)
            return True
        except Exception as e:
            print(f"Error loading reference data: {e}")
            return False
    
    def get_phrase_data(self, phrase_id: str) -> Dict:
        """Get data for a specific phrase."""
        return self.reference_data.get(phrase_id, {})
    
    def get_phrases_by_difficulty(self, difficulty: str) -> List[str]:
        """Get phrases by difficulty level."""
        phrases = []
        for phrase_id, data in self.reference_data.items():
            if data.get('difficulty_level') == difficulty:
                phrases.append(phrase_id)
        return phrases
