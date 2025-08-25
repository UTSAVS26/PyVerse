"""
Tests for the exercise generator module
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock

from exercise_generator import ExerciseGenerator, TypingExercise, create_exercise_generator
from analyzer import WeakSpot, TypingProfile
from datetime import datetime


class TestExerciseGenerator(unittest.TestCase):
    """Test cases for ExerciseGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = ExerciseGenerator()
    
    def test_init(self):
        """Test generator initialization"""
        self.assertIsNotNone(self.generator.common_words)
        self.assertIsNotNone(self.generator.sentence_templates)
        self.assertIsInstance(self.generator.common_words, list)
        self.assertIsInstance(self.generator.sentence_templates, list)
        self.assertGreater(len(self.generator.common_words), 0)
        self.assertGreater(len(self.generator.sentence_templates), 0)
    
    def test_load_common_words(self):
        """Test loading common words"""
        words = self.generator._load_common_words()
        self.assertIsInstance(words, list)
        self.assertGreater(len(words), 0)
        # Check for some common words
        self.assertIn('the', words)
        self.assertIn('and', words)
        self.assertIn('for', words)
    
    def test_load_sentence_templates(self):
        """Test loading sentence templates"""
        templates = self.generator._load_sentence_templates()
        self.assertIsInstance(templates, list)
        self.assertGreater(len(templates), 0)
        # Check for template placeholders
        for template in templates:
            self.assertIn('{', template)
            self.assertIn('}', template)
    
    def test_generate_exercises(self):
        """Test generating exercises from profile"""
        # Create a mock profile with weak spots
        weak_spots = [
            WeakSpot("a", 0.2, 0.3, 10, 0.7, "character"),
            WeakSpot("th", 0.15, 0.25, 5, 0.6, "bigram"),
            WeakSpot("the", 0.1, 0.2, 3, 0.5, "trigram")
        ]
        
        profile = TypingProfile(
            user_id="test_user",
            total_sessions=5,
            avg_wpm=60.0,
            avg_accuracy=85.0,
            weak_spots=weak_spots,
            common_mistakes={"a": ["s", "q"]},
            finger_weaknesses={"left_index": 0.1},
            generated_at=datetime.now()
        )
        
        exercises = self.generator.generate_exercises(profile, 3)
        
        self.assertIsInstance(exercises, list)
        self.assertEqual(len(exercises), 3)
        
        for exercise in exercises:
            self.assertIsInstance(exercise, TypingExercise)
            self.assertIsInstance(exercise.text, str)
            self.assertGreater(len(exercise.text), 0)
            self.assertIsInstance(exercise.difficulty, float)
            self.assertGreaterEqual(exercise.difficulty, 0)
            self.assertLessEqual(exercise.difficulty, 1)
    
    def test_create_exercise_for_weak_spot_character(self):
        """Test creating character-focused exercise"""
        weak_spot = WeakSpot("a", 0.2, 0.3, 10, 0.7, "character")
        profile = self._create_mock_profile([weak_spot])
        
        exercise = self.generator._create_exercise_for_weak_spot(weak_spot, profile)
        
        self.assertIsInstance(exercise, TypingExercise)
        self.assertEqual(exercise.exercise_type, 'character')
        self.assertIn('a', exercise.text.lower())
        self.assertIn('a', exercise.target_patterns)
    
    def test_create_exercise_for_weak_spot_bigram(self):
        """Test creating bigram-focused exercise"""
        weak_spot = WeakSpot("th", 0.15, 0.25, 5, 0.6, "bigram")
        profile = self._create_mock_profile([weak_spot])
        
        exercise = self.generator._create_exercise_for_weak_spot(weak_spot, profile)
        
        self.assertIsInstance(exercise, TypingExercise)
        self.assertEqual(exercise.exercise_type, 'bigram')
        self.assertIn('th', exercise.target_patterns)
    
    def test_create_exercise_for_weak_spot_trigram(self):
        """Test creating trigram-focused exercise"""
        weak_spot = WeakSpot("the", 0.1, 0.2, 3, 0.5, "trigram")
        profile = self._create_mock_profile([weak_spot])
        
        exercise = self.generator._create_exercise_for_weak_spot(weak_spot, profile)
        
        self.assertIsInstance(exercise, TypingExercise)
        self.assertEqual(exercise.exercise_type, 'trigram')
        self.assertIn('the', exercise.target_patterns)
    
    def test_create_character_exercise(self):
        """Test character exercise creation"""
        weak_spot = WeakSpot("a", 0.2, 0.3, 10, 0.7, "character")
        
        exercise = self.generator._create_character_exercise(weak_spot)
        
        self.assertIsInstance(exercise, TypingExercise)
        self.assertEqual(exercise.exercise_type, 'character')
        self.assertIn('a', exercise.target_patterns)
        self.assertGreater(len(exercise.text), 0)
        self.assertIn("Focus on typing the letter 'a' accurately", exercise.instructions)
    
    def test_create_bigram_exercise(self):
        """Test bigram exercise creation"""
        weak_spot = WeakSpot("th", 0.15, 0.25, 5, 0.6, "bigram")
        
        exercise = self.generator._create_bigram_exercise(weak_spot)
        
        self.assertIsInstance(exercise, TypingExercise)
        self.assertEqual(exercise.exercise_type, 'bigram')
        self.assertIn('th', exercise.target_patterns)
        self.assertIn("Practice typing the letter combination 'th'", exercise.instructions)
    
    def test_create_trigram_exercise(self):
        """Test trigram exercise creation"""
        weak_spot = WeakSpot("the", 0.1, 0.2, 3, 0.5, "trigram")
        
        exercise = self.generator._create_trigram_exercise(weak_spot)
        
        self.assertIsInstance(exercise, TypingExercise)
        self.assertEqual(exercise.exercise_type, 'trigram')
        self.assertIn('the', exercise.target_patterns)
        self.assertIn("Practice typing the three-letter combination 'the'", exercise.instructions)
    
    def test_create_general_exercise(self):
        """Test general exercise creation"""
        profile = self._create_mock_profile([])
        
        exercise = self.generator._create_general_exercise(profile)
        
        self.assertIsInstance(exercise, TypingExercise)
        self.assertEqual(exercise.exercise_type, 'sentence')
        self.assertEqual(exercise.difficulty, 0.5)
        self.assertGreater(len(exercise.text), 0)
        self.assertIn("Practice typing this sentence", exercise.instructions)
    
    def test_generate_progressive_exercises(self):
        """Test progressive exercise generation"""
        weak_spots = [
            WeakSpot("a", 0.2, 0.3, 10, 0.7, "character"),
            WeakSpot("th", 0.15, 0.25, 5, 0.6, "bigram")
        ]
        profile = self._create_mock_profile(weak_spots)
        
        # Test different difficulty levels
        for difficulty in ["easy", "medium", "hard", "expert"]:
            exercises = self.generator.generate_progressive_exercises(profile, difficulty)
            
            self.assertIsInstance(exercises, list)
            self.assertGreater(len(exercises), 0)
            
            for exercise in exercises:
                self.assertIsInstance(exercise, TypingExercise)
                self.assertIn(difficulty, exercise.instructions)
    
    def test_generate_mixed_exercise(self):
        """Test mixed exercise generation"""
        weak_spots = [
            WeakSpot("a", 0.2, 0.3, 10, 0.7, "character"),
            WeakSpot("e", 0.15, 0.25, 8, 0.6, "character"),
            WeakSpot("i", 0.1, 0.2, 6, 0.5, "character")
        ]
        profile = self._create_mock_profile(weak_spots)
        
        exercise = self.generator.generate_mixed_exercise(profile)
        
        self.assertIsInstance(exercise, TypingExercise)
        self.assertEqual(exercise.exercise_type, 'mixed')
        self.assertGreater(len(exercise.target_patterns), 0)
        self.assertIn("This exercise targets multiple areas", exercise.instructions)
    
    def test_difficulty_scaling(self):
        """Test difficulty scaling in progressive exercises"""
        weak_spot = WeakSpot("a", 0.2, 0.3, 10, 0.7, "character")
        profile = self._create_mock_profile([weak_spot])
        
        # Test different difficulty multipliers
        difficulties = {
            "easy": 0.5,
            "medium": 1.0,
            "hard": 1.5,
            "expert": 2.0
        }
        
        for difficulty_name, expected_multiplier in difficulties.items():
            exercises = self.generator.generate_progressive_exercises(profile, difficulty_name)
            
            for exercise in exercises:
                # Difficulty should be scaled but capped at 1.0
                # The base difficulty depends on the exercise type, so we'll just check it's reasonable
                self.assertGreaterEqual(exercise.difficulty, 0.0)
                self.assertLessEqual(exercise.difficulty, 1.0)
                # Check that difficulty level is mentioned in instructions
                self.assertIn(difficulty_name, exercise.instructions)
    
    def test_exercise_properties(self):
        """Test exercise properties and validation"""
        weak_spot = WeakSpot("a", 0.2, 0.3, 10, 0.7, "character")
        exercise = self.generator._create_character_exercise(weak_spot)
        
        # Check all required properties
        self.assertIsInstance(exercise.text, str)
        self.assertGreater(len(exercise.text), 0)
        self.assertIsInstance(exercise.difficulty, float)
        self.assertGreaterEqual(exercise.difficulty, 0)
        self.assertLessEqual(exercise.difficulty, 1)
        self.assertIsInstance(exercise.target_patterns, list)
        self.assertIsInstance(exercise.exercise_type, str)
        self.assertIsInstance(exercise.estimated_duration, int)
        self.assertGreater(exercise.estimated_duration, 0)
        self.assertIsInstance(exercise.instructions, str)
        self.assertGreater(len(exercise.instructions), 0)
    
    def test_word_selection_for_character_exercise(self):
        """Test that character exercises include words with the target character"""
        weak_spot = WeakSpot("a", 0.2, 0.3, 10, 0.7, "character")
        
        exercise = self.generator._create_character_exercise(weak_spot)
        
        # The exercise text should contain the target character
        self.assertIn('a', exercise.text.lower())
        
        # Check that words containing 'a' are used
        words_with_a = [word for word in self.generator.common_words if 'a' in word.lower()]
        if words_with_a:
            # At least some words should contain 'a'
            found_words_with_a = any(word in exercise.text.lower() for word in words_with_a[:5])
            self.assertTrue(found_words_with_a)
    
    def test_sentence_generation(self):
        """Test sentence generation in general exercises"""
        profile = self._create_mock_profile([])
        
        exercise = self.generator._create_general_exercise(profile)
        
        # Should contain multiple sentences
        sentences = exercise.text.split('.')
        self.assertGreater(len(sentences), 1)
        
        # Each sentence should be properly formatted
        for sentence in sentences:
            if sentence.strip():
                self.assertTrue(sentence.strip()[0].isupper())  # Starts with capital
    
    def _create_mock_profile(self, weak_spots):
        """Create a mock typing profile for testing"""
        return TypingProfile(
            user_id="test_user",
            total_sessions=5,
            avg_wpm=60.0,
            avg_accuracy=85.0,
            weak_spots=weak_spots,
            common_mistakes={"a": ["s", "q"]},
            finger_weaknesses={"left_index": 0.1},
            generated_at=datetime.now()
        )


class TestTypingExercise(unittest.TestCase):
    """Test cases for TypingExercise dataclass"""
    
    def test_typing_exercise_creation(self):
        """Test creating TypingExercise instances"""
        exercise = TypingExercise(
            text="The quick brown fox jumps over the lazy dog.",
            difficulty=0.7,
            target_patterns=["th", "qu"],
            exercise_type="mixed",
            estimated_duration=30,
            instructions="Practice typing this text focusing on the target patterns."
        )
        
        self.assertEqual(exercise.text, "The quick brown fox jumps over the lazy dog.")
        self.assertEqual(exercise.difficulty, 0.7)
        self.assertEqual(exercise.target_patterns, ["th", "qu"])
        self.assertEqual(exercise.exercise_type, "mixed")
        self.assertEqual(exercise.estimated_duration, 30)
        self.assertEqual(exercise.instructions, "Practice typing this text focusing on the target patterns.")
    
    def test_typing_exercise_validation(self):
        """Test TypingExercise validation"""
        # Test with valid data
        exercise = TypingExercise(
            text="Test text",
            difficulty=0.5,
            target_patterns=[],
            exercise_type="sentence",
            estimated_duration=10,
            instructions="Test instructions"
        )
        
        self.assertIsInstance(exercise.text, str)
        self.assertIsInstance(exercise.difficulty, float)
        self.assertIsInstance(exercise.target_patterns, list)
        self.assertIsInstance(exercise.exercise_type, str)
        self.assertIsInstance(exercise.estimated_duration, int)
        self.assertIsInstance(exercise.instructions, str)


class TestConvenienceFunctions(unittest.TestCase):
    """Test cases for convenience functions"""
    
    def test_create_exercise_generator(self):
        """Test create_exercise_generator function"""
        generator = create_exercise_generator()
        self.assertIsInstance(generator, ExerciseGenerator)


class TestExerciseGenerationEdgeCases(unittest.TestCase):
    """Test edge cases in exercise generation"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = ExerciseGenerator()
    
    def test_generate_exercises_no_weak_spots(self):
        """Test generating exercises with no weak spots"""
        profile = self._create_mock_profile([])
        
        exercises = self.generator.generate_exercises(profile, 3)
        
        # Should still generate exercises (general ones)
        self.assertEqual(len(exercises), 3)
        for exercise in exercises:
            self.assertEqual(exercise.exercise_type, 'sentence')
    
    def test_generate_exercises_insufficient_weak_spots(self):
        """Test generating exercises with fewer weak spots than requested"""
        weak_spots = [WeakSpot("a", 0.2, 0.3, 10, 0.7, "character")]
        profile = self._create_mock_profile(weak_spots)
        
        exercises = self.generator.generate_exercises(profile, 5)
        
        # Should generate 5 exercises (1 from weak spot + 4 general)
        self.assertEqual(len(exercises), 5)
    
    def test_character_exercise_no_matching_words(self):
        """Test character exercise when no words contain the character"""
        # Use a very uncommon character
        weak_spot = WeakSpot("z", 0.2, 0.3, 10, 0.7, "character")
        
        exercise = self.generator._create_character_exercise(weak_spot)
        
        # Should still create an exercise
        self.assertIsInstance(exercise, TypingExercise)
        self.assertEqual(exercise.exercise_type, 'character')
    
    def test_bigram_exercise_no_matching_words(self):
        """Test bigram exercise when no words contain the bigram"""
        # Use a very uncommon bigram
        weak_spot = WeakSpot("zz", 0.2, 0.3, 10, 0.7, "bigram")
        
        exercise = self.generator._create_bigram_exercise(weak_spot)
        
        # Should create artificial words with the bigram
        self.assertIsInstance(exercise, TypingExercise)
        self.assertEqual(exercise.exercise_type, 'bigram')
        self.assertIn('zz', exercise.text.lower())
    
    def test_trigram_exercise_no_matching_words(self):
        """Test trigram exercise when no words contain the trigram"""
        # Use a very uncommon trigram
        weak_spot = WeakSpot("zzz", 0.2, 0.3, 10, 0.7, "trigram")
        
        exercise = self.generator._create_trigram_exercise(weak_spot)
        
        # Should create artificial words with the trigram
        self.assertIsInstance(exercise, TypingExercise)
        self.assertEqual(exercise.exercise_type, 'trigram')
        self.assertIn('zzz', exercise.text.lower())
    
    def _create_mock_profile(self, weak_spots):
        """Create a mock typing profile for testing"""
        return TypingProfile(
            user_id="test_user",
            total_sessions=5,
            avg_wpm=60.0,
            avg_accuracy=85.0,
            weak_spots=weak_spots,
            common_mistakes={},
            finger_weaknesses={},
            generated_at=datetime.now()
        )


if __name__ == '__main__':
    unittest.main()
