"""
KeyMentor - Exercise Generator
Builds personalized typing drills targeting identified weak spots.
"""

import random
import string
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from analyzer import WeakSpot, TypingProfile


@dataclass
class TypingExercise:
    """Represents a typing exercise"""
    text: str
    difficulty: float  # 0.0 to 1.0
    target_patterns: List[str]
    exercise_type: str  # 'character', 'bigram', 'trigram', 'word', 'sentence'
    estimated_duration: int  # seconds
    instructions: str


class ExerciseGenerator:
    """Generates personalized typing exercises based on weak spots"""
    
    def __init__(self):
        self.common_words = self._load_common_words()
        self.sentence_templates = self._load_sentence_templates()
    
    def _load_common_words(self) -> List[str]:
        """Load common English words for exercise generation"""
        return [
            "the", "be", "to", "of", "and", "a", "in", "that", "have", "i", "it", "for", "not",
            "on", "with", "he", "as", "you", "do", "at", "this", "but", "his", "by", "from",
            "they", "we", "say", "her", "she", "or", "an", "will", "my", "one", "all", "would",
            "there", "their", "what", "so", "up", "out", "if", "about", "who", "get", "which",
            "go", "me", "when", "make", "can", "like", "time", "no", "just", "him", "know",
            "take", "people", "into", "year", "your", "good", "some", "could", "them", "see",
            "other", "than", "then", "now", "look", "only", "come", "its", "over", "think",
            "also", "back", "after", "use", "two", "how", "our", "work", "first", "well",
            "way", "even", "new", "want", "because", "any", "these", "give", "day", "most",
            "us", "very", "here", "just", "under", "into", "through", "during", "before",
            "between", "after", "above", "below", "from", "up", "down", "in", "out", "on",
            "at", "by", "for", "with", "against", "among", "throughout", "despite", "towards",
            "upon", "within", "without", "against", "among", "between", "beyond", "during",
            "except", "including", "like", "near", "off", "over", "past", "since", "through",
            "toward", "under", "until", "up", "upon", "with", "within", "without"
        ]
    
    def _load_sentence_templates(self) -> List[str]:
        """Load sentence templates for generating exercises"""
        return [
            "The {adj} {noun} {verb} {adv}.",
            "{Noun} {verb} {adv} in the {adj} {noun}.",
            "When {noun} {verb}, the {adj} {noun} {verb} {adv}.",
            "The {adj} {noun} {verb} {adv} because {noun} {verb}.",
            "{Noun} and {noun} {verb} {adv} through the {adj} {noun}.",
            "In the {adj} {noun}, {noun} {verb} {adv}.",
            "The {adj} {noun} {verb} {adv} while {noun} {verb}.",
            "{Noun} {verb} {adv} when the {adj} {noun} {verb}.",
            "Through {adj} {noun}, {noun} {verb} {adv}.",
            "The {adj} {noun} {verb} {adv} near the {adj} {noun}."
        ]
    
    def generate_exercises(self, profile: TypingProfile, 
                          num_exercises: int = 5) -> List[TypingExercise]:
        """Generate personalized exercises based on user's weak spots"""
        exercises = []
        
        # Generate exercises for top weak spots
        for weak_spot in profile.weak_spots[:num_exercises]:
            exercise = self._create_exercise_for_weak_spot(weak_spot, profile)
            if exercise:
                exercises.append(exercise)
        
        # If we don't have enough weak spots, generate general exercises
        while len(exercises) < num_exercises:
            exercise = self._create_general_exercise(profile)
            exercises.append(exercise)
        
        return exercises[:num_exercises]
    
    def _create_exercise_for_weak_spot(self, weak_spot: WeakSpot, 
                                     profile: TypingProfile) -> Optional[TypingExercise]:
        """Create a specific exercise targeting a weak spot"""
        
        if weak_spot.pattern_type == 'character':
            return self._create_character_exercise(weak_spot)
        elif weak_spot.pattern_type == 'bigram':
            return self._create_bigram_exercise(weak_spot)
        elif weak_spot.pattern_type == 'trigram':
            return self._create_trigram_exercise(weak_spot)
        else:
            return self._create_general_exercise(profile)
    
    def _create_character_exercise(self, weak_spot: WeakSpot) -> TypingExercise:
        """Create exercise focusing on a specific character"""
        char = weak_spot.pattern
        difficulty = min(weak_spot.difficulty_score * 2, 1.0)
        
        # Generate text with high frequency of the target character
        words = [word for word in self.common_words if char in word.lower()]
        if not words:
            words = self.common_words
        
        # Create sentences with high frequency of the character
        sentences = []
        for _ in range(3):
            sentence_words = random.sample(words, min(5, len(words)))
            sentence = " ".join(sentence_words).capitalize() + "."
            sentences.append(sentence)
        
        text = " ".join(sentences)
        
        return TypingExercise(
            text=text,
            difficulty=difficulty,
            target_patterns=[char],
            exercise_type='character',
            estimated_duration=int(len(text) / 5),  # Rough estimate
            instructions=f"Focus on typing the letter '{char}' accurately. This character has been identified as one of your weak spots."
        )
    
    def _create_bigram_exercise(self, weak_spot: WeakSpot) -> TypingExercise:
        """Create exercise focusing on a specific bigram"""
        bigram = weak_spot.pattern
        difficulty = min(weak_spot.difficulty_score * 1.5, 1.0)
        
        # Find words containing the bigram
        words_with_bigram = [word for word in self.common_words if bigram in word.lower()]
        
        if not words_with_bigram:
            # Create artificial words with the bigram
            words_with_bigram = [f"test{bigram}test", f"{bigram}word", f"word{bigram}"]
        
        # Create sentences with these words
        sentences = []
        for _ in range(3):
            sentence_words = random.sample(words_with_bigram, min(4, len(words_with_bigram)))
            # Add some common words to make it more natural
            sentence_words.extend(random.sample(self.common_words, 3))
            random.shuffle(sentence_words)
            sentence = " ".join(sentence_words).capitalize() + "."
            sentences.append(sentence)
        
        text = " ".join(sentences)
        
        return TypingExercise(
            text=text,
            difficulty=difficulty,
            target_patterns=[bigram],
            exercise_type='bigram',
            estimated_duration=int(len(text) / 5),
            instructions=f"Practice typing the letter combination '{bigram}' smoothly. This combination has been identified as challenging for you."
        )
    
    def _create_trigram_exercise(self, weak_spot: WeakSpot) -> TypingExercise:
        """Create exercise focusing on a specific trigram"""
        trigram = weak_spot.pattern
        difficulty = min(weak_spot.difficulty_score * 1.3, 1.0)
        
        # Find words containing the trigram
        words_with_trigram = [word for word in self.common_words if trigram in word.lower()]
        
        if not words_with_trigram:
            # Create artificial words with the trigram
            words_with_trigram = [f"test{trigram}test", f"{trigram}word", f"word{trigram}"]
        
        # Create sentences with these words
        sentences = []
        for _ in range(2):
            sentence_words = random.sample(words_with_trigram, min(3, len(words_with_trigram)))
            # Add common words
            sentence_words.extend(random.sample(self.common_words, 4))
            random.shuffle(sentence_words)
            sentence = " ".join(sentence_words).capitalize() + "."
            sentences.append(sentence)
        
        text = " ".join(sentences)
        
        return TypingExercise(
            text=text,
            difficulty=difficulty,
            target_patterns=[trigram],
            exercise_type='trigram',
            estimated_duration=int(len(text) / 5),
            instructions=f"Practice typing the three-letter combination '{trigram}' accurately. This pattern requires special attention."
        )
    
    def _create_general_exercise(self, profile: TypingProfile) -> TypingExercise:
        """Create a general typing exercise"""
        # Use sentence templates to create varied exercises
        template = random.choice(self.sentence_templates)
        
        # Fill in the template with random words
        adj_words = ["quick", "lazy", "bright", "dark", "happy", "sad", "big", "small"]
        noun_words = ["fox", "dog", "cat", "bird", "tree", "house", "car", "book"]
        verb_words = ["jumps", "runs", "walks", "flies", "sits", "stands", "moves", "stops"]
        adv_words = ["quickly", "slowly", "carefully", "easily", "well", "badly", "high", "low"]
        
        text = template.format(
            adj=random.choice(adj_words),
            noun=random.choice(noun_words),
            verb=random.choice(verb_words),
            adv=random.choice(adv_words),
            Noun=random.choice(noun_words).capitalize()
        )
        
        # Add more sentences for variety
        for _ in range(2):
            template2 = random.choice(self.sentence_templates)
            text += " " + template2.format(
                adj=random.choice(adj_words),
                noun=random.choice(noun_words),
                verb=random.choice(verb_words),
                adv=random.choice(adv_words),
                Noun=random.choice(noun_words).capitalize()
            )
        
        return TypingExercise(
            text=text,
            difficulty=0.5,  # Medium difficulty
            target_patterns=[],
            exercise_type='sentence',
            estimated_duration=int(len(text) / 5),
            instructions="Practice typing this sentence accurately and at a comfortable speed."
        )
    
    def generate_progressive_exercises(self, profile: TypingProfile, 
                                     difficulty_level: str = "medium") -> List[TypingExercise]:
        """Generate exercises with progressive difficulty"""
        exercises = []
        
        # Determine difficulty multiplier based on level
        difficulty_multipliers = {
            "easy": 0.5,
            "medium": 1.0,
            "hard": 1.5,
            "expert": 2.0
        }
        
        multiplier = difficulty_multipliers.get(difficulty_level, 1.0)
        
        # Generate exercises with adjusted difficulty
        base_exercises = self.generate_exercises(profile, 3)
        
        for exercise in base_exercises:
            # Adjust difficulty
            adjusted_difficulty = min(exercise.difficulty * multiplier, 1.0)
            
            # Create progressive version
            progressive_exercise = TypingExercise(
                text=exercise.text,
                difficulty=adjusted_difficulty,
                target_patterns=exercise.target_patterns,
                exercise_type=exercise.exercise_type,
                estimated_duration=int(exercise.estimated_duration * multiplier),
                instructions=f"{exercise.instructions} (Difficulty: {difficulty_level})"
            )
            
            exercises.append(progressive_exercise)
        
        return exercises
    
    def generate_mixed_exercise(self, profile: TypingProfile) -> TypingExercise:
        """Generate a mixed exercise incorporating multiple weak spots"""
        # Get top weak spots
        top_weak_spots = profile.weak_spots[:5]
        
        # Create text that incorporates these patterns
        text_parts = []
        target_patterns = []
        
        for weak_spot in top_weak_spots:
            if weak_spot.pattern_type == 'character':
                # Add words with this character
                words_with_char = [word for word in self.common_words 
                                 if weak_spot.pattern in word.lower()]
                if words_with_char:
                    text_parts.append(random.choice(words_with_char))
                    target_patterns.append(weak_spot.pattern)
        
        # Add some common words to make it flow
        text_parts.extend(random.sample(self.common_words, 5))
        random.shuffle(text_parts)
        
        text = " ".join(text_parts).capitalize() + "."
        
        # Calculate average difficulty
        avg_difficulty = sum(ws.difficulty_score for ws in top_weak_spots) / len(top_weak_spots)
        
        return TypingExercise(
            text=text,
            difficulty=min(avg_difficulty, 1.0),
            target_patterns=target_patterns,
            exercise_type='mixed',
            estimated_duration=int(len(text) / 5),
            instructions="This exercise targets multiple areas for improvement. Focus on accuracy and smooth typing."
        )


def create_exercise_generator() -> ExerciseGenerator:
    """Create a new exercise generator instance"""
    return ExerciseGenerator()


if __name__ == "__main__":
    # Example usage
    from analyzer import create_analyzer
    
    generator = create_exercise_generator()
    analyzer = create_analyzer()
    
    try:
        # Analyze user typing (this would normally come from real data)
        profile = analyzer.analyze_user_typing()
        
        # Generate exercises
        exercises = generator.generate_exercises(profile, 3)
        
        print("Generated Exercises:")
        for i, exercise in enumerate(exercises, 1):
            print(f"\nExercise {i}:")
            print(f"Text: {exercise.text}")
            print(f"Type: {exercise.exercise_type}")
            print(f"Difficulty: {exercise.difficulty:.2f}")
            print(f"Target Patterns: {exercise.target_patterns}")
            print(f"Instructions: {exercise.instructions}")
        
        # Generate progressive exercises
        progressive = generator.generate_progressive_exercises(profile, "hard")
        print(f"\nProgressive Exercises (Hard):")
        for i, exercise in enumerate(progressive, 1):
            print(f"Exercise {i}: {exercise.text[:50]}...")
        
    except ValueError as e:
        print(f"Error: {e}")
        print("No typing data available. Run some typing sessions first.")
