"""
Flashcard Formatter Module

Formats questions and answers into structured flashcards.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class QuestionType(Enum):
    """Types of questions that can be generated."""
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    FILL_BLANK = "fill_blank"
    TRUE_FALSE = "true_false"
    MULTIPLE_CHOICE = "multiple_choice"


@dataclass
class Flashcard:
    """Represents a single flashcard."""
    question: str
    answer: str
    question_type: QuestionType
    difficulty: str = "medium"
    tags: List[str] = None
    source_text: str = ""
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}


class FlashcardFormatter:
    """Formats questions and answers into structured flashcards."""
    
    def __init__(self):
        """Initialize the flashcard formatter."""
        self.difficulty_keywords = {
            'easy': ['basic', 'simple', 'fundamental', 'introductory'],
            'medium': ['intermediate', 'standard', 'common'],
            'hard': ['advanced', 'complex', 'detailed', 'specialized']
        }
    
    def create_flashcard(self, question: str, answer: str, question_type: QuestionType = QuestionType.FACTUAL,
                        source_text: str = "", tags: List[str] = None) -> Flashcard:
        """
        Create a flashcard from question and answer.
        
        Args:
            question: The question text
            answer: The answer text
            question_type: Type of question
            source_text: Original source text
            tags: Tags for categorization
            
        Returns:
            Flashcard object
        """
        if tags is None:
            tags = []
        
        # Determine difficulty based on content
        difficulty = self._determine_difficulty(question, answer)
        
        # Create metadata
        metadata = {
            'question_length': len(question),
            'answer_length': len(answer),
            'created_timestamp': None  # Could be set to datetime.now()
        }
        
        return Flashcard(
            question=question.strip(),
            answer=answer.strip(),
            question_type=question_type,
            difficulty=difficulty,
            tags=tags,
            source_text=source_text,
            metadata=metadata
        )
    
    def create_flashcards_from_qa_pairs(self, qa_pairs: List[Tuple[str, str]], 
                                      question_type: QuestionType = QuestionType.FACTUAL,
                                      source_text: str = "", tags: List[str] = None) -> List[Flashcard]:
        """
        Create multiple flashcards from question-answer pairs.
        
        Args:
            qa_pairs: List of (question, answer) tuples
            question_type: Type of questions
            source_text: Original source text
            tags: Tags for categorization
            
        Returns:
            List of Flashcard objects
        """
        flashcards = []
        
        for question, answer in qa_pairs:
            if question and answer:  # Skip empty questions/answers
                flashcard = self.create_flashcard(
                    question=question,
                    answer=answer,
                    question_type=question_type,
                    source_text=source_text,
                    tags=tags
                )
                flashcards.append(flashcard)
        
        return flashcards
    
    def create_fill_blank_flashcards(self, text: str, keywords: List[str], 
                                   num_cards: int = 5) -> List[Flashcard]:
        """
        Create fill-in-the-blank flashcards.
        
        Args:
            text: Source text
            keywords: Keywords to create blanks for
            num_cards: Number of flashcards to create
            
        Returns:
            List of fill-in-the-blank flashcards
        """
        import nltk
        from nltk.tokenize import sent_tokenize
        
        flashcards = []
        sentences = sent_tokenize(text)
        
        for sentence in sentences[:num_cards]:
            for keyword in keywords:
                if keyword.lower() in sentence.lower():
                    # Create fill-in-the-blank question
                    question = sentence.replace(keyword, "____")
                    answer = keyword
                    
                    flashcard = self.create_flashcard(
                        question=question,
                        answer=answer,
                        question_type=QuestionType.FILL_BLANK,
                        source_text=text,
                        tags=['fill_blank']
                    )
                    flashcards.append(flashcard)
                    break
        
        return flashcards[:num_cards]
    
    def create_true_false_flashcards(self, statements: List[str], 
                                   num_cards: int = 5) -> List[Flashcard]:
        """
        Create true/false flashcards from statements.
        
        Args:
            statements: List of statements
            num_cards: Number of flashcards to create
            
        Returns:
            List of true/false flashcards
        """
        flashcards = []
        
        for statement in statements[:num_cards]:
            # For now, we'll mark all as true (in a real system, you'd need logic to determine truth)
            flashcard = self.create_flashcard(
                question=f"True or False: {statement}",
                answer="True",  # This would need more sophisticated logic
                question_type=QuestionType.TRUE_FALSE,
                tags=['true_false']
            )
            flashcards.append(flashcard)
        
        return flashcards
    
    def format_flashcard_for_display(self, flashcard: Flashcard, format_type: str = "simple") -> str:
        """
        Format a flashcard for display.
        
        Args:
            flashcard: Flashcard object
            format_type: Type of formatting ('simple', 'detailed', 'anki')
            
        Returns:
            Formatted string
        """
        if format_type == "simple":
            return f"Q: {flashcard.question}\nA: {flashcard.answer}"
        
        elif format_type == "detailed":
            return (f"Question: {flashcard.question}\n"
                   f"Answer: {flashcard.answer}\n"
                   f"Type: {flashcard.question_type.value}\n"
                   f"Difficulty: {flashcard.difficulty}\n"
                   f"Tags: {', '.join(flashcard.tags)}")
        
        elif format_type == "anki":
            return f"{flashcard.question}\t{flashcard.answer}"
        
        else:
            return f"Q: {flashcard.question}\nA: {flashcard.answer}"
    
    def format_flashcards_for_export(self, flashcards: List[Flashcard], 
                                   format_type: str = "csv") -> str:
        """
        Format multiple flashcards for export.
        
        Args:
            flashcards: List of flashcards
            format_type: Export format ('csv', 'anki', 'json')
            
        Returns:
            Formatted string for export
        """
        if format_type == "csv":
            return self._format_csv(flashcards)
        elif format_type == "anki":
            return self._format_anki(flashcards)
        elif format_type == "json":
            return self._format_json(flashcards)
        else:
            return self._format_csv(flashcards)
    
    def _determine_difficulty(self, question: str, answer: str) -> str:
        """Determine the difficulty level of a flashcard."""
        text = f"{question} {answer}".lower()
        
        # Count difficulty indicators
        difficulty_scores = {'easy': 0, 'medium': 0, 'hard': 0}
        
        for difficulty, keywords in self.difficulty_keywords.items():
            for keyword in keywords:
                if keyword in text:
                    difficulty_scores[difficulty] += 1
        
        # Return the difficulty with highest score, default to medium
        max_score = max(difficulty_scores.values())
        if max_score == 0:
            return "medium"
        
        for difficulty, score in difficulty_scores.items():
            if score == max_score:
                return difficulty
        
        return "medium"
    
    def _format_csv(self, flashcards: List[Flashcard]) -> str:
        """Format flashcards as CSV."""
        csv_lines = ["Question,Answer,Type,Difficulty,Tags"]
        
        for flashcard in flashcards:
            tags_str = ";".join(flashcard.tags)
            csv_line = f'"{flashcard.question}","{flashcard.answer}","{flashcard.question_type.value}","{flashcard.difficulty}","{tags_str}"'
            csv_lines.append(csv_line)
        
        return "\n".join(csv_lines)
    
    def _format_anki(self, flashcards: List[Flashcard]) -> str:
        """Format flashcards for Anki import."""
        anki_lines = []
        
        for flashcard in flashcards:
            anki_line = f"{flashcard.question}\t{flashcard.answer}"
            anki_lines.append(anki_line)
        
        return "\n".join(anki_lines)
    
    def _format_json(self, flashcards: List[Flashcard]) -> str:
        """Format flashcards as JSON."""
        import json
        
        flashcard_data = []
        for flashcard in flashcards:
            data = {
                'question': flashcard.question,
                'answer': flashcard.answer,
                'question_type': flashcard.question_type.value,
                'difficulty': flashcard.difficulty,
                'tags': flashcard.tags,
                'metadata': flashcard.metadata
            }
            flashcard_data.append(data)
        
        return json.dumps(flashcard_data, indent=2)
    
    def validate_flashcard(self, flashcard: Flashcard) -> Tuple[bool, List[str]]:
        """
        Validate a flashcard for quality.
        
        Args:
            flashcard: Flashcard to validate
            
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Check question length
        if len(flashcard.question) < 5:
            issues.append("Question too short")
        elif len(flashcard.question) > 500:
            issues.append("Question too long")
        
        # Check answer length
        if len(flashcard.answer) < 3:
            issues.append("Answer too short")
        elif len(flashcard.answer) > 1000:
            issues.append("Answer too long")
        
        # Check for empty content
        if not flashcard.question.strip():
            issues.append("Empty question")
        if not flashcard.answer.strip():
            issues.append("Empty answer")
        
        # Check for duplicate content
        if flashcard.question.lower() == flashcard.answer.lower():
            issues.append("Question and answer are identical")
        
        return len(issues) == 0, issues
    
    def filter_flashcards_by_quality(self, flashcards: List[Flashcard], 
                                   min_quality_score: float = 0.7) -> List[Flashcard]:
        """
        Filter flashcards based on quality criteria.
        
        Args:
            flashcards: List of flashcards to filter
            min_quality_score: Minimum quality score (0-1)
            
        Returns:
            List of high-quality flashcards
        """
        quality_flashcards = []
        
        for flashcard in flashcards:
            is_valid, issues = self.validate_flashcard(flashcard)
            
            if is_valid:
                # Calculate quality score (simple heuristic)
                quality_score = self._calculate_quality_score(flashcard)
                
                if quality_score >= min_quality_score:
                    quality_flashcards.append(flashcard)
        
        return quality_flashcards
    
    def _calculate_quality_score(self, flashcard: Flashcard) -> float:
        """Calculate a quality score for a flashcard (0-1)."""
        score = 1.0
        
        # Penalize very short or very long content
        if len(flashcard.question) < 10:
            score -= 0.3
        if len(flashcard.answer) < 10:
            score -= 0.3
        
        if len(flashcard.question) > 200:
            score -= 0.1
        if len(flashcard.answer) > 300:
            score -= 0.1
        
        # Bonus for good question types
        if flashcard.question_type in [QuestionType.CONCEPTUAL, QuestionType.FILL_BLANK]:
            score += 0.05
        
        # Bonus for appropriate difficulty
        if flashcard.difficulty == "medium":
            score += 0.05
        
        return max(0.0, min(1.0, score))
