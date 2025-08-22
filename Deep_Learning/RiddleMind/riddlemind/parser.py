"""
Natural language parser for RiddleMind.

This module handles the parsing of logic puzzles written in natural language
and converts them into structured logical constraints.
"""

import re
import spacy
from typing import List, Dict, Set, Tuple, Optional
from .constraints import (
    Constraint, ConstraintSet, ConstraintType, 
    ConstraintBuilder, ConstraintValidator
)


class PuzzleParser:
    """Parses logic puzzles from natural language into logical constraints."""
    
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """Initialize the parser with spaCy model."""
        try:
            self.nlp = spacy.load(spacy_model)
        except OSError:
            try:
                # Fallback to basic English model
                self.nlp = spacy.load("en_core_web_sm")
            except OSError:
                # If both fail, create a minimal NLP object
                self.nlp = spacy.blank("en")
        
        # Define patterns for different types of constraints
        self.comparison_patterns = {
            "older": [
                r"(\w+)\s+is\s+older\s+than\s+(\w+)",
                r"(\w+)\s+is\s+the\s+older\s+one",
                r"(\w+)\s+is\s+(\d+)\s+years\s+older\s+than\s+(\w+)"
            ],
            "younger": [
                r"(\w+)\s+is\s+younger\s+than\s+(\w+)",
                r"(\w+)\s+is\s+the\s+younger\s+one",
                r"(\w+)\s+is\s+(\d+)\s+years\s+younger\s+than\s+(\w+)"
            ],
            "taller": [
                r"(\w+)\s+is\s+taller\s+than\s+(\w+)",
                r"(\w+)\s+is\s+the\s+taller\s+one"
            ],
            "shorter": [
                r"(\w+)\s+is\s+shorter\s+than\s+(\w+)",
                r"(\w+)\s+is\s+the\s+shorter\s+one"
            ],
            "bigger": [
                r"(\w+)\s+is\s+bigger\s+than\s+(\w+)",
                r"(\w+)\s+is\s+larger\s+than\s+(\w+)"
            ],
            "smaller": [
                r"(\w+)\s+is\s+smaller\s+than\s+(\w+)",
                r"(\w+)\s+is\s+the\s+smaller\s+one"
            ]
        }
        
        self.relationship_patterns = {
            "brother": [
                r"(\w+)\s+is\s+(\w+)'s\s+brother",
                r"(\w+)\s+and\s+(\w+)\s+are\s+brothers"
            ],
            "sister": [
                r"(\w+)\s+is\s+(\w+)'s\s+sister",
                r"(\w+)\s+and\s+(\w+)\s+are\s+sisters"
            ],
            "left_of": [
                r"(\w+)\s+is\s+to\s+the\s+left\s+of\s+(\w+)",
                r"(\w+)\s+sits\s+to\s+the\s+left\s+of\s+(\w+)"
            ],
            "right_of": [
                r"(\w+)\s+is\s+to\s+the\s+right\s+of\s+(\w+)",
                r"(\w+)\s+sits\s+to\s+the\s+right\s+of\s+(\w+)"
            ],
            "next_to": [
                r"(\w+)\s+is\s+next\s+to\s+(\w+)",
                r"(\w+)\s+sits\s+next\s+to\s+(\w+)"
            ]
        }
        
        self.conditional_patterns = [
            r"if\s+(.+?)\s+then\s+(.+)",
            r"when\s+(.+?)\s+then\s+(.+)",
            r"(.+?)\s+implies\s+(.+)"
        ]
        
        self.question_patterns = [
            r"who\s+is\s+(.+?)\?",
            r"which\s+(.+?)\s+is\s+(.+?)\?",
            r"what\s+is\s+(.+?)\?",
            r"find\s+(.+?)\."
        ]
    
    def parse(self, text: str) -> Tuple[ConstraintSet, List[str]]:
        """
        Parse natural language text into logical constraints.
        
        Args:
            text: The natural language puzzle text
            
        Returns:
            Tuple of (constraint_set, questions)
        """
        text = text.strip()
        constraint_set = ConstraintSet()
        questions = []
        
        # Split text into sentences
        doc = self.nlp(text)
        sentences = [sent.text.strip() for sent in doc.sents]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check if it's a question
            if self._is_question(sentence):
                questions.append(sentence)
                continue
            
            # Parse constraints from the sentence
            constraints = self._parse_sentence(sentence)
            constraint_set.add_constraints(constraints)
        
        return constraint_set, questions
    
    def _is_question(self, sentence: str) -> bool:
        """Check if a sentence is a question."""
        sentence_lower = sentence.lower()
        question_indicators = ["who", "what", "where", "when", "why", "how", "which"]
        
        # Check for question words at the beginning
        for indicator in question_indicators:
            if sentence_lower.startswith(indicator):
                return True
        
        # Check for question mark
        if sentence.endswith("?"):
            return True
        
        # Check for question patterns
        for pattern in self.question_patterns:
            if re.search(pattern, sentence_lower):
                return True
        
        return False
    
    def _parse_sentence(self, sentence: str) -> List[Constraint]:
        """Parse a single sentence into constraints."""
        constraints = []
        sentence_lower = sentence.lower()
        
        # Parse comparison constraints
        for comparison_type, patterns in self.comparison_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sentence_lower)
                for match in matches:
                    if len(match.groups()) >= 2:
                        entity1 = match.group(1).title()
                        entity2 = match.group(2).title()
                        
                        constraint = ConstraintBuilder.create_comparison_constraint(
                            comparison_type, entity1, entity2, sentence
                        )
                        constraints.append(constraint)
        
        # Parse relationship constraints
        for relationship_type, patterns in self.relationship_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, sentence_lower)
                for match in matches:
                    if len(match.groups()) >= 2:
                        entity1 = match.group(1).title()
                        entity2 = match.group(2).title()
                        
                        constraint = ConstraintBuilder.create_relationship_constraint(
                            relationship_type, entity1, entity2, sentence
                        )
                        constraints.append(constraint)
        
        # Parse conditional constraints
        for pattern in self.conditional_patterns:
            matches = re.finditer(pattern, sentence_lower)
            for match in matches:
                if len(match.groups()) >= 2:
                    antecedent_text = match.group(1).strip()
                    consequent_text = match.group(2).strip()
                    
                    # Parse antecedent and consequent
                    antecedent_constraints = self._parse_sentence(antecedent_text)
                    consequent_constraints = self._parse_sentence(consequent_text)
                    
                    if antecedent_constraints and consequent_constraints:
                        for ant in antecedent_constraints:
                            for cons in consequent_constraints:
                                constraint = ConstraintBuilder.create_conditional_constraint(
                                    ant, cons, sentence
                                )
                                constraints.append(constraint)
        
        # Parse equality constraints
        equality_patterns = [
            r"(\w+)\s+equals\s+(\w+)",
            r"(\w+)\s+is\s+equal\s+to\s+(\w+)",
            r"(\w+)\s+=\s+(\w+)"
        ]
        
        for pattern in equality_patterns:
            matches = re.finditer(pattern, sentence_lower)
            for match in matches:
                if len(match.groups()) >= 2:
                    entity1 = match.group(1).title()
                    entity2 = match.group(2).title()
                    
                    constraint = ConstraintBuilder.create_equality_constraint(
                        entity1, entity2, sentence
                    )
                    constraints.append(constraint)
        
        return constraints
    
    def extract_entities(self, text: str) -> Set[str]:
        """Extract named entities from text."""
        doc = self.nlp(text)
        entities = set()
        
        # Extract named entities
        for ent in doc.ents:
            if ent.label_ in ["PERSON", "ORG", "GPE"]:
                entities.add(ent.text.title())
        
        # Extract capitalized words that might be names
        words = text.split()
        for word in words:
            if word[0].isupper() and len(word) > 1:
                entities.add(word.title())
        
        return entities
    
    def validate_parsing(self, constraint_set: ConstraintSet) -> Dict[str, List[str]]:
        """Validate the parsed constraints and return any issues."""
        issues = {
            "warnings": [],
            "errors": []
        }
        
        # Check for contradictions
        contradictions = ConstraintValidator.check_contradictions(constraint_set)
        if contradictions:
            for c1, c2 in contradictions:
                issues["errors"].append(
                    f"Contradiction found: {c1} contradicts {c2}"
                )
        
        # Check for missing entities
        if len(constraint_set.entities) < 3:
            issues["warnings"].append(
                "Few entities detected. Make sure names are properly capitalized."
            )
        
        # Check for empty constraint set
        if len(constraint_set) == 0:
            issues["warnings"].append(
                "No constraints were parsed. Check the input format."
            )
        
        return issues
