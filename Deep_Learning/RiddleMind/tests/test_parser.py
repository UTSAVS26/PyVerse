"""
Tests for the parser module.
"""

import pytest
from unittest.mock import Mock, patch
from riddlemind.parser import PuzzleParser
from riddlemind.constraints import ConstraintSet, ConstraintType


class TestPuzzleParser:
    """Test the PuzzleParser class."""
    
    def test_parser_initialization(self):
        """Test parser initialization."""
        parser = PuzzleParser()
        assert parser is not None
        assert hasattr(parser, 'nlp')
    
    def test_is_question(self):
        """Test question detection."""
        parser = PuzzleParser()
        
        # Test question words
        assert parser._is_question("Who is the oldest?")
        assert parser._is_question("What is Alice's age?")
        assert parser._is_question("Which person is taller?")
        assert parser._is_question("Where does Alice sit?")
        assert parser._is_question("When did this happen?")
        assert parser._is_question("Why is Alice older?")
        assert parser._is_question("How old is Alice?")
        
        # Test question mark
        assert parser._is_question("Is Alice older than Bob?")
        
        # Test non-questions
        assert not parser._is_question("Alice is older than Bob.")
        assert not parser._is_question("This is a statement.")
        assert not parser._is_question("Alice and Bob are friends.")
    
    def test_parse_comparison_constraints(self):
        """Test parsing comparison constraints."""
        parser = PuzzleParser()
        
        # Test age comparisons
        text = "Alice is older than Bob. Charlie is younger than Alice."
        constraint_set, questions = parser.parse(text)
        
        assert len(constraint_set) == 2
        assert len(questions) == 0
        
        # Check that constraints are parsed correctly
        older_constraints = constraint_set.get_constraints_by_predicate("older")
        younger_constraints = constraint_set.get_constraints_by_predicate("younger")
        
        assert len(older_constraints) == 1
        assert len(younger_constraints) == 1
        
        assert older_constraints[0].arguments == ["Alice", "Bob"]
        assert younger_constraints[0].arguments == ["Charlie", "Alice"]
    
    def test_parse_height_constraints(self):
        """Test parsing height constraints."""
        parser = PuzzleParser()
        
        text = "Alice is taller than Bob. Bob is taller than Charlie."
        constraint_set, questions = parser.parse(text)
        
        assert len(constraint_set) == 2
        
        taller_constraints = constraint_set.get_constraints_by_predicate("taller")
        assert len(taller_constraints) == 2
        
        assert taller_constraints[0].arguments == ["Alice", "Bob"]
        assert taller_constraints[1].arguments == ["Bob", "Charlie"]
    
    def test_parse_relationship_constraints(self):
        """Test parsing relationship constraints."""
        parser = PuzzleParser()
        
        text = "Alice is Bob's sister. Bob is Charlie's brother."
        constraint_set, questions = parser.parse(text)
        
        assert len(constraint_set) == 2
        
        sister_constraints = constraint_set.get_constraints_by_predicate("sister")
        brother_constraints = constraint_set.get_constraints_by_predicate("brother")
        
        assert len(sister_constraints) == 1
        assert len(brother_constraints) == 1
        
        assert sister_constraints[0].arguments == ["Alice", "Bob"]
        assert brother_constraints[0].arguments == ["Bob", "Charlie"]
    
    def test_parse_spatial_constraints(self):
        """Test parsing spatial constraints."""
        parser = PuzzleParser()
        
        text = "Alice sits to the left of Bob. Bob sits to the left of Charlie."
        constraint_set, questions = parser.parse(text)
        
        assert len(constraint_set) == 2
        
        left_constraints = constraint_set.get_constraints_by_predicate("left_of")
        assert len(left_constraints) == 2
        
        assert left_constraints[0].arguments == ["Alice", "Bob"]
        assert left_constraints[1].arguments == ["Bob", "Charlie"]
    
    def test_parse_equality_constraints(self):
        """Test parsing equality constraints."""
        parser = PuzzleParser()
        
        text = "Alice equals Bob. Charlie is equal to David."
        constraint_set, questions = parser.parse(text)
        
        assert len(constraint_set) == 2
        
        equals_constraints = constraint_set.get_constraints_by_predicate("equals")
        assert len(equals_constraints) == 2
        
        assert equals_constraints[0].arguments == ["Alice", "Bob"]
        assert equals_constraints[1].arguments == ["Charlie", "David"]
    
    def test_parse_questions(self):
        """Test parsing questions."""
        parser = PuzzleParser()
        
        text = "Alice is older than Bob. Who is the oldest?"
        constraint_set, questions = parser.parse(text)
        
        assert len(constraint_set) == 1
        assert len(questions) == 1
        assert questions[0] == "Who is the oldest?"
    
    def test_parse_complex_text(self):
        """Test parsing complex text with multiple constraints and questions."""
        parser = PuzzleParser()
        
        text = """
        Alice is older than Bob. 
        Bob is taller than Charlie. 
        Alice is Bob's sister. 
        Who is the oldest? 
        Who is the tallest?
        """
        
        constraint_set, questions = parser.parse(text)
        
        assert len(constraint_set) == 3
        assert len(questions) == 2
        
        # Check constraints
        older_constraints = constraint_set.get_constraints_by_predicate("older")
        taller_constraints = constraint_set.get_constraints_by_predicate("taller")
        sister_constraints = constraint_set.get_constraints_by_predicate("sister")
        
        assert len(older_constraints) == 1
        assert len(taller_constraints) == 1
        assert len(sister_constraints) == 1
        
        # Check questions
        assert "Who is the oldest?" in questions
        assert "Who is the tallest?" in questions
    
    def test_parse_empty_text(self):
        """Test parsing empty text."""
        parser = PuzzleParser()
        
        constraint_set, questions = parser.parse("")
        
        assert len(constraint_set) == 0
        assert len(questions) == 0
    
    def test_parse_whitespace_only(self):
        """Test parsing whitespace-only text."""
        parser = PuzzleParser()
        
        constraint_set, questions = parser.parse("   \n\t   ")
        
        assert len(constraint_set) == 0
        assert len(questions) == 0
    
    def test_extract_entities(self):
        """Test entity extraction."""
        parser = PuzzleParser()
        
        text = "Alice is older than Bob. Charlie is Bob's brother."
        entities = parser.extract_entities(text)
        
        # Should extract capitalized names
        assert "Alice" in entities
        assert "Bob" in entities
        assert "Charlie" in entities
    
    def test_validate_parsing(self):
        """Test parsing validation."""
        parser = PuzzleParser()
        
        # Test with valid constraints
        constraint_set = ConstraintSet()
        from riddlemind.constraints import ConstraintBuilder
        
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("older", "Alice", "Bob")
        )
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("taller", "Alice", "Charlie")
        )
        
        issues = parser.validate_parsing(constraint_set)
        
        assert len(issues["errors"]) == 0
        assert len(issues["warnings"]) == 0
        
        # Test with contradictions
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("younger", "Bob", "Alice")
        )
        
        issues = parser.validate_parsing(constraint_set)
        
        assert len(issues["errors"]) > 0  # Should detect contradiction
        
        # Test with few entities
        constraint_set = ConstraintSet()
        constraint_set.add_constraint(
            ConstraintBuilder.create_comparison_constraint("older", "Alice", "Bob")
        )
        
        issues = parser.validate_parsing(constraint_set)
        
        assert len(issues["warnings"]) > 0  # Should warn about few entities
        
        # Test with empty constraint set
        constraint_set = ConstraintSet()
        
        issues = parser.validate_parsing(constraint_set)
        
        assert len(issues["warnings"]) > 0  # Should warn about no constraints
    
    def test_parse_with_spacy_fallback(self):
        """Test parser with spaCy model fallback."""
        with patch('spacy.load') as mock_load:
            # Simulate spaCy model not found
            mock_load.side_effect = OSError("Model not found")
            
            # Should not raise an exception
            parser = PuzzleParser()
            assert parser is not None
    
    def test_parse_case_sensitivity(self):
        """Test that parsing is case-insensitive but preserves entity names."""
        parser = PuzzleParser()
        
        text = "ALICE is OLDER than BOB. charlie is YOUNGER than alice."
        constraint_set, questions = parser.parse(text)
        
        assert len(constraint_set) == 2
        
        # Check that entity names are properly capitalized
        older_constraints = constraint_set.get_constraints_by_predicate("older")
        younger_constraints = constraint_set.get_constraints_by_predicate("younger")
        
        assert older_constraints[0].arguments == ["Alice", "Bob"]
        assert younger_constraints[0].arguments == ["Charlie", "Alice"]
    
    def test_parse_multiple_sentences(self):
        """Test parsing multiple sentences."""
        parser = PuzzleParser()
        
        text = "Alice is older than Bob. Bob is taller than Charlie. Who is the oldest?"
        constraint_set, questions = parser.parse(text)
        
        assert len(constraint_set) == 2
        assert len(questions) == 1
        
        # Check that all sentences are processed
        older_constraints = constraint_set.get_constraints_by_predicate("older")
        taller_constraints = constraint_set.get_constraints_by_predicate("taller")
        
        assert len(older_constraints) == 1
        assert len(taller_constraints) == 1
