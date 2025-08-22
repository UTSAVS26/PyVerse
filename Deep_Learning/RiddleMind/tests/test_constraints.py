"""
Tests for the constraints module.
"""

import pytest
from riddlemind.constraints import (
    Constraint, ConstraintSet, ConstraintType, 
    ConstraintBuilder, ConstraintValidator
)


class TestConstraint:
    """Test the Constraint class."""
    
    def test_constraint_creation(self):
        """Test basic constraint creation."""
        constraint = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"],
            source_text="Alice is older than Bob"
        )
        
        assert constraint.constraint_type == ConstraintType.COMPARISON
        assert constraint.predicate == "older"
        assert constraint.arguments == ["Alice", "Bob"]
        assert constraint.confidence == 1.0
        assert constraint.source_text == "Alice is older than Bob"
    
    def test_constraint_string_representation(self):
        """Test string representation of constraints."""
        constraint = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        assert str(constraint) == "older(Alice, Bob)"
    
    def test_constraint_prolog_format(self):
        """Test Prolog format conversion."""
        constraint = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        assert constraint.to_prolog() == "older(Alice, Bob)."
    
    def test_constraint_entities(self):
        """Test entity extraction from constraints."""
        constraint = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        entities = constraint.get_entities()
        assert entities == {"Alice", "Bob"}
    
    def test_constraint_validation(self):
        """Test constraint validation."""
        # Valid constraint
        constraint = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        assert constraint.predicate == "older"
        
        # Invalid constraint - empty predicate
        with pytest.raises(ValueError, match="Predicate cannot be empty"):
            Constraint(
                constraint_type=ConstraintType.COMPARISON,
                predicate="",
                arguments=["Alice", "Bob"]
            )
        
        # Invalid constraint - empty arguments
        with pytest.raises(ValueError, match="Arguments cannot be empty"):
            Constraint(
                constraint_type=ConstraintType.COMPARISON,
                predicate="older",
                arguments=[]
            )
        
        # Invalid constraint - confidence out of range
        with pytest.raises(ValueError, match="Confidence must be between 0.0 and 1.0"):
            Constraint(
                constraint_type=ConstraintType.COMPARISON,
                predicate="older",
                arguments=["Alice", "Bob"],
                confidence=1.5
            )


class TestConstraintSet:
    """Test the ConstraintSet class."""
    
    def test_constraint_set_creation(self):
        """Test basic constraint set creation."""
        constraint_set = ConstraintSet()
        assert len(constraint_set) == 0
        assert len(constraint_set.entities) == 0
    
    def test_adding_constraints(self):
        """Test adding constraints to the set."""
        constraint_set = ConstraintSet()
        
        constraint1 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        constraint2 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="taller",
            arguments=["Alice", "Charlie"]
        )
        
        constraint_set.add_constraint(constraint1)
        constraint_set.add_constraint(constraint2)
        
        assert len(constraint_set) == 2
        assert len(constraint_set.entities) == 3
        assert "Alice" in constraint_set.entities
        assert "Bob" in constraint_set.entities
        assert "Charlie" in constraint_set.entities
    
    def test_adding_multiple_constraints(self):
        """Test adding multiple constraints at once."""
        constraint_set = ConstraintSet()
        
        constraints = [
            Constraint(
                constraint_type=ConstraintType.COMPARISON,
                predicate="older",
                arguments=["Alice", "Bob"]
            ),
            Constraint(
                constraint_type=ConstraintType.COMPARISON,
                predicate="taller",
                arguments=["Alice", "Charlie"]
            )
        ]
        
        constraint_set.add_constraints(constraints)
        
        assert len(constraint_set) == 2
        assert len(constraint_set.entities) == 3
    
    def test_getting_constraints_by_type(self):
        """Test filtering constraints by type."""
        constraint_set = ConstraintSet()
        
        comparison_constraint = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        relationship_constraint = Constraint(
            constraint_type=ConstraintType.RELATIONSHIP,
            predicate="brother",
            arguments=["Alice", "Bob"]
        )
        
        constraint_set.add_constraints([comparison_constraint, relationship_constraint])
        
        comparison_constraints = constraint_set.get_constraints_by_type(ConstraintType.COMPARISON)
        relationship_constraints = constraint_set.get_constraints_by_type(ConstraintType.RELATIONSHIP)
        
        assert len(comparison_constraints) == 1
        assert len(relationship_constraints) == 1
        assert comparison_constraints[0].predicate == "older"
        assert relationship_constraints[0].predicate == "brother"
    
    def test_getting_constraints_by_predicate(self):
        """Test filtering constraints by predicate."""
        constraint_set = ConstraintSet()
        
        constraint1 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        constraint2 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Bob", "Charlie"]
        )
        
        constraint3 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="taller",
            arguments=["Alice", "Bob"]
        )
        
        constraint_set.add_constraints([constraint1, constraint2, constraint3])
        
        older_constraints = constraint_set.get_constraints_by_predicate("older")
        taller_constraints = constraint_set.get_constraints_by_predicate("taller")
        
        assert len(older_constraints) == 2
        assert len(taller_constraints) == 1
    
    def test_getting_entities_with_constraints(self):
        """Test getting constraints involving a specific entity."""
        constraint_set = ConstraintSet()
        
        constraint1 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        constraint2 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="taller",
            arguments=["Alice", "Charlie"]
        )
        
        constraint_set.add_constraints([constraint1, constraint2])
        
        alice_constraints = constraint_set.get_entities_with_constraints("Alice")
        bob_constraints = constraint_set.get_entities_with_constraints("Bob")
        
        assert len(alice_constraints) == 2
        assert len(bob_constraints) == 1
    
    def test_constraint_set_string_representation(self):
        """Test string representation of constraint sets."""
        constraint_set = ConstraintSet()
        
        constraint1 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        constraint2 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="taller",
            arguments=["Alice", "Charlie"]
        )
        
        constraint_set.add_constraints([constraint1, constraint2])
        
        expected = "older(Alice, Bob)\ntaller(Alice, Charlie)"
        assert str(constraint_set) == expected
    
    def test_constraint_set_prolog_format(self):
        """Test Prolog format conversion for constraint sets."""
        constraint_set = ConstraintSet()
        
        constraint1 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        constraint2 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="taller",
            arguments=["Alice", "Charlie"]
        )
        
        constraint_set.add_constraints([constraint1, constraint2])
        
        expected = "older(Alice, Bob).\ntaller(Alice, Charlie)."
        assert constraint_set.to_prolog() == expected


class TestConstraintBuilder:
    """Test the ConstraintBuilder class."""
    
    def test_create_comparison_constraint(self):
        """Test creating comparison constraints."""
        constraint = ConstraintBuilder.create_comparison_constraint(
            "older", "Alice", "Bob", "Alice is older than Bob"
        )
        
        assert constraint.constraint_type == ConstraintType.COMPARISON
        assert constraint.predicate == "older"
        assert constraint.arguments == ["Alice", "Bob"]
        assert constraint.source_text == "Alice is older than Bob"
    
    def test_create_relationship_constraint(self):
        """Test creating relationship constraints."""
        constraint = ConstraintBuilder.create_relationship_constraint(
            "brother", "Alice", "Bob", "Alice is Bob's brother"
        )
        
        assert constraint.constraint_type == ConstraintType.RELATIONSHIP
        assert constraint.predicate == "brother"
        assert constraint.arguments == ["Alice", "Bob"]
        assert constraint.source_text == "Alice is Bob's brother"
    
    def test_create_equality_constraint(self):
        """Test creating equality constraints."""
        constraint = ConstraintBuilder.create_equality_constraint(
            "Alice", "Bob", "Alice equals Bob"
        )
        
        assert constraint.constraint_type == ConstraintType.EQUALITY
        assert constraint.predicate == "equals"
        assert constraint.arguments == ["Alice", "Bob"]
        assert constraint.source_text == "Alice equals Bob"
    
    def test_create_conditional_constraint(self):
        """Test creating conditional constraints."""
        antecedent = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        consequent = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="taller",
            arguments=["Alice", "Bob"]
        )
        
        constraint = ConstraintBuilder.create_conditional_constraint(
            antecedent, consequent, "If Alice is older than Bob then Alice is taller than Bob"
        )
        
        assert constraint.constraint_type == ConstraintType.CONDITIONAL
        assert constraint.predicate == "implies"
        assert constraint.arguments == ["older(Alice, Bob)", "taller(Alice, Bob)"]
        assert constraint.source_text == "If Alice is older than Bob then Alice is taller than Bob"


class TestConstraintValidator:
    """Test the ConstraintValidator class."""
    
    def test_check_contradictions(self):
        """Test contradiction detection."""
        constraint_set = ConstraintSet()
        
        # Add contradictory constraints
        constraint1 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        constraint2 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="younger",
            arguments=["Bob", "Alice"]
        )
        
        constraint_set.add_constraints([constraint1, constraint2])
        
        contradictions = ConstraintValidator.check_contradictions(constraint_set)
        
        assert len(contradictions) == 1
        assert contradictions[0][0] == constraint1
        assert contradictions[0][1] == constraint2
    
    def test_check_no_contradictions(self):
        """Test when no contradictions exist."""
        constraint_set = ConstraintSet()
        
        constraint1 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        constraint2 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="taller",
            arguments=["Alice", "Bob"]
        )
        
        constraint_set.add_constraints([constraint1, constraint2])
        
        contradictions = ConstraintValidator.check_contradictions(constraint_set)
        
        assert len(contradictions) == 0
    
    def test_check_transitivity(self):
        """Test transitivity detection."""
        constraint_set = ConstraintSet()
        
        # A > B, B > C should imply A > C
        constraint1 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        constraint2 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Bob", "Charlie"]
        )
        
        constraint_set.add_constraints([constraint1, constraint2])
        
        transitive_constraints = ConstraintValidator.check_transitivity(constraint_set)
        
        assert len(transitive_constraints) == 1
        assert transitive_constraints[0].predicate == "older"
        assert transitive_constraints[0].arguments == ["Alice", "Charlie"]
    
    def test_check_no_transitivity(self):
        """Test when no transitivity is possible."""
        constraint_set = ConstraintSet()
        
        constraint1 = Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate="older",
            arguments=["Alice", "Bob"]
        )
        
        constraint_set.add_constraint(constraint1)
        
        transitive_constraints = ConstraintValidator.check_transitivity(constraint_set)
        
        assert len(transitive_constraints) == 0
