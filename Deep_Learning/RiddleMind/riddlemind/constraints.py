"""
Constraint representation module for RiddleMind.

This module defines the data structures used to represent logical constraints
and relationships in logic puzzles.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import re


class ConstraintType(Enum):
    """Types of logical constraints."""
    COMPARISON = "comparison"  # older, younger, taller, shorter, etc.
    RELATIONSHIP = "relationship"  # brother, sister, left_of, right_of, etc.
    EQUALITY = "equality"  # A = B
    CONDITIONAL = "conditional"  # If A then B
    NEGATION = "negation"  # not A


@dataclass
class Constraint:
    """Represents a single logical constraint."""
    
    constraint_type: ConstraintType
    predicate: str
    arguments: List[str]
    confidence: float = 1.0
    source_text: str = ""
    
    def __post_init__(self):
        """Validate constraint after initialization."""
        if not self.predicate:
            raise ValueError("Predicate cannot be empty")
        if not self.arguments:
            raise ValueError("Arguments cannot be empty")
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def __str__(self) -> str:
        """String representation of the constraint."""
        args_str = ", ".join(self.arguments)
        return f"{self.predicate}({args_str})"
    
    def __repr__(self) -> str:
        """Detailed string representation."""
        return f"Constraint({self.constraint_type.value}, {self.predicate}, {self.arguments}, conf={self.confidence})"
    
    def to_prolog(self) -> str:
        """Convert to Prolog-style predicate."""
        args_str = ", ".join(self.arguments)
        return f"{self.predicate}({args_str})."
    
    def get_entities(self) -> Set[str]:
        """Get all entities mentioned in this constraint."""
        return set(self.arguments)


@dataclass
class ConstraintSet:
    """A collection of logical constraints."""
    
    constraints: List[Constraint] = field(default_factory=list)
    entities: Set[str] = field(default_factory=set)
    
    def add_constraint(self, constraint: Constraint) -> None:
        """Add a constraint to the set."""
        self.constraints.append(constraint)
        self.entities.update(constraint.get_entities())
    
    def add_constraints(self, constraints: List[Constraint]) -> None:
        """Add multiple constraints to the set."""
        for constraint in constraints:
            self.add_constraint(constraint)
    
    def get_constraints_by_type(self, constraint_type: ConstraintType) -> List[Constraint]:
        """Get all constraints of a specific type."""
        return [c for c in self.constraints if c.constraint_type == constraint_type]
    
    def get_constraints_by_predicate(self, predicate: str) -> List[Constraint]:
        """Get all constraints with a specific predicate."""
        return [c for c in self.constraints if c.predicate == predicate]
    
    def get_entities_with_constraints(self, entity: str) -> List[Constraint]:
        """Get all constraints involving a specific entity."""
        return [c for c in self.constraints if entity in c.arguments]
    
    def __len__(self) -> int:
        return len(self.constraints)
    
    def __iter__(self):
        return iter(self.constraints)
    
    def __str__(self) -> str:
        return "\n".join([str(c) for c in self.constraints])
    
    def to_prolog(self) -> str:
        """Convert all constraints to Prolog format."""
        return "\n".join([c.to_prolog() for c in self.constraints])


class ConstraintBuilder:
    """Helper class for building constraints from parsed text."""
    
    @staticmethod
    def create_comparison_constraint(comparison_type: str, entity1: str, entity2: str, 
                                   source_text: str = "") -> Constraint:
        """Create a comparison constraint (older, younger, taller, etc.)."""
        return Constraint(
            constraint_type=ConstraintType.COMPARISON,
            predicate=comparison_type,
            arguments=[entity1, entity2],
            source_text=source_text
        )
    
    @staticmethod
    def create_relationship_constraint(relationship: str, entity1: str, entity2: str,
                                     source_text: str = "") -> Constraint:
        """Create a relationship constraint (brother, sister, left_of, etc.)."""
        return Constraint(
            constraint_type=ConstraintType.RELATIONSHIP,
            predicate=relationship,
            arguments=[entity1, entity2],
            source_text=source_text
        )
    
    @staticmethod
    def create_equality_constraint(entity1: str, entity2: str,
                                 source_text: str = "") -> Constraint:
        """Create an equality constraint."""
        return Constraint(
            constraint_type=ConstraintType.EQUALITY,
            predicate="equals",
            arguments=[entity1, entity2],
            source_text=source_text
        )
    
    @staticmethod
    def create_conditional_constraint(antecedent: Constraint, consequent: Constraint,
                                    source_text: str = "") -> Constraint:
        """Create a conditional constraint (if A then B)."""
        return Constraint(
            constraint_type=ConstraintType.CONDITIONAL,
            predicate="implies",
            arguments=[str(antecedent), str(consequent)],
            source_text=source_text
        )


class ConstraintValidator:
    """Validates logical consistency of constraints."""
    
    @staticmethod
    def check_contradictions(constraint_set: ConstraintSet) -> List[Tuple[Constraint, Constraint]]:
        """Check for contradictory constraints."""
        contradictions = []
        
        # Check for direct contradictions (A > B and B > A)
        comparisons = constraint_set.get_constraints_by_type(ConstraintType.COMPARISON)
        
        for i, c1 in enumerate(comparisons):
            for c2 in comparisons[i+1:]:
                # Check if same entities but different predicates
                if c1.arguments == c2.arguments:
                    # Same entities, same predicate - not a contradiction
                    continue
                elif c1.arguments == c2.arguments[::-1]:
                    # Same entities in reverse order
                    # Check if predicates are opposites
                    opposites = {
                        "older": "younger",
                        "younger": "older",
                        "taller": "shorter", 
                        "shorter": "taller",
                        "bigger": "smaller",
                        "smaller": "bigger"
                    }
                    if opposites.get(c1.predicate) == c2.predicate:
                        contradictions.append((c1, c2))
                    elif c1.predicate == c2.predicate:
                        # Same predicate with reversed arguments is a contradiction
                        # e.g., "Alice is older than Bob" and "Bob is older than Alice"
                        contradictions.append((c1, c2))
        
        return contradictions
    
    @staticmethod
    def check_transitivity(constraint_set: ConstraintSet) -> List[Constraint]:
        """Check for new constraints that can be derived through transitivity."""
        new_constraints = []
        comparisons = constraint_set.get_constraints_by_type(ConstraintType.COMPARISON)
        
        # Build a graph of relationships
        graph = {}
        for constraint in comparisons:
            if constraint.predicate in ["older", "taller", "bigger"]:
                if constraint.arguments[0] not in graph:
                    graph[constraint.arguments[0]] = set()
                graph[constraint.arguments[0]].add(constraint.arguments[1])
        
        # Check for transitive relationships
        for start in graph:
            visited = set()
            stack = [start]
            
            while stack:
                current = stack.pop()
                if current in visited:
                    continue
                visited.add(current)
                
                if current in graph:
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            # Check if this creates a new transitive relationship
                            if neighbor in graph:
                                for end in graph[neighbor]:
                                    if end not in graph.get(start, set()):
                                        # New transitive relationship found
                                        new_constraint = ConstraintBuilder.create_comparison_constraint(
                                            "older", start, end,
                                            f"Transitive: {start} > {neighbor} > {end}"
                                        )
                                        new_constraints.append(new_constraint)
                        stack.append(neighbor)
        
        return new_constraints
