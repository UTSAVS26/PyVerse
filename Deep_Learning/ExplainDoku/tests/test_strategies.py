"""
Tests for solving strategies
"""

import pytest
from explaindoku.core.grid import Grid
from explaindoku.core.constraints import ConstraintManager
from explaindoku.core.strategies.singles import SinglesStrategy, SingleResult
from explaindoku.core.strategies.locked_candidates import LockedCandidatesStrategy, LockedCandidateResult
from explaindoku.core.strategies.pairs_triples import PairsTriplesStrategy, PairTripleResult
from explaindoku.core.strategies.fish import FishStrategy, FishResult


class TestSinglesStrategy:
    """Test SinglesStrategy class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.grid = Grid()
        self.cm = ConstraintManager(self.grid)
        self.strategy = SinglesStrategy(self.cm)
    
    def test_find_naked_singles_empty_grid(self):
        """Test finding naked singles in empty grid"""
        results = self.strategy.find_naked_singles()
        assert len(results) == 0
    
    def test_find_naked_singles_with_singleton(self):
        """Test finding naked singles when one exists"""
        # Create a grid where one cell has only one candidate
        grid_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        grid = Grid.from_string(grid_str)
        cm = ConstraintManager(grid)
        strategy = SinglesStrategy(cm)
        
        results = strategy.find_naked_singles()
        assert len(results) > 0
        
        for result in results:
            assert isinstance(result, SingleResult)
            assert result.technique == "naked_single"
            assert result.cell_position is not None
            assert result.value is not None
    
    def test_find_hidden_singles(self):
        """Test finding hidden singles"""
        # Create a grid with hidden singles
        grid_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        grid = Grid.from_string(grid_str)
        cm = ConstraintManager(grid)
        strategy = SinglesStrategy(cm)
        
        results = strategy.find_hidden_singles()
        
        # May or may not find hidden singles, but should return valid results
        for result in results:
            assert isinstance(result, SingleResult)
            assert result.technique == "hidden_single"
            assert result.cell_position is not None
            assert result.value is not None
            assert result.unit_type is not None
            assert result.unit_index is not None
    
    def test_apply_naked_single(self):
        """Test applying naked single"""
        # Create a result
        result = SingleResult(
            technique="naked_single",
            cell_position=(0, 0),
            value=5
        )
        
        # Apply it
        success = self.strategy.apply_naked_single(result)
        assert success
        
        # Check that value was placed
        assert self.cm.grid.get_value(0, 0) == 5
    
    def test_apply_hidden_single(self):
        """Test applying hidden single"""
        # Create a result
        result = SingleResult(
            technique="hidden_single",
            cell_position=(0, 0),
            value=5,
            unit_type="row",
            unit_index=0
        )
        
        # Apply it
        success = self.strategy.apply_hidden_single(result)
        assert success
        
        # Check that value was placed
        assert self.cm.grid.get_value(0, 0) == 5
    
    def test_apply_singles(self):
        """Test applying all singles"""
        # Create a grid with singles
        grid_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        grid = Grid.from_string(grid_str)
        cm = ConstraintManager(grid)
        strategy = SinglesStrategy(cm)
        
        results = strategy.apply_singles()
        
        assert isinstance(results, list)
        for result in results:
            assert isinstance(result, SingleResult)
    
    def test_get_singles_explanation(self):
        """Test explanation generation"""
        result = SingleResult(
            technique="naked_single",
            cell_position=(0, 0),
            value=5
        )
        
        explanation = self.strategy.get_singles_explanation(result)
        assert "Naked Single" in explanation
        assert "R1C1" in explanation
        assert "5" in explanation


class TestLockedCandidatesStrategy:
    """Test LockedCandidatesStrategy class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.grid = Grid()
        self.cm = ConstraintManager(self.grid)
        self.strategy = LockedCandidatesStrategy(self.cm)
    
    def test_find_pointing_candidates(self):
        """Test finding pointing candidates"""
        results = self.strategy.find_pointing_candidates()
        
        # May or may not find pointing candidates, but should return valid results
        for result in results:
            assert isinstance(result, LockedCandidateResult)
            assert result.technique.startswith("pointing")
            assert result.digit is not None
            assert result.unit_type is not None
            assert result.unit_index is not None
            assert result.locked_cells is not None
            assert result.eliminations is not None
    
    def test_find_claiming_candidates(self):
        """Test finding claiming candidates"""
        results = self.strategy.find_claiming_candidates()
        
        # May or may not find claiming candidates, but should return valid results
        for result in results:
            assert isinstance(result, LockedCandidateResult)
            assert result.technique.startswith("claiming")
            assert result.digit is not None
            assert result.unit_type is not None
            assert result.unit_index is not None
            assert result.locked_cells is not None
            assert result.eliminations is not None
    
    def test_apply_locked_candidates(self):
        """Test applying locked candidates"""
        # Create a result with eliminations
        result = LockedCandidateResult(
            technique="pointing_row",
            digit=5,
            unit_type="row",
            unit_index=0,
            eliminations=[((0, 3), 5), ((0, 4), 5)],
            locked_cells=[(0, 0), (0, 1)]
        )
        
        # Apply it
        success = self.strategy.apply_locked_candidates(result)
        assert success
    
    def test_get_locked_candidates_explanation(self):
        """Test explanation generation"""
        result = LockedCandidateResult(
            technique="pointing_row",
            digit=5,
            unit_type="row",
            unit_index=0,
            eliminations=[((0, 3), 5)],
            locked_cells=[(0, 0), (0, 1)]
        )
        
        explanation = self.strategy.get_locked_candidates_explanation(result)
        assert "Locked Candidates" in explanation
        assert "Pointing" in explanation


class TestPairsTriplesStrategy:
    """Test PairsTriplesStrategy class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.grid = Grid()
        self.cm = ConstraintManager(self.grid)
        self.strategy = PairsTriplesStrategy(self.cm)
    
    def test_find_naked_pairs(self):
        """Test finding naked pairs"""
        results = self.strategy.find_naked_pairs()
        
        # May or may not find naked pairs, but should return valid results
        for result in results:
            assert isinstance(result, PairTripleResult)
            assert result.technique == "naked_pair"
            assert result.digits is not None
            assert len(result.digits) == 2
            assert result.cells is not None
            assert len(result.cells) == 2
            assert result.eliminations is not None
    
    def test_find_naked_triples(self):
        """Test finding naked triples"""
        results = self.strategy.find_naked_triples()
        
        # May or may not find naked triples, but should return valid results
        for result in results:
            assert isinstance(result, PairTripleResult)
            assert result.technique == "naked_triple"
            assert result.digits is not None
            assert len(result.digits) == 3
            assert result.cells is not None
            assert len(result.cells) == 3
            assert result.eliminations is not None
    
    def test_find_hidden_pairs(self):
        """Test finding hidden pairs"""
        results = self.strategy.find_hidden_pairs()
        
        # May or may not find hidden pairs, but should return valid results
        for result in results:
            assert isinstance(result, PairTripleResult)
            assert result.technique == "hidden_pair"
            assert result.digits is not None
            assert len(result.digits) == 2
            assert result.cells is not None
            assert len(result.cells) == 2
            assert result.eliminations is not None
    
    def test_find_hidden_triples(self):
        """Test finding hidden triples"""
        results = self.strategy.find_hidden_triples()
        
        # May or may not find hidden triples, but should return valid results
        for result in results:
            assert isinstance(result, PairTripleResult)
            assert result.technique == "hidden_triple"
            assert result.digits is not None
            assert len(result.digits) == 3
            assert result.cells is not None
            assert len(result.cells) == 3
            assert result.eliminations is not None
    
    def test_apply_pairs_triples(self):
        """Test applying pairs/triples"""
        # Create a result with eliminations
        result = PairTripleResult(
            technique="naked_pair",
            digits={1, 2},
            unit_type="row",
            unit_index=0,
            cells=[(0, 0), (0, 1)],
            eliminations=[((0, 2), 1), ((0, 2), 2)]
        )
        
        # Apply it
        success = self.strategy.apply_pairs_triples(result)
        assert success
    
    def test_get_pairs_triples_explanation(self):
        """Test explanation generation"""
        result = PairTripleResult(
            technique="naked_pair",
            digits={1, 2},
            unit_type="row",
            unit_index=0,
            cells=[(0, 0), (0, 1)],
            eliminations=[((0, 2), 1)]
        )
        
        explanation = self.strategy.get_pairs_triples_explanation(result)
        assert "Naked Pair" in explanation


class TestFishStrategy:
    """Test FishStrategy class"""
    
    def setup_method(self):
        """Set up test fixtures"""
        self.grid = Grid()
        self.cm = ConstraintManager(self.grid)
        self.strategy = FishStrategy(self.cm)
    
    def test_find_x_wing(self):
        """Test finding X-Wing patterns"""
        results = self.strategy.find_x_wing()
        
        # May or may not find X-Wing patterns, but should return valid results
        for result in results:
            assert isinstance(result, FishResult)
            assert result.technique == "x_wing"
            assert result.digit is not None
            assert result.base_units is not None
            assert len(result.base_units) == 2
            assert result.cover_units is not None
            assert len(result.cover_units) == 2
            assert result.eliminations is not None
    
    def test_find_swordfish(self):
        """Test finding Swordfish patterns"""
        results = self.strategy.find_swordfish()
        
        # May or may not find Swordfish patterns, but should return valid results
        for result in results:
            assert isinstance(result, FishResult)
            assert result.technique == "swordfish"
            assert result.digit is not None
            assert result.base_units is not None
            assert len(result.base_units) == 3
            assert result.cover_units is not None
            assert len(result.cover_units) == 3
            assert result.eliminations is not None
    
    def test_find_jellyfish(self):
        """Test finding Jellyfish patterns"""
        results = self.strategy.find_jellyfish()
        
        # May or may not find Jellyfish patterns, but should return valid results
        for result in results:
            assert isinstance(result, FishResult)
            assert result.technique == "jellyfish"
            assert result.digit is not None
            assert result.base_units is not None
            assert len(result.base_units) == 4
            assert result.cover_units is not None
            assert len(result.cover_units) == 4
            assert result.eliminations is not None
    
    def test_apply_fish(self):
        """Test applying fish patterns"""
        # Create a result with eliminations
        result = FishResult(
            technique="x_wing",
            digit=5,
            base_units=[0, 1],
            cover_units=[2, 3],
            eliminations=[((4, 2), 5), ((4, 3), 5)]
        )
        
        # Apply it
        success = self.strategy.apply_fish(result)
        assert success
    
    def test_get_fish_explanation(self):
        """Test explanation generation"""
        result = FishResult(
            technique="x_wing",
            digit=5,
            base_units=[0, 1],
            cover_units=[2, 3],
            eliminations=[((4, 2), 5)]
        )
        
        explanation = self.strategy.get_fish_explanation(result)
        assert "X_Wing" in explanation
        assert "5" in explanation
