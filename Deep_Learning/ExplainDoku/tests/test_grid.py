"""
Tests for the Grid module
"""

import pytest
import numpy as np
from explaindoku.core.grid import Grid, Cell


class TestCell:
    """Test Cell class"""
    
    def test_cell_creation(self):
        """Test cell creation with different parameters"""
        # Empty cell
        cell = Cell(0, 0)
        assert cell.row == 0
        assert cell.col == 0
        assert cell.value is None
        assert cell.candidates == set(range(1, 10))
        
        # Cell with value
        cell = Cell(1, 2, value=5)
        assert cell.row == 1
        assert cell.col == 2
        assert cell.value == 5
        assert cell.candidates == set()
        
        # Cell with custom candidates
        cell = Cell(3, 4, candidates={1, 3, 5})
        assert cell.candidates == {1, 3, 5}
    
    def test_cell_properties(self):
        """Test cell properties"""
        cell = Cell(2, 3)
        
        assert cell.position == "R3C4"
        assert not cell.is_filled
        assert cell.candidate_count == 9
        
        cell.set_value(7)
        assert cell.is_filled
        assert cell.candidate_count == 0
        assert cell.value == 7
    
    def test_candidate_operations(self):
        """Test candidate operations"""
        cell = Cell(0, 0)
        
        # Remove candidate
        assert cell.remove_candidate(5)
        assert 5 not in cell.candidates
        assert cell.candidate_count == 8
        
        # Remove non-existent candidate
        assert not cell.remove_candidate(5)
        assert cell.candidate_count == 8
    
    def test_set_value(self):
        """Test setting cell value"""
        cell = Cell(0, 0)
        cell.set_value(3)
        
        assert cell.value == 3
        assert cell.candidates == set()
        assert cell.is_filled


class TestGrid:
    """Test Grid class"""
    
    def test_grid_creation(self):
        """Test grid creation"""
        # Empty grid
        grid = Grid()
        assert grid.size == 9
        assert grid.box_size == 3
        assert len(grid.cells) == 81
        
        # Grid with initial data
        initial_data = np.full((9, 9), None, dtype=object)
        initial_data[0, 0] = 5
        grid = Grid(initial_data)
        assert grid.get_value(0, 0) == 5
    
    def test_grid_from_string(self):
        """Test creating grid from string"""
        # Valid string
        grid_str = "530070000600195000098000060800060003400803001700020006060000280000419005000080079"
        grid = Grid.from_string(grid_str)
        assert grid.get_value(0, 0) == 5
        assert grid.get_value(0, 1) == 3
        assert grid.get_value(0, 2) is None  # Empty cell
        
        # Invalid string length
        with pytest.raises(ValueError):
            Grid.from_string("12345")
        
        # String with dots
        grid_str = "5.3.....6.6.195....98.....68..6...3..4.8.3..1..7..2..6..6.....28...419..5....8..79"
        # This string is 82 characters, so it should fail
        with pytest.raises(ValueError):
            Grid.from_string(grid_str)
    
    def test_cell_access(self):
        """Test accessing cells"""
        grid = Grid()
        
        # Get cell
        cell = grid.get_cell(2, 3)
        assert isinstance(cell, Cell)
        assert cell.row == 2
        assert cell.col == 3
        
        # Set and get value
        grid.set_value(2, 3, 7)
        assert grid.get_value(2, 3) == 7
        assert grid.get_cell(2, 3).value == 7
    
    def test_position_validation(self):
        """Test position validation"""
        grid = Grid()
        
        assert grid.is_valid_position(0, 0)
        assert grid.is_valid_position(8, 8)
        assert not grid.is_valid_position(-1, 0)
        assert not grid.is_valid_position(0, 9)
        assert not grid.is_valid_position(9, 0)
    
    def test_unit_operations(self):
        """Test unit operations (rows, columns, boxes)"""
        grid = Grid()
        
        # Row operations
        row_positions = grid.get_row(2)
        assert len(row_positions) == 9
        assert (2, 0) in row_positions
        assert (2, 8) in row_positions
        
        # Column operations
        col_positions = grid.get_col(3)
        assert len(col_positions) == 9
        assert (0, 3) in col_positions
        assert (8, 3) in col_positions
        
        # Box operations
        box_positions = grid.get_box(1, 1)
        assert len(box_positions) == 9
        assert (0, 0) in box_positions
        assert (2, 2) in box_positions
        
        # Box from different position
        box_positions = grid.get_box(4, 4)
        assert len(box_positions) == 9
        assert (3, 3) in box_positions
        assert (5, 5) in box_positions
    
    def test_peer_relationships(self):
        """Test peer relationships"""
        grid = Grid()
        
        # Test peers for corner cell
        peers = grid.get_peers(0, 0)
        assert len(peers) == 20  # 8 from row + 8 from col + 4 from box (excluding self)
        assert (0, 1) in peers  # Same row
        assert (1, 0) in peers  # Same column
        assert (1, 1) in peers  # Same box
        
        # Test peers for center cell
        peers = grid.get_peers(4, 4)
        assert len(peers) == 20
        assert (4, 0) in peers  # Same row
        assert (0, 4) in peers  # Same column
        assert (3, 3) in peers  # Same box
    
    def test_valid_move_checking(self):
        """Test valid move checking"""
        grid = Grid()
        
        # Empty grid - all moves should be valid
        assert grid.is_valid_move(0, 0, 5)
        assert grid.is_valid_move(4, 4, 3)
        
        # Place a value
        grid.set_value(0, 0, 5)
        
        # Invalid moves
        assert not grid.is_valid_move(0, 0, 3)  # Cell already filled
        assert not grid.is_valid_move(0, 1, 5)  # Same row
        assert not grid.is_valid_move(1, 0, 5)  # Same column
        assert not grid.is_valid_move(1, 1, 5)  # Same box
        
        # Valid moves
        assert grid.is_valid_move(0, 1, 3)  # Different value, different position
        assert grid.is_valid_move(8, 8, 5)  # Far away position
    
    def test_solved_checking(self):
        """Test solved state checking"""
        grid = Grid()
        
        # Empty grid
        assert not grid.is_solved()
        
        # Partially filled grid
        grid.set_value(0, 0, 5)
        assert not grid.is_solved()
        
        # Create a solved grid
        solved_str = "123456789456789123789123456234567891567891234891234567345678912678912345912345678"
        solved_grid = Grid.from_string(solved_str)
        assert solved_grid.is_solved()
    
    def test_validity_checking(self):
        """Test grid validity checking"""
        grid = Grid()
        
        # Empty grid is valid
        assert grid.is_valid()
        
        # Valid partial grid
        grid.set_value(0, 0, 5)
        grid.set_value(0, 1, 3)
        assert grid.is_valid()
        
        # Invalid grid - duplicate in row
        grid.set_value(0, 2, 5)
        assert not grid.is_valid()
        
        # Reset and test column conflict
        grid = Grid()
        grid.set_value(0, 0, 5)
        grid.set_value(1, 0, 5)
        assert not grid.is_valid()
        
        # Reset and test box conflict
        grid = Grid()
        grid.set_value(0, 0, 5)
        grid.set_value(1, 1, 5)
        assert not grid.is_valid()
    
    def test_grid_copying(self):
        """Test grid copying"""
        grid = Grid()
        grid.set_value(0, 0, 5)
        grid.set_value(4, 4, 3)
        
        # Deep copy
        grid_copy = grid.copy()
        
        # Original and copy should be independent
        assert grid.get_value(0, 0) == 5
        assert grid_copy.get_value(0, 0) == 5
        
        grid_copy.set_value(0, 0, 7)
        assert grid.get_value(0, 0) == 5
        assert grid_copy.get_value(0, 0) == 7
    
    def test_string_conversion(self):
        """Test string conversion methods"""
        grid = Grid()
        grid.set_value(0, 0, 5)
        grid.set_value(0, 1, 3)
        grid.set_value(8, 8, 9)
        
        # To string
        grid_str = grid.to_string()
        assert len(grid_str) == 81
        assert grid_str[0] == '5'
        assert grid_str[1] == '3'
        assert grid_str[80] == '9'
        
        # Display string
        display_str = grid.to_display_string()
        assert '5' in display_str
        assert '3' in display_str
        assert '9' in display_str
        assert '|' in display_str  # Box separators
        assert '-' in display_str  # Row separators
    
    def test_empty_filled_cells(self):
        """Test getting empty and filled cells"""
        grid = Grid()
        
        # Initially all cells are empty
        assert len(grid.get_empty_cells()) == 81
        assert len(grid.get_filled_cells()) == 0
        
        # Add some values
        grid.set_value(0, 0, 5)
        grid.set_value(4, 4, 3)
        grid.set_value(8, 8, 9)
        
        assert len(grid.get_empty_cells()) == 78
        assert len(grid.get_filled_cells()) == 3
        
        # Check specific positions
        filled_positions = grid.get_filled_cells()
        assert (0, 0) in filled_positions
        assert (4, 4) in filled_positions
        assert (8, 8) in filled_positions
