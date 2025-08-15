import random
import time
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass, field
from enum import Enum
import sys
import os

# Add the parent directory to the path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.timer import GameTimer


class CardState(Enum):
    """Enum for card states."""
    HIDDEN = "hidden"
    REVEALED = "revealed"
    MATCHED = "matched"


@dataclass
class Card:
    """Represents a single card in the memory game."""
    id: int
    value: str
    state: CardState = CardState.HIDDEN
    row: int = 0
    col: int = 0
    
    def __post_init__(self):
        if self.state is None:
            self.state = CardState.HIDDEN
    
    def reveal(self) -> None:
        """Reveal the card."""
        if self.state == CardState.HIDDEN:
            self.state = CardState.REVEALED
    
    def hide(self) -> None:
        """Hide the card."""
        if self.state == CardState.REVEALED:
            self.state = CardState.HIDDEN
    
    def match(self) -> None:
        """Mark the card as matched."""
        self.state = CardState.MATCHED
    
    def is_hidden(self) -> bool:
        """Check if card is hidden."""
        return self.state == CardState.HIDDEN
    
    def is_revealed(self) -> bool:
        """Check if card is revealed."""
        return self.state == CardState.REVEALED
    
    def is_matched(self) -> bool:
        """Check if card is matched."""
        return self.state == CardState.MATCHED


class GameState(Enum):
    """Enum for game states."""
    WAITING = "waiting"
    PLAYING = "playing"
    PAUSED = "paused"
    COMPLETED = "completed"


class MemoryGame:
    """Core memory game logic."""
    
    def __init__(self, grid_size: int = 4):
        self.grid_size = grid_size
        self.cards: List[Card] = []
        self.revealed_cards: List[Card] = []
        self.matched_pairs = 0
        self.total_pairs = (grid_size * grid_size) // 2  # For odd grids, this will be the max possible pairs
        self.moves = 0
        self.mistakes = 0
        self.game_state = GameState.WAITING
        self.timer = GameTimer()
        self.card_values = self._generate_card_values()
        self._initialize_cards()
    
    def _generate_card_values(self) -> List[str]:
        """Generate card values for the game."""
        # Emoji-based card values for visual appeal
        emojis = ["🐶", "🐱", "🐭", "🐹", "🐰", "🦊", "🐻", "🐼", "🐨", "🐯", 
                  "🦁", "🐮", "🐷", "🐸", "🐵", "🐔", "🐧", "🐦", "🦆", "🦅"]
        
        # Calculate total cards needed
        total_cards = self.grid_size * self.grid_size
        
        # For odd grid sizes, we need to handle the case where we can't have perfect pairs
        if total_cards % 2 == 1:
            # For odd grid sizes, we'll have one extra card that doesn't have a pair
            needed_pairs = total_cards // 2
            selected_emojis = emojis[:needed_pairs]
            
            # Create pairs of each emoji
            card_values = []
            for emoji in selected_emojis:
                card_values.extend([emoji, emoji])
            
            # Add one extra card to make the total odd
            if len(card_values) < total_cards:
                card_values.append(emojis[needed_pairs])
        else:
            # For even grid sizes, we can have perfect pairs
            needed_pairs = total_cards // 2
            selected_emojis = emojis[:needed_pairs]
            
            # Create pairs of each emoji
            card_values = []
            for emoji in selected_emojis:
                card_values.extend([emoji, emoji])
        
        return card_values
    
    def _initialize_cards(self) -> None:
        """Initialize the game board with cards."""
        self.cards = []
        card_values = self.card_values.copy()
        random.shuffle(card_values)
        
        card_id = 0
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                card = Card(
                    id=card_id,
                    value=card_values[card_id],
                    row=row,
                    col=col
                )
                self.cards.append(card)
                card_id += 1
    
    def start_game(self) -> None:
        """Start a new game."""
        self.game_state = GameState.PLAYING
        self.moves = 0
        self.mistakes = 0
        self.matched_pairs = 0
        self.revealed_cards = []
        self.timer.start_game()
        self._initialize_cards()
    
    def get_card_at(self, row: int, col: int) -> Optional[Card]:
        """Get card at specific position."""
        for card in self.cards:
            if card.row == row and card.col == col:
                return card
        return None
    
    def click_card(self, row: int, col: int) -> Dict[str, Any]:
        """Handle card click and return game state update."""
        if self.game_state != GameState.PLAYING:
            return {"error": "Game not in playing state"}
        
        card = self.get_card_at(row, col)
        if not card:
            return {"error": "Invalid card position"}
        
        if card.is_matched() or card.is_revealed():
            return {"error": "Card already revealed or matched"}
        
        # Record the move
        reaction_time = self.timer.record_move()
        self.moves += 1
        
        # Reveal the card
        card.reveal()
        self.revealed_cards.append(card)
        
        result = {
            "card_revealed": True,
            "card_value": card.value,
            "card_position": (row, col),
            "reaction_time": reaction_time,
            "moves": self.moves
        }
        
        # Check if we have two revealed cards
        if len(self.revealed_cards) == 2:
            result.update(self._check_match())
        
        return result
    
    def _check_match(self) -> Dict[str, Any]:
        """Check if the two revealed cards match."""
        card1, card2 = self.revealed_cards
        
        if card1.value == card2.value:
            # Match found
            card1.match()
            card2.match()
            self.matched_pairs += 1
            self.revealed_cards = []
            
            result = {
                "match_found": True,
                "matched_cards": [(card1.row, card1.col), (card2.row, card2.col)],
                "matched_pairs": self.matched_pairs,
                "total_pairs": self.total_pairs
            }
            
            # Check if game is complete
            if self.matched_pairs == self.total_pairs:
                result.update(self._complete_game())
            
            return result
        else:
            # No match
            self.mistakes += 1
            result = {
                "match_found": False,
                "mistakes": self.mistakes
            }
            
            # Schedule hiding of cards after a delay
            # In a real implementation, this would be handled by the UI
            return result
    
    def hide_unmatched_cards(self) -> List[Tuple[int, int]]:
        """Hide unmatched revealed cards."""
        positions_to_hide = []
        for card in self.revealed_cards:
            if not card.is_matched():
                card.hide()
                positions_to_hide.append((card.row, card.col))
        
        self.revealed_cards = []
        return positions_to_hide
    
    def _complete_game(self) -> Dict[str, Any]:
        """Complete the game and return final statistics."""
        self.game_state = GameState.COMPLETED
        total_time = self.timer.end_game()
        
        return {
            "game_completed": True,
            "total_time": total_time,
            "final_stats": self.get_game_stats()
        }
    
    def get_game_stats(self) -> Dict[str, Any]:
        """Get comprehensive game statistics."""
        timer_stats = self.timer.get_reaction_time_stats()
        
        return {
            "total_time": self.timer.get_total_time(),
            "moves": self.moves,
            "mistakes": self.mistakes,
            "matched_pairs": self.matched_pairs,
            "total_pairs": self.total_pairs,
            "completion_percentage": (self.matched_pairs / self.total_pairs) * 100,
            "avg_reaction_time": timer_stats['avg_reaction_time'],
            "min_reaction_time": timer_stats['min_reaction_time'],
            "max_reaction_time": timer_stats['max_reaction_time'],
            "std_reaction_time": timer_stats['std_reaction_time'],
            "total_moves": timer_stats['total_moves']
        }
    
    def get_board_state(self) -> List[List[Dict[str, Any]]]:
        """Get current board state for UI rendering."""
        board = []
        for row in range(self.grid_size):
            board_row = []
            for col in range(self.grid_size):
                card = self.get_card_at(row, col)
                if card:
                    board_row.append({
                        "value": card.value if not card.is_hidden() else "?",
                        "state": card.state.value,
                        "row": row,
                        "col": col,
                        "is_hidden": card.is_hidden(),
                        "is_revealed": card.is_revealed(),
                        "is_matched": card.is_matched()
                    })
            board.append(board_row)
        return board
    
    def is_game_complete(self) -> bool:
        """Check if the game is complete."""
        return self.matched_pairs == self.total_pairs
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current game state for UI updates."""
        return {
            "game_state": self.game_state.value,
            "moves": self.moves,
            "mistakes": self.mistakes,
            "matched_pairs": self.matched_pairs,
            "total_pairs": self.total_pairs,
            "completion_percentage": (self.matched_pairs / self.total_pairs) * 100,
            "board": self.get_board_state(),
            "stats": self.get_game_stats()
        }
    
    def reset_game(self) -> None:
        """Reset the game to initial state."""
        self.game_state = GameState.WAITING
        self.cards = []
        self.revealed_cards = []
        self.matched_pairs = 0
        self.moves = 0
        self.mistakes = 0
        self.timer = GameTimer()
        self._initialize_cards() 