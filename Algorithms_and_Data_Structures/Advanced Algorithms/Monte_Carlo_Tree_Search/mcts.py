"""
Monte Carlo Tree Search (MCTS) Implementation

A heuristic search algorithm for decision processes, particularly game playing.
Combines the generality of random sampling with the precision of tree search.
"""

import random
import math
import time
from abc import ABC, abstractmethod
from typing import List, Any, Optional, Tuple, Dict
from dataclasses import dataclass
from copy import deepcopy
from enum import Enum


class GameState(ABC):
    """Abstract base class for game states."""
    
    @abstractmethod
    def get_legal_actions(self) -> List[Any]:
        """Get all legal actions from current state."""
        pass
    
    @abstractmethod
    def apply_action(self, action: Any) -> 'GameState':
        """Apply action and return new state."""
        pass
    
    @abstractmethod
    def is_terminal(self) -> bool:
        """Check if state is terminal."""
        pass
    
    @abstractmethod
    def get_reward(self) -> float:
        """Get reward for terminal state (1 for win, 0 for loss, 0.5 for draw)."""
        pass
    
    @abstractmethod
    def get_current_player(self) -> int:
        """Get current player (0 or 1)."""
        pass
    
    @abstractmethod
    def clone(self) -> 'GameState':
        """Create a deep copy of the state."""
        pass


@dataclass
class MCTSNode:
    """Node in the MCTS tree."""
    state: GameState
    parent: Optional['MCTSNode'] = None
    action: Any = None
    children: List['MCTSNode'] = None
    visits: int = 0
    value: float = 0.0
    untried_actions: List[Any] = None
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
        if self.untried_actions is None:
            self.untried_actions = self.state.get_legal_actions()
    
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried."""
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        """Check if node represents terminal state."""
        return self.state.is_terminal()
    
    def get_ucb_value(self, exploration_constant: float) -> float:
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


class MCTS:
    """
    Monte Carlo Tree Search Implementation
    
    Attributes:
        exploration_constant (float): UCB1 exploration constant
        simulation_count (int): Number of simulations per iteration
        time_limit (float): Maximum time to spend on search
        iteration_limit (int): Maximum number of iterations
        verbose (bool): Whether to print debug information
    """
    
    def __init__(self, exploration_constant: float = 1.414, simulation_count: int = 1000,
                 time_limit: float = 1.0, iteration_limit: int = 10000, verbose: bool = False):
        self.exploration_constant = exploration_constant
        self.simulation_count = simulation_count
        self.time_limit = time_limit
        self.iteration_limit = iteration_limit
        self.verbose = verbose
    
    def search(self, initial_state: GameState) -> Any:
        """
        Perform MCTS search to find best action.
        
        Args:
            initial_state: Starting game state
            
        Returns:
            Best action to take
        """
        root = MCTSNode(state=initial_state)
        
        start_time = time.time()
        iterations = 0
        
        while (time.time() - start_time < self.time_limit and 
               iterations < self.iteration_limit):
            
            # Selection
            node = self._select(root)
            
            # Expansion
            if not node.is_terminal():
                node = self._expand(node)
            
            # Simulation
            reward = self._simulate(node.state)
            
            # Backpropagation
            self._backpropagate(node, reward)
            
            iterations += 1
        
        if self.verbose:
            print(f"MCTS completed {iterations} iterations in {time.time() - start_time:.3f}s")
        
        # Return best action
        best_child = max(root.children, key=lambda c: c.visits)
        return best_child.action
    
    def _select(self, node: MCTSNode) -> MCTSNode:
        """Select a node using UCB1."""
        while not node.is_terminal() and node.is_fully_expanded():
            node = max(node.children, key=lambda c: c.get_ucb_value(self.exploration_constant))
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand a node by creating a child."""
        action = random.choice(node.untried_actions)
        node.untried_actions.remove(action)
        
        new_state = node.state.apply_action(action)
        child = MCTSNode(state=new_state, parent=node, action=action)
        node.children.append(child)
        
        return child
    
    def _simulate(self, state: GameState) -> float:
        """Simulate a random playout from the given state."""
        current_state = state.clone()
        
        while not current_state.is_terminal():
            actions = current_state.get_legal_actions()
            if not actions:
                break
            action = random.choice(actions)
            current_state = current_state.apply_action(action)
        
        return current_state.get_reward()
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate the reward up the tree."""
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    def get_action_probabilities(self, state: GameState) -> Dict[Any, float]:
        """Get probability distribution over actions based on visit counts."""
        root = MCTSNode(state=state)
        
        start_time = time.time()
        iterations = 0
        
        while (time.time() - start_time < self.time_limit and 
               iterations < self.iteration_limit):
            
            node = self._select(root)
            if not node.is_terminal():
                node = self._expand(node)
            reward = self._simulate(node.state)
            self._backpropagate(node, reward)
            iterations += 1
        
        # Calculate probabilities based on visit counts
        total_visits = sum(child.visits for child in root.children)
        probabilities = {}
        
        for child in root.children:
            if total_visits > 0:
                probabilities[child.action] = child.visits / total_visits
            else:
                probabilities[child.action] = 1.0 / len(root.children)
        
        return probabilities


# Example Game: Tic-Tac-Toe
class TicTacToeState(GameState):
    """Tic-Tac-Toe game state."""
    
    def __init__(self, board=None, current_player=0):
        if board is None:
            self.board = [[None, None, None] for _ in range(3)]
        else:
            self.board = board
        self.current_player = current_player
    
    def get_legal_actions(self):
        """Get all empty positions."""
        actions = []
        for i in range(3):
            for j in range(3):
                if self.board[i][j] is None:
                    actions.append((i, j))
        return actions
    
    def apply_action(self, action):
        """Apply action and return new state."""
        i, j = action
        new_board = [row[:] for row in self.board]
        new_board[i][j] = self.current_player
        return TicTacToeState(new_board, 1 - self.current_player)
    
    def is_terminal(self):
        """Check if game is over."""
        return (self._check_winner() is not None or 
                len(self.get_legal_actions()) == 0)
    
    def get_reward(self):
        """Get reward for current player."""
        winner = self._check_winner()
        if winner is None:
            return 0.5  # Draw
        elif winner == 0:
            return 1.0  # Player 0 wins
        else:
            return 0.0  # Player 1 wins
    
    def get_current_player(self):
        """Get current player."""
        return self.current_player
    
    def clone(self):
        """Create a deep copy."""
        new_board = [row[:] for row in self.board]
        return TicTacToeState(new_board, self.current_player)
    
    def _check_winner(self):
        """Check if there's a winner."""
        # Check rows
        for i in range(3):
            if (self.board[i][0] is not None and 
                self.board[i][0] == self.board[i][1] == self.board[i][2]):
                return self.board[i][0]
        
        # Check columns
        for j in range(3):
            if (self.board[0][j] is not None and 
                self.board[0][j] == self.board[1][j] == self.board[2][j]):
                return self.board[0][j]
        
        # Check diagonals
        if (self.board[0][0] is not None and 
            self.board[0][0] == self.board[1][1] == self.board[2][2]):
            return self.board[0][0]
        
        if (self.board[0][2] is not None and 
            self.board[0][2] == self.board[1][1] == self.board[2][0]):
            return self.board[0][2]
        
        return None
    
    def __str__(self):
        """String representation of the board."""
        symbols = {None: ' ', 0: 'X', 1: 'O'}
        result = []
        for i in range(3):
            row = []
            for j in range(3):
                row.append(symbols[self.board[i][j]])
            result.append('|'.join(row))
        return '\n-----\n'.join(result)


# Example Game: Connect Four
class ConnectFourState(GameState):
    """Connect Four game state."""
    
    def __init__(self, board=None, current_player=0):
        if board is None:
            self.board = [[None for _ in range(7)] for _ in range(6)]
        else:
            self.board = board
        self.current_player = current_player
    
    def get_legal_actions(self):
        """Get all columns that aren't full."""
        actions = []
        for col in range(7):
            if self.board[0][col] is None:
                actions.append(col)
        return actions
    
    def apply_action(self, action):
        """Apply action and return new state."""
        col = action
        new_board = [row[:] for row in self.board]
        
        # Find the lowest empty position in the column
        for row in range(5, -1, -1):
            if new_board[row][col] is None:
                new_board[row][col] = self.current_player
                break
        
        return ConnectFourState(new_board, 1 - self.current_player)
    
    def is_terminal(self):
        """Check if game is over."""
        return (self._check_winner() is not None or 
                len(self.get_legal_actions()) == 0)
    
    def get_reward(self):
        """Get reward for current player."""
        winner = self._check_winner()
        if winner is None:
            return 0.5  # Draw
        elif winner == 0:
            return 1.0  # Player 0 wins
        else:
            return 0.0  # Player 1 wins
    
    def get_current_player(self):
        """Get current player."""
        return self.current_player
    
    def clone(self):
        """Create a deep copy."""
        new_board = [row[:] for row in self.board]
        return ConnectFourState(new_board, self.current_player)
    
    def _check_winner(self):
        """Check if there's a winner."""
        # Check horizontal
        for row in range(6):
            for col in range(4):
                if (self.board[row][col] is not None and
                    self.board[row][col] == self.board[row][col+1] == 
                    self.board[row][col+2] == self.board[row][col+3]):
                    return self.board[row][col]
        
        # Check vertical
        for row in range(3):
            for col in range(7):
                if (self.board[row][col] is not None and
                    self.board[row][col] == self.board[row+1][col] == 
                    self.board[row+2][col] == self.board[row+3][col]):
                    return self.board[row][col]
        
        # Check diagonal (positive slope)
        for row in range(3):
            for col in range(4):
                if (self.board[row][col] is not None and
                    self.board[row][col] == self.board[row+1][col+1] == 
                    self.board[row+2][col+2] == self.board[row+3][col+3]):
                    return self.board[row][col]
        
        # Check diagonal (negative slope)
        for row in range(3, 6):
            for col in range(4):
                if (self.board[row][col] is not None and
                    self.board[row][col] == self.board[row-1][col+1] == 
                    self.board[row-2][col+2] == self.board[row-3][col+3]):
                    return self.board[row][col]
        
        return None


# Example Game: 2048 (Simplified)
class Game2048State(GameState):
    """Simplified 2048 game state."""
    
    def __init__(self, board=None, score=0):
        if board is None:
            self.board = [[0 for _ in range(4)] for _ in range(4)]
            # Add two initial tiles
            self._add_random_tile()
            self._add_random_tile()
        else:
            self.board = board
        self.score = score
    
    def get_legal_actions(self):
        """Get all possible moves (up, down, left, right)."""
        actions = []
        for direction in ['up', 'down', 'left', 'right']:
            if self._can_move(direction):
                actions.append(direction)
        return actions
    
    def apply_action(self, action):
        """Apply move and return new state."""
        new_board = [row[:] for row in self.board]
        new_score = self.score
        
        if action == 'up':
            new_board, new_score = self._move_up(new_board, new_score)
        elif action == 'down':
            new_board, new_score = self._move_down(new_board, new_score)
        elif action == 'left':
            new_board, new_score = self._move_left(new_board, new_score)
        elif action == 'right':
            new_board, new_score = self._move_right(new_board, new_score)
        
        new_state = Game2048State(new_board, new_score)
        new_state._add_random_tile()
        return new_state
    
    def is_terminal(self):
        """Check if game is over."""
        return len(self.get_legal_actions()) == 0
    
    def get_reward(self):
        """Get reward based on score."""
        return self.score / 1000.0  # Normalize score
    
    def get_current_player(self):
        """Single player game."""
        return 0
    
    def clone(self):
        """Create a deep copy."""
        new_board = [row[:] for row in self.board]
        return Game2048State(new_board, self.score)
    
    def _can_move(self, direction):
        """Check if move is possible."""
        test_board = [row[:] for row in self.board]
        if direction == 'up':
            return self._move_up(test_board, 0)[1] > 0
        elif direction == 'down':
            return self._move_down(test_board, 0)[1] > 0
        elif direction == 'left':
            return self._move_left(test_board, 0)[1] > 0
        elif direction == 'right':
            return self._move_right(test_board, 0)[1] > 0
        return False
    
    def _move_up(self, board, score):
        """Move tiles up."""
        for col in range(4):
            # Merge tiles
            merged = []
            for row in range(4):
                if board[row][col] != 0:
                    merged.append(board[row][col])
            
            # Combine adjacent equal tiles
            i = 0
            while i < len(merged) - 1:
                if merged[i] == merged[i + 1]:
                    merged[i] *= 2
                    score += merged[i]
                    merged.pop(i + 1)
                i += 1
            
            # Fill with zeros
            while len(merged) < 4:
                merged.append(0)
            
            # Update board
            for row in range(4):
                board[row][col] = merged[row]
        
        return board, score
    
    def _move_down(self, board, score):
        """Move tiles down."""
        for col in range(4):
            # Merge tiles
            merged = []
            for row in range(3, -1, -1):
                if board[row][col] != 0:
                    merged.append(board[row][col])
            
            # Combine adjacent equal tiles
            i = 0
            while i < len(merged) - 1:
                if merged[i] == merged[i + 1]:
                    merged[i] *= 2
                    score += merged[i]
                    merged.pop(i + 1)
                i += 1
            
            # Fill with zeros
            while len(merged) < 4:
                merged.append(0)
            
            # Update board
            for row in range(4):
                board[row][col] = merged[3 - row]
        
        return board, score
    
    def _move_left(self, board, score):
        """Move tiles left."""
        for row in range(4):
            # Merge tiles
            merged = []
            for col in range(4):
                if board[row][col] != 0:
                    merged.append(board[row][col])
            
            # Combine adjacent equal tiles
            i = 0
            while i < len(merged) - 1:
                if merged[i] == merged[i + 1]:
                    merged[i] *= 2
                    score += merged[i]
                    merged.pop(i + 1)
                i += 1
            
            # Fill with zeros
            while len(merged) < 4:
                merged.append(0)
            
            # Update board
            for col in range(4):
                board[row][col] = merged[col]
        
        return board, score
    
    def _move_right(self, board, score):
        """Move tiles right."""
        for row in range(4):
            # Merge tiles
            merged = []
            for col in range(3, -1, -1):
                if board[row][col] != 0:
                    merged.append(board[row][col])
            
            # Combine adjacent equal tiles
            i = 0
            while i < len(merged) - 1:
                if merged[i] == merged[i + 1]:
                    merged[i] *= 2
                    score += merged[i]
                    merged.pop(i + 1)
                i += 1
            
            # Fill with zeros
            while len(merged) < 4:
                merged.append(0)
            
            # Update board
            for col in range(4):
                board[row][col] = merged[3 - col]
        
        return board, score
    
    def _add_random_tile(self):
        """Add a random tile (2 or 4) to the board."""
        empty_positions = []
        for i in range(4):
            for j in range(4):
                if self.board[i][j] == 0:
                    empty_positions.append((i, j))
        
        if empty_positions:
            i, j = random.choice(empty_positions)
            self.board[i][j] = 2 if random.random() < 0.9 else 4


def main():
    """Example usage of MCTS."""
    print("=== Monte Carlo Tree Search Demo ===\n")
    
    # Example 1: Tic-Tac-Toe
    print("1. Tic-Tac-Toe Game:")
    state = TicTacToeState()
    
    mcts = MCTS(exploration_constant=1.414, time_limit=0.5, verbose=True)
    
    print("Initial board:")
    print(state)
    
    # Play a few moves
    for move in range(3):
        if not state.is_terminal():
            action = mcts.search(state)
            state = state.apply_action(action)
            print(f"\nMove {move + 1} (Player {1 - state.current_player}):")
            print(state)
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Connect Four
    print("2. Connect Four Game:")
    state = ConnectFourState()
    
    mcts = MCTS(exploration_constant=1.414, time_limit=0.3, verbose=True)
    
    # Play a few moves
    for move in range(3):
        if not state.is_terminal():
            action = mcts.search(state)
            state = state.apply_action(action)
            print(f"Move {move + 1}: Column {action}")
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: 2048
    print("3. 2048 Game:")
    state = Game2048State()
    
    mcts = MCTS(exploration_constant=1.414, time_limit=0.2, verbose=True)
    
    # Play a few moves
    for move in range(5):
        if not state.is_terminal():
            action = mcts.search(state)
            state = state.apply_action(action)
            print(f"Move {move + 1}: {action}")
            print(f"Score: {state.score}")
    
    print(f"\nFinal score: {state.score}")


if __name__ == "__main__":
    main() 