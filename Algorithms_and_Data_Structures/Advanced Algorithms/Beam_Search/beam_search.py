"""
Beam Search Algorithm Implementation

A heuristic search algorithm that explores a graph by expanding the most promising
nodes in a limited set (beam).
"""

import heapq
from abc import ABC, abstractmethod
from typing import List, Any, Callable, Optional, Tuple
from dataclasses import dataclass
from copy import deepcopy


@dataclass
class SearchNode:
    """Represents a node in the search tree."""
    state: Any
    path: List[Any]
    cost: float
    heuristic: float
    f_value: float  # cost + heuristic
    
    def __lt__(self, other):
        return self.f_value < other.f_value


class Problem(ABC):
    """Abstract base class for problems that can be solved with beam search."""
    
    @abstractmethod
    def get_initial_state(self) -> Any:
        """Return the initial state of the problem."""
        pass
    
    @abstractmethod
    def get_successors(self, state: Any) -> List[Tuple[Any, float]]:
        """Return list of (successor_state, action_cost) pairs."""
        pass
    
    @abstractmethod
    def is_goal(self, state: Any) -> bool:
        """Check if the given state is a goal state."""
        pass
    
    @abstractmethod
    def heuristic(self, state: Any) -> float:
        """Return heuristic value for the given state."""
        pass


class BeamSearch:
    """
    Beam Search Algorithm Implementation
    
    Attributes:
        beam_width (int): Number of nodes to keep at each level
        max_iterations (int): Maximum number of iterations
        verbose (bool): Whether to print debug information
    """
    
    def __init__(self, beam_width: int = 5, max_iterations: int = 100, verbose: bool = False):
        self.beam_width = beam_width
        self.max_iterations = max_iterations
        self.verbose = verbose
        self.nodes_expanded = 0
        self.nodes_generated = 0
    
    def search(self, problem: Problem) -> Optional[List[Any]]:
        """
        Perform beam search on the given problem.
        
        Args:
            problem: Problem instance to solve
            
        Returns:
            List of actions to reach goal, or None if no solution found
        """
        initial_state = problem.get_initial_state()
        initial_node = SearchNode(
            state=initial_state,
            path=[],
            cost=0.0,
            heuristic=problem.heuristic(initial_state),
            f_value=problem.heuristic(initial_state)
        )
        
        # Initialize beam with initial node
        beam = [initial_node]
        self.nodes_generated = 1
        
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"Iteration {iteration + 1}, Beam size: {len(beam)}")
            
            # Check if any node in beam is a goal
            for node in beam:
                if problem.is_goal(node.state):
                    if self.verbose:
                        print(f"Goal found! Path length: {len(node.path)}")
                        print(f"Total cost: {node.cost}")
                        print(f"Nodes expanded: {self.nodes_expanded}")
                        print(f"Nodes generated: {self.nodes_generated}")
                    return node.path
            
            # Generate all successors
            all_successors = []
            for node in beam:
                self.nodes_expanded += 1
                successors = problem.get_successors(node.state)
                
                for successor_state, action_cost in successors:
                    self.nodes_generated += 1
                    new_cost = node.cost + action_cost
                    heuristic = problem.heuristic(successor_state)
                    f_value = new_cost + heuristic
                    
                    new_path = node.path + [successor_state]
                    successor_node = SearchNode(
                        state=successor_state,
                        path=new_path,
                        cost=new_cost,
                        heuristic=heuristic,
                        f_value=f_value
                    )
                    all_successors.append(successor_node)
            
            if not all_successors:
                if self.verbose:
                    print("No successors generated. Search failed.")
                return None
            
            # Select top-k nodes for next beam
            beam = self._select_best_nodes(all_successors, self.beam_width)
            
            if self.verbose:
                best_f = min(node.f_value for node in beam)
                print(f"Best f-value: {best_f}")
        
        if self.verbose:
            print(f"Maximum iterations ({self.max_iterations}) reached.")
        return None
    
    def _select_best_nodes(self, nodes: List[SearchNode], k: int) -> List[SearchNode]:
        """Select the k best nodes based on f-value."""
        return heapq.nsmallest(k, nodes)
    
    def get_statistics(self) -> dict:
        """Return search statistics."""
        return {
            'nodes_expanded': self.nodes_expanded,
            'nodes_generated': self.nodes_generated,
            'beam_width': self.beam_width,
            'max_iterations': self.max_iterations
        }


# Example Problem: 8-Puzzle
class EightPuzzleProblem(Problem):
    """8-Puzzle problem implementation for beam search."""
    
    def __init__(self, initial_state=None, goal_state=None):
        if initial_state is None:
            # Default initial state
            self.initial_state = [
                [1, 2, 3],
                [4, 0, 6],
                [7, 5, 8]
            ]
        else:
            self.initial_state = initial_state
            
        if goal_state is None:
            # Default goal state
            self.goal_state = [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 0]
            ]
        else:
            self.goal_state = goal_state
    
    def get_initial_state(self):
        return deepcopy(self.initial_state)
    
    def get_successors(self, state):
        """Generate all possible moves from current state."""
        successors = []
        
        # Find position of empty cell (0)
        empty_pos = None
        for i in range(3):
            for j in range(3):
                if state[i][j] == 0:
                    empty_pos = (i, j)
                    break
            if empty_pos:
                break
        
        if not empty_pos:
            return successors
        
        i, j = empty_pos
        
        # Possible moves: up, down, left, right
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        for di, dj in moves:
            ni, nj = i + di, j + dj
            
            if 0 <= ni < 3 and 0 <= nj < 3:
                # Create new state
                new_state = deepcopy(state)
                new_state[i][j] = new_state[ni][nj]
                new_state[ni][nj] = 0
                
                successors.append((new_state, 1.0))  # Cost of 1 for each move
        
        return successors
    
    def is_goal(self, state):
        """Check if state matches goal state."""
        return state == self.goal_state
    
    def heuristic(self, state):
        """Manhattan distance heuristic."""
        total_distance = 0
        
        for i in range(3):
            for j in range(3):
                if state[i][j] != 0:  # Don't count empty cell
                    # Find where this number should be
                    target_i, target_j = self._find_position(self.goal_state, state[i][j])
                    distance = abs(i - target_i) + abs(j - target_j)
                    total_distance += distance
        
        return total_distance
    
    def _find_position(self, state, value):
        """Find position of a value in state."""
        for i in range(3):
            for j in range(3):
                if state[i][j] == value:
                    return i, j
        return None


# Example Problem: Word Prediction
class WordPredictionProblem(Problem):
    """Simple word prediction problem for beam search."""
    
    def __init__(self, vocabulary, bigram_model, start_word="<START>"):
        self.vocabulary = vocabulary
        self.bigram_model = bigram_model  # dict of (word1, word2) -> probability
        self.start_word = start_word
        self.max_length = 10
    
    def get_initial_state(self):
        return [self.start_word]
    
    def get_successors(self, state):
        """Generate possible next words."""
        if len(state) >= self.max_length:
            return []
        
        current_word = state[-1]
        successors = []
        
        for word in self.vocabulary:
            if word != self.start_word:
                prob = self.bigram_model.get((current_word, word), 0.001)
                successors.append((state + [word], -prob))  # Negative for minimization
        
        return successors
    
    def is_goal(self, state):
        """Goal is reached when we have a complete sentence."""
        return len(state) >= 5 and state[-1] == "<END>"
    
    def heuristic(self, state):
        """Heuristic based on sentence length and word probabilities."""
        if len(state) == 1:
            return 0
        
        # Calculate average log probability
        total_log_prob = 0
        for i in range(1, len(state)):
            prev_word, curr_word = state[i-1], state[i]
            prob = self.bigram_model.get((prev_word, curr_word), 0.001)
            total_log_prob += prob
        
        return -total_log_prob / len(state)  # Negative for minimization


def main():
    """Example usage of beam search."""
    print("=== Beam Search Algorithm Demo ===\n")
    
    # Example 1: 8-Puzzle
    print("1. Solving 8-Puzzle Problem:")
    puzzle = EightPuzzleProblem()
    beam_search = BeamSearch(beam_width=3, max_iterations=50, verbose=True)
    
    solution = beam_search.search(puzzle)
    if solution:
        print(f"Solution found with {len(solution)} moves!")
    else:
        print("No solution found within iteration limit.")
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Word Prediction
    print("2. Word Prediction Problem:")
    vocabulary = ["the", "cat", "sat", "on", "mat", "dog", "ran", "fast", "<END>"]
    bigram_model = {
        ("<START>", "the"): 0.8,
        ("the", "cat"): 0.6,
        ("the", "dog"): 0.4,
        ("cat", "sat"): 0.7,
        ("dog", "ran"): 0.8,
        ("sat", "on"): 0.9,
        ("ran", "fast"): 0.7,
        ("on", "mat"): 0.8,
        ("fast", "<END>"): 0.9,
        ("mat", "<END>"): 0.7,
    }
    
    word_problem = WordPredictionProblem(vocabulary, bigram_model)
    beam_search = BeamSearch(beam_width=2, max_iterations=20, verbose=True)
    
    solution = beam_search.search(word_problem)
    if solution:
        # solution is a list of states, each state is a list of words
        # Take the last state (final sentence)
        final_state = solution[-1] if solution else []
        print(f"Generated sentence: {' '.join(final_state)}")
    else:
        print("No complete sentence generated.")


if __name__ == "__main__":
    main() 