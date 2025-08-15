"""
Evaluation module for trained agents.
Tests agents against random and rule-based opponents.
"""

import numpy as np
import random
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.tictactoe_env import TicTacToeEnv
from agents.q_learning_agent import QLearningAgent
from agents.monte_carlo_agent import MonteCarloAgent
from utils.state_utils import check_winner, get_valid_moves


class RandomAgent:
    """Simple random agent for evaluation."""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
    
    def select_action(self, board: np.ndarray, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        return random.choice(valid_moves)


class MinimaxAgent:
    """Minimax agent for evaluation."""
    
    def __init__(self, player_id: int, max_depth: int = 9):
        self.player_id = player_id
        self.max_depth = max_depth
    
    def select_action(self, board: np.ndarray, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        best_score = float('-inf')
        best_action = None
        
        for action in valid_moves:
            # Make move
            board[action[0], action[1]] = self.player_id
            
            # Get score
            score = self._minimax(board, self.max_depth, False, float('-inf'), float('inf'))
            
            # Undo move
            board[action[0], action[1]] = 0
            
            if score > best_score:
                best_score = score
                best_action = action
        
        return best_action
    
    def _minimax(self, board: np.ndarray, depth: int, is_maximizing: bool, 
                 alpha: float, beta: float) -> float:
        """Minimax algorithm with alpha-beta pruning."""
        winner = check_winner(board)
        
        if winner == self.player_id:
            return 1.0
        elif winner == 3 - self.player_id:
            return -1.0
        elif winner == 0 or depth == 0:
            return 0.0
        
        valid_moves = get_valid_moves(board)
        
        if is_maximizing:
            max_score = float('-inf')
            for action in valid_moves:
                board[action[0], action[1]] = self.player_id
                score = self._minimax(board, depth - 1, False, alpha, beta)
                board[action[0], action[1]] = 0
                max_score = max(max_score, score)
                alpha = max(alpha, score)
                if beta <= alpha:
                    break
            return max_score
        else:
            min_score = float('inf')
            for action in valid_moves:
                board[action[0], action[1]] = 3 - self.player_id
                score = self._minimax(board, depth - 1, True, alpha, beta)
                board[action[0], action[1]] = 0
                min_score = min(min_score, score)
                beta = min(beta, score)
                if beta <= alpha:
                    break
            return min_score


def play_game(agent1, agent2, env: TicTacToeEnv) -> int:
    """
    Play a single game between two agents.
    
    Args:
        agent1: First agent (player 1)
        agent2: Second agent (player 2)
        env: Game environment
        
    Returns:
        Winner (1, 2, or 0 for draw)
    """
    env.reset()
    
    while not env.is_terminal():
        current_agent = agent1 if env.current_player == 1 else agent2
        valid_moves = env.get_valid_actions()
        
        if not valid_moves:
            break
        
        action = current_agent.select_action(env.get_observation(), valid_moves)
        env.step(action)
    
    return env.get_winner()


def evaluate_agent_vs_random(agent, num_games: int = 1000) -> Tuple[float, float, float]:
    """
    Evaluate agent against random agent.
    
    Args:
        agent: Agent to evaluate
        num_games: Number of games to play
        
    Returns:
        (win_rate, draw_rate, loss_rate) tuple
    """
    env = TicTacToeEnv()
    random_agent = RandomAgent(3 - agent.get_player_id())
    
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(num_games):
        if agent.get_player_id() == 1:
            winner = play_game(agent, random_agent, env)
        else:
            winner = play_game(random_agent, agent, env)
        
        if winner == agent.get_player_id():
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1
    
    win_rate = wins / num_games
    draw_rate = draws / num_games
    loss_rate = losses / num_games
    
    return win_rate, draw_rate, loss_rate


def evaluate_agent_vs_minimax(agent, num_games: int = 100) -> Tuple[float, float, float]:
    """
    Evaluate agent against minimax agent.
    
    Args:
        agent: Agent to evaluate
        num_games: Number of games to play
        
    Returns:
        (win_rate, draw_rate, loss_rate) tuple
    """
    env = TicTacToeEnv()
    minimax_agent = MinimaxAgent(3 - agent.get_player_id())
    
    wins = 0
    draws = 0
    losses = 0
    
    for _ in range(num_games):
        if agent.get_player_id() == 1:
            winner = play_game(agent, minimax_agent, env)
        else:
            winner = play_game(minimax_agent, agent, env)
        
        if winner == agent.get_player_id():
            wins += 1
        elif winner == 0:
            draws += 1
        else:
            losses += 1
    
    win_rate = wins / num_games
    draw_rate = draws / num_games
    loss_rate = losses / num_games
    
    return win_rate, draw_rate, loss_rate


def evaluate_agent(agent_file: str, agent_type: str = 'q_learning') -> None:
    """
    Evaluate a trained agent.
    
    Args:
        agent_file: Path to the trained agent file
        agent_type: Type of agent ('q_learning' or 'monte_carlo')
    """
    print(f"Evaluating {agent_type} agent from {agent_file}...")
    
    # Load agent
    if agent_type == 'q_learning':
        agent = QLearningAgent(player_id=1)
    elif agent_type == 'monte_carlo':
        agent = MonteCarloAgent(player_id=1)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    agent.load(agent_file)
    agent.set_training_mode(False)
    
    # Evaluate against random agent
    print("\n=== Evaluation vs Random Agent ===")
    win_rate, draw_rate, loss_rate = evaluate_agent_vs_random(agent, num_games=1000)
    print(f"Win rate: {win_rate:.1%}")
    print(f"Draw rate: {draw_rate:.1%}")
    print(f"Loss rate: {loss_rate:.1%}")
    
    # Evaluate against minimax agent
    print("\n=== Evaluation vs Minimax Agent ===")
    win_rate, draw_rate, loss_rate = evaluate_agent_vs_minimax(agent, num_games=100)
    print(f"Win rate: {win_rate:.1%}")
    print(f"Draw rate: {draw_rate:.1%}")
    print(f"Loss rate: {loss_rate:.1%}")
    
    # Print agent statistics
    print(f"\n=== Agent Statistics ===")
    print(f"Q-table size: {agent.get_q_table_size()} states")
    print(f"Total Q-values: {agent.get_total_q_values()}")


if __name__ == "__main__":
    # Example usage
    try:
        evaluate_agent('trained_q_agent.pkl', 'q_learning')
    except FileNotFoundError:
        print("No trained agent found. Please train an agent first.")
        print("Run: python training/self_play_qlearn.py") 