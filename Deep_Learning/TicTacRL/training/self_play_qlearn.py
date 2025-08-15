"""
Self-play training for Q-learning agent.
Trains agent by playing against itself and evaluates against random agents.
"""

import numpy as np
import random
import matplotlib.pyplot as plt
from typing import List, Tuple
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.tictactoe_env import TicTacToeEnv
from agents.q_learning_agent import QLearningAgent
from utils.state_utils import encode_board_state, get_reward


class RandomAgent:
    """Simple random agent for evaluation."""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
    
    def select_action(self, board: np.ndarray, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        return random.choice(valid_moves)


def play_game(agent1, agent2, env: TicTacToeEnv, training_mode: bool = True) -> Tuple[int, List]:
    """
    Play a single game between two agents.
    
    Args:
        agent1: First agent (player 1)
        agent2: Second agent (player 2)
        env: Game environment
        training_mode: Whether to update agents during play
        
    Returns:
        (winner, game_history) tuple
    """
    env.reset()
    game_history = []
    
    while not env.is_terminal():
        current_agent = agent1 if env.current_player == 1 else agent2
        valid_moves = env.get_valid_actions()
        
        if not valid_moves:
            break
        
        # Select action
        action = current_agent.select_action(env.get_observation(), valid_moves)
        
        # Store state before action
        state = env.get_state()
        
        # Take action
        next_board, reward, done, info = env.step(action)
        next_state = env.get_state()
        
        # Update agent if in training mode
        if training_mode and hasattr(current_agent, 'update'):
            current_agent.update(state, action, reward, next_state, done)
        
        # Store experience for Monte Carlo if needed
        if hasattr(current_agent, 'update_episode'):
            game_history.append((state, action, reward))
    
    winner = env.get_winner()
    return winner, game_history


def evaluate_agent(agent, num_games: int = 100) -> Tuple[float, float, float]:
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
        # Play as both X and O
        if agent.get_player_id() == 1:
            winner, _ = play_game(agent, random_agent, env, training_mode=False)
        else:
            winner, _ = play_game(random_agent, agent, env, training_mode=False)
        
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


def train_q_learning_agent(num_episodes: int = 10000, eval_interval: int = 1000) -> QLearningAgent:
    """
    Train Q-learning agent through self-play.
    
    Args:
        num_episodes: Number of training episodes
        eval_interval: Interval for evaluation
        
    Returns:
        Trained Q-learning agent
    """
    print("Training Q-learning agent...")
    
    # Create agents
    agent1 = QLearningAgent(player_id=1, epsilon=0.1)
    agent2 = QLearningAgent(player_id=2, epsilon=0.1)
    env = TicTacToeEnv()
    
    # Training history
    win_rates = []
    episode_numbers = []
    
    for episode in range(1, num_episodes + 1):
        # Self-play training
        winner, _ = play_game(agent1, agent2, env, training_mode=True)
        
        # Evaluation
        if episode % eval_interval == 0:
            win_rate, draw_rate, loss_rate = evaluate_agent(agent1)
            win_rates.append(win_rate)
            episode_numbers.append(episode)
            
            print(f"Episode: {episode:,} | Win rate vs random: {win_rate:.1%} | "
                  f"Q-table size: {agent1.get_q_table_size()}")
        
        # Decay epsilon
        if episode % 1000 == 0:
            new_epsilon = max(0.01, agent1.get_epsilon() * 0.95)
            agent1.set_epsilon(new_epsilon)
            agent2.set_epsilon(new_epsilon)
    
    # Final evaluation
    final_win_rate, final_draw_rate, final_loss_rate = evaluate_agent(agent1, num_games=1000)
    print(f"\nFinal Results:")
    print(f"Win rate: {final_win_rate:.1%}")
    print(f"Draw rate: {final_draw_rate:.1%}")
    print(f"Loss rate: {final_loss_rate:.1%}")
    print(f"Final Q-table size: {agent1.get_q_table_size()} states")
    print(f"Total Q-values: {agent1.get_total_q_values()}")
    
    # Plot training progress
    if win_rates:
        plt.figure(figsize=(10, 6))
        plt.plot(episode_numbers, win_rates)
        plt.title('Q-Learning Training Progress')
        plt.xlabel('Episode')
        plt.ylabel('Win Rate vs Random Agent')
        plt.grid(True)
        plt.savefig('q_learning_training.png')
        plt.close()
    
    return agent1


if __name__ == "__main__":
    # Train the agent
    trained_agent = train_q_learning_agent(num_episodes=20000, eval_interval=1000)
    
    # Save the trained agent
    trained_agent.save('trained_q_agent.pkl')
    print("Trained agent saved to 'trained_q_agent.pkl'") 