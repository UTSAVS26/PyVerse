"""
User interface for playing against trained AI agents.
Provides a command-line interface for human vs AI gameplay.
"""

import numpy as np
import os
from typing import Tuple, List
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.tictactoe_env import TicTacToeEnv
from agents.q_learning_agent import QLearningAgent
from agents.monte_carlo_agent import MonteCarloAgent
from utils.state_utils import print_board, get_2d_position


class HumanPlayer:
    """Human player interface."""
    
    def __init__(self, player_id: int):
        self.player_id = player_id
    
    def select_action(self, board: np.ndarray, valid_moves: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Get human player's move."""
        while True:
            try:
                print("\nEnter your move (0-8):")
                print("Board positions:")
                print(" 0 | 1 | 2 ")
                print("---+---+---")
                print(" 3 | 4 | 5 ")
                print("---+---+---")
                print(" 6 | 7 | 8 ")
                
                move = int(input("Move: "))
                if 0 <= move <= 8:
                    row, col = get_2d_position(move)
                    if (row, col) in valid_moves:
                        return (row, col)
                    else:
                        print("Invalid move! Position already taken.")
                else:
                    print("Invalid input! Enter a number between 0 and 8.")
            except ValueError:
                print("Invalid input! Enter a number between 0 and 8.")


def play_human_vs_ai(agent_file: str, agent_type: str = 'q_learning', human_first: bool = True) -> None:
    """
    Play a game between human and AI.
    
    Args:
        agent_file: Path to the trained agent file
        agent_type: Type of agent ('q_learning' or 'monte_carlo')
        human_first: Whether human plays first (X)
    """
    # Load AI agent
    if agent_type == 'q_learning':
        ai_agent = QLearningAgent(player_id=2 if human_first else 1)
    elif agent_type == 'monte_carlo':
        ai_agent = MonteCarloAgent(player_id=2 if human_first else 1)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
    
    try:
        ai_agent.load(agent_file)
        ai_agent.set_training_mode(False)
    except FileNotFoundError:
        print(f"Agent file {agent_file} not found!")
        print("Please train an agent first using:")
        print("  python training/self_play_qlearn.py")
        print("  or")
        print("  python training/self_play_mc.py")
        return
    
    # Create human player
    human_player = HumanPlayer(player_id=1 if human_first else 2)
    
    # Create environment
    env = TicTacToeEnv()
    
    print(f"\n🎮 TicTacRL - Human vs AI")
    print(f"AI Agent: {agent_type.upper()}")
    print(f"You play as: {'X (first)' if human_first else 'O (second)'}")
    print(f"AI plays as: {'O (second)' if human_first else 'X (first)'}")
    print("\n" + "="*40)
    
    # Game loop
    while not env.is_terminal():
        env.render()
        
        current_player = human_player if env.current_player == (1 if human_first else 2) else ai_agent
        valid_moves = env.get_valid_actions()
        
        if not valid_moves:
            break
        
        # Get move
        if current_player == human_player:
            action = human_player.select_action(env.get_observation(), valid_moves)
            print(f"Your move: {action}")
        else:
            action = ai_agent.select_action(env.get_observation(), valid_moves)
            print(f"AI move: {action}")
        
        # Make move
        env.step(action)
    
    # Game result
    env.render()
    winner = env.get_winner()
    
    print("\n" + "="*40)
    if winner == 0:
        print("🎯 Result: Draw!")
    elif (winner == 1 and human_first) or (winner == 2 and not human_first):
        print("🎉 Result: You win!")
    else:
        print("🤖 Result: AI wins!")
    print("="*40)


def main():
    """Main function for the UI."""
    print("🎮 Welcome to TicTacRL!")
    print("Play against trained AI agents.")
    
    # Check for available agents
    available_agents = []
    if os.path.exists('trained_q_agent.pkl'):
        available_agents.append(('q_learning', 'trained_q_agent.pkl'))
    if os.path.exists('trained_mc_agent.pkl'):
        available_agents.append(('monte_carlo', 'trained_mc_agent.pkl'))
    
    if not available_agents:
        print("\n❌ No trained agents found!")
        print("Please train an agent first:")
        print("  python training/self_play_qlearn.py")
        print("  or")
        print("  python training/self_play_mc.py")
        return
    
    # Select agent
    print("\nAvailable agents:")
    for i, (agent_type, agent_file) in enumerate(available_agents):
        print(f"  {i+1}. {agent_type.upper()} agent ({agent_file})")
    
    while True:
        try:
            choice = int(input("\nSelect agent (1-{}): ".format(len(available_agents))))
            if 1 <= choice <= len(available_agents):
                agent_type, agent_file = available_agents[choice - 1]
                break
            else:
                print("Invalid choice!")
        except ValueError:
            print("Invalid input!")
    
    # Select who goes first
    print("\nWho goes first?")
    print("  1. Human (X)")
    print("  2. AI (X)")
    
    while True:
        try:
            first_choice = int(input("Choice (1-2): "))
            if first_choice in [1, 2]:
                human_first = (first_choice == 1)
                break
            else:
                print("Invalid choice!")
        except ValueError:
            print("Invalid input!")
    
    # Play game
    play_human_vs_ai(agent_file, agent_type, human_first)
    
    # Play again option
    while True:
        play_again = input("\nPlay again? (y/n): ").lower()
        if play_again in ['y', 'yes']:
            play_human_vs_ai(agent_file, agent_type, human_first)
        elif play_again in ['n', 'no']:
            print("Thanks for playing! 👋")
            break
        else:
            print("Please enter 'y' or 'n'")


if __name__ == "__main__":
    main() 