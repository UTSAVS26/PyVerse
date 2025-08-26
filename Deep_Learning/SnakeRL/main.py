#!/usr/bin/env python3
"""
Snake Game AI - Main Training Script

This script provides the main entry point for training and evaluating
reinforcement learning agents on the Snake game environment.
"""

import argparse
import os
import sys
import numpy as np
from typing import Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from environment.snake_env import SnakeEnv
from agents.qlearning_agent import QLearningAgent
from agents.dqn_agent import DQNAgent
from utils.logger import TrainingLogger
from utils.visualization import TrainingVisualizer

def create_agent(agent_type: str, state_size: int, action_size: int, **kwargs) -> Any:
    """
    Create an agent based on the specified type
    
    Args:
        agent_type: Type of agent ('qlearning' or 'dqn')
        state_size: Dimension of state space
        action_size: Number of possible actions
        **kwargs: Additional agent parameters
        
    Returns:
        Initialized agent
    """
    if agent_type.lower() == 'qlearning':
        return QLearningAgent(state_size, action_size, **kwargs)
    elif agent_type.lower() == 'dqn':
        return DQNAgent(state_size, action_size, **kwargs)
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

def train_agent(env: SnakeEnv, agent: Any, episodes: int, 
                render: bool = False, log_interval: int = 100) -> Dict[str, Any]:
    """
    Train an agent on the Snake environment
    
    Args:
        env: Snake environment
        agent: Agent to train
        episodes: Number of episodes to train
        render: Whether to render the game
        log_interval: Interval for logging progress
        
    Returns:
        Dictionary containing training results
    """
    # Setup logging and visualization
    agent_name = agent.__class__.__name__
    logger = TrainingLogger(agent_name=agent_name)
    visualizer = TrainingVisualizer()
    
    # Log training start
    agent_info = agent.get_info()
    logger.start_training(episodes, agent_info)
    
    # Training metrics
    scores = []
    rewards = []
    best_score = 0
    
    # Training loop
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            # Choose action
            action = agent.act(state)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Train agent
            agent.train(state, action, reward, next_state, done)
            
            state = next_state
            total_reward += reward
        
        # Update epsilon
        agent.update_epsilon()
        
        # Record metrics
        score = info['score']
        scores.append(score)
        rewards.append(total_reward)
        best_score = max(best_score, score)
        
        # Calculate average score
        avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        
        # Log progress
        logger.log_episode(episode, score, total_reward, agent.epsilon, avg_score)
        
        # Render occasionally
        if render and episode % log_interval == 0:
            print(f"Episode {episode}: Score={score}, Avg Score={avg_score:.1f}, Epsilon={agent.epsilon:.3f}")
    
    # Log training completion
    final_avg_score = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
    logger.log_training_complete(best_score, final_avg_score)
    
    # Create visualizations
    training_data = logger.get_training_data()
    visualizer.create_summary_plot(training_data, agent_name)
    
    # Close logger
    logger.close()
    
    return {
        'scores': scores,
        'rewards': rewards,
        'best_score': best_score,
        'final_avg_score': final_avg_score,
        'training_data': training_data
    }

def play_game(env: SnakeEnv, agent: Any, episodes: int = 5) -> None:
    """
    Watch a trained agent play the game
    
    Args:
        env: Snake environment
        agent: Trained agent
        episodes: Number of episodes to watch
    """
    print(f"Watching {agent.__class__.__name__} play {episodes} episodes...")
    
    for episode in range(1, episodes + 1):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.act(state)
            state, reward, done, info = env.step(action)
            total_reward += reward
        
        print(f"Episode {episode}: Score={info['score']}, Reward={total_reward:.1f}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Snake Game AI Training')
    parser.add_argument('--mode', choices=['train', 'play'], default='train',
                       help='Mode: train or play')
    parser.add_argument('--agent', choices=['qlearning', 'dqn'], default='dqn',
                       help='Agent type: qlearning or dqn')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of episodes to train')
    parser.add_argument('--grid-size', type=int, default=10,
                       help='Grid size for the game')
    parser.add_argument('--render', action='store_true',
                       help='Render the game during training')
    parser.add_argument('--model-path', type=str,
                       help='Path to load/save model')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for rendering')
    
    args = parser.parse_args()
    
    # Create environment
    env = SnakeEnv(
        grid_size=args.grid_size,
        render_mode="human" if args.render or args.mode == 'play' else "rgb_array",
        fps=args.fps
    )
    
    # Create agent
    state_size = env.get_state_space()
    action_size = env.get_action_space()
    
    if args.agent == 'qlearning':
        agent_params = {
            'learning_rate': 0.1,
            'discount_factor': 0.95,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995
        }
    else:  # DQN
        agent_params = {
            'learning_rate': 0.001,
            'discount_factor': 0.95,
            'memory_size': 10000,
            'batch_size': 32,
            'target_update': 100,
            'epsilon': 1.0,
            'epsilon_min': 0.01,
            'epsilon_decay': 0.995
        }
    
    agent = create_agent(args.agent, state_size, action_size, **agent_params)
    
    # Load model if specified
    if args.model_path and os.path.exists(args.model_path):
        print(f"Loading model from {args.model_path}")
        agent.load(args.model_path)
    
    try:
        if args.mode == 'train':
            # Train agent
            print(f"Training {args.agent.upper()} agent for {args.episodes} episodes...")
            results = train_agent(env, agent, args.episodes, args.render)
            
            # Save model
            if args.model_path:
                dir_name = os.path.dirname(args.model_path)
                if dir_name:
                    os.makedirs(dir_name, exist_ok=True)
                agent.save(args.model_path)
                print(f"Model saved to {args.model_path}")
            
            # Print results
            print(f"\nTraining completed!")
            print(f"Best score: {results['best_score']}")
            print(f"Final average score: {results['final_avg_score']:.2f}")
            
        elif args.mode == 'play':
            # Play game
            if not args.model_path or not os.path.exists(args.model_path):
                print("Error: Model path required for play mode")
                return
            
            play_game(env, agent, episodes=args.episodes)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        env.close()

if __name__ == "__main__":
    main()
