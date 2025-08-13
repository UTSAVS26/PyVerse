import sys
import os
import matplotlib.pyplot as plt
import numpy as np
from typing import List, Dict, Any, Tuple
import time

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.flappy_bird import FlappyBirdGame
from agents.dqn_agent import DQNAgent

class DQNTrainer:
    """Trainer for DQN agents."""
    
    def __init__(self, max_episodes: int = 1000, max_fitness: float = 1000,
                 render_training: bool = False, save_interval: int = 100):
        self.max_episodes = max_episodes
        self.max_fitness = max_fitness
        self.render_training = render_training
        self.save_interval = save_interval
        
        # Training statistics
        self.episode_rewards = []
        self.episode_scores = []
        self.epsilon_history = []
        self.avg_rewards = []
        self.avg_scores = []
        
        # Create game and agent
        self.game = FlappyBirdGame(headless=not render_training)
        self.agent = DQNAgent()
        
        # Best agent tracking
        self.best_score = 0
        self.best_episode = 0
        
    def train(self) -> DQNAgent:
        """Train the DQN agent."""
        print(f"Starting DQN training for {self.max_episodes} episodes...")
        print(f"Initial epsilon: {self.agent.epsilon}")
        print(f"Learning rate: {self.agent.learning_rate}")
        print("-" * 50)
        
        start_time = time.time()
        
        for episode in range(self.max_episodes):
            # Reset episode
            state = self.game.reset()
            self.agent.reset_episode()
            
            episode_reward = 0
            episode_score = 0
            done = False
            
            # Run episode
            while not done:
                # Get action from agent
                action = self.agent.get_action(state)
                
                # Take step in game
                next_state, reward, done, info = self.game.step(action)
                
                # Update agent
                self.agent.update(state, action, reward, next_state, done)
                
                # Update episode statistics
                episode_reward += reward
                episode_score = info.get('score', 0)
                
                # Render if requested
                if self.render_training:
                    self.game.render()
                    
                state = next_state
                
            # Update best score
            if episode_score > self.best_score:
                self.best_score = episode_score
                self.best_episode = episode
                
            # Store episode statistics
            self.episode_rewards.append(episode_reward)
            self.episode_scores.append(episode_score)
            self.epsilon_history.append(self.agent.epsilon)
            
            # Calculate running averages
            if episode >= 99:
                avg_reward = np.mean(self.episode_rewards[-100:])
                avg_score = np.mean(self.episode_scores[-100:])
                self.avg_rewards.append(avg_reward)
                self.avg_scores.append(avg_score)
            else:
                avg_reward = np.mean(self.episode_rewards)
                avg_score = np.mean(self.episode_scores)
                self.avg_rewards.append(avg_reward)
                self.avg_scores.append(avg_score)
                
            # Print progress
            if (episode + 1) % 50 == 0:
                print(f"Episode {episode + 1}/{self.max_episodes}")
                print(f"  Score: {episode_score}, Reward: {episode_reward:.1f}")
                print(f"  Avg Score (last 100): {avg_score:.1f}")
                print(f"  Avg Reward (last 100): {avg_reward:.1f}")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                print(f"  Best Score: {self.best_score} (episode {self.best_episode})")
                print()
                
            # Save agent periodically
            if (episode + 1) % self.save_interval == 0:
                self._save_agent(episode + 1)
                
            # Check for convergence
            if avg_score >= self.max_fitness and episode >= 100:
                print(f"\nConvergence reached! Avg score: {avg_score}")
                break
                
        # Training complete
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.1f} seconds")
        print(f"Best score achieved: {self.best_score} (episode {self.best_episode})")
        
        # Save final agent
        self._save_agent("final")
        
        # Plot training progress
        self._plot_training_progress()
        
        return self.agent
        
    def _save_agent(self, episode: str) -> None:
        """Save the agent to file."""
        filename = f"best_dqn_ep_{episode}.pth"
        filepath = os.path.join("models", filename)
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        self.agent.save(filepath)
        print(f"  Saved agent to {filepath}")
        
    def _plot_training_progress(self) -> None:
        """Plot training progress."""
        if not self.episode_rewards:
            return
            
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(1, len(self.episode_rewards) + 1)
        
        # Episode rewards
        ax1.plot(episodes, self.episode_rewards, 'b-', alpha=0.6, label='Episode Reward')
        ax1.plot(episodes, self.avg_rewards, 'r-', linewidth=2, label='Avg Reward (100 ep)')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.set_title('DQN Training Progress - Rewards')
        ax1.legend()
        ax1.grid(True)
        
        # Episode scores
        ax2.plot(episodes, self.episode_scores, 'g-', alpha=0.6, label='Episode Score')
        ax2.plot(episodes, self.avg_scores, 'r-', linewidth=2, label='Avg Score (100 ep)')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Score')
        ax2.set_title('DQN Training Progress - Scores')
        ax2.legend()
        ax2.grid(True)
        
        # Epsilon decay
        ax3.plot(episodes, self.epsilon_history, 'purple', linewidth=2)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Epsilon')
        ax3.set_title('DQN Training Progress - Epsilon Decay')
        ax3.grid(True)
        
        # Memory size
        memory_sizes = [len(self.agent.memory)] * len(episodes)
        ax4.plot(episodes, memory_sizes, 'orange', linewidth=2)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Memory Size')
        ax4.set_title('DQN Training Progress - Memory Size')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('dqn_training_progress.png', dpi=300, bbox_inches='tight')
        plt.show()
        
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        return {
            'episodes_completed': len(self.episode_rewards),
            'best_score': self.best_score,
            'best_episode': self.best_episode,
            'final_avg_reward': self.avg_rewards[-1] if self.avg_rewards else 0,
            'final_avg_score': self.avg_scores[-1] if self.avg_scores else 0,
            'final_epsilon': self.agent.epsilon,
            'reward_history': self.episode_rewards,
            'score_history': self.episode_scores,
            'epsilon_history': self.epsilon_history,
            'avg_reward_history': self.avg_rewards,
            'avg_score_history': self.avg_scores
        }

def train_dqn(episodes: int = 1000, learning_rate: float = 0.001,
              epsilon: float = 1.0, epsilon_decay: float = 0.995,
              render: bool = False, save_interval: int = 100) -> DQNAgent:
    """
    Train a DQN agent.
    
    Args:
        episodes: Number of episodes to train
        learning_rate: Learning rate for the optimizer
        epsilon: Initial exploration rate
        epsilon_decay: Epsilon decay rate
        render: Whether to render training
        save_interval: How often to save the agent
        
    Returns:
        agent: The trained agent
    """
    trainer = DQNTrainer(
        max_episodes=episodes,
        render_training=render,
        save_interval=save_interval
    )
    
    # Update agent hyperparameters
    trainer.agent.learning_rate = learning_rate
    trainer.agent.epsilon = epsilon
    trainer.agent.epsilon_decay = epsilon_decay
    
    return trainer.train()

def main():
    """Main function for DQN training."""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Train DQN agent for FlapAI')
    parser.add_argument('--episodes', type=int, default=1000, 
                       help='Number of episodes to train')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--epsilon', type=float, default=1.0,
                       help='Initial epsilon')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                       help='Epsilon decay rate')
    parser.add_argument('--render', action='store_true',
                       help='Render training')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='Save interval for agent')
    
    args = parser.parse_args()
    
    # Train the agent
    agent = train_dqn(
        episodes=args.episodes,
        learning_rate=args.learning_rate,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        render=args.render,
        save_interval=args.save_interval
    )
    
    print(f"\nTraining complete! Agent saved.")
    print(f"Best score achieved: {agent.best_score}")

if __name__ == "__main__":
    main() 