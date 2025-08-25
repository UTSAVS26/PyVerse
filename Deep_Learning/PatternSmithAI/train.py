"""
PatternSmithAI - Training Module
Handles training the AI agent with feedback and reward functions.
"""

import os
import time
import json
import random
import numpy as np
from typing import List, Dict, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime

from canvas import PatternCanvas, ColorPalette
from patterns import PatternFactory
from agent import PatternAgent, PatternEvaluator


class PatternTrainer:
    """Handles training of the pattern generation AI agent."""
    
    def __init__(self, canvas_size: int = 800, output_dir: str = "gallery"):
        self.canvas = PatternCanvas(canvas_size, canvas_size)
        self.color_palette = ColorPalette()
        self.agent = PatternAgent(self.canvas, self.color_palette)
        self.evaluator = PatternEvaluator()
        self.output_dir = output_dir
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Training statistics
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'scores': [],
            'epsilon_values': []
        }
    
    def train_episode(self, steps: int = 15) -> Dict[str, float]:
        """Train the agent for one episode."""
        self.canvas.clear()
        total_reward = 0.0
        
        for step in range(steps):
            # Get current state
            state = self.agent.get_state()
            
            # Select and execute action
            action = self.agent.select_action(state)
            next_state, reward = self.agent.execute_action(action)
            
            # Store experience
            done = (step == steps - 1)
            self.agent.remember(state, action, reward, next_state, done)
            
            # Train the agent
            self.agent.replay()
            
            total_reward += reward
        
        # Evaluate final pattern
        scores = self.evaluator.evaluate_pattern(self.canvas)
        overall_score = self.evaluator.get_overall_score(scores)
        
        return {
            'total_reward': total_reward,
            'overall_score': overall_score,
            'scores': scores,
            'epsilon': self.agent.epsilon
        }
    
    def train(self, episodes: int = 100, steps_per_episode: int = 15, 
              save_interval: int = 10, target_update_interval: int = 5):
        """Train the agent for multiple episodes."""
        print(f"Starting training for {episodes} episodes...")
        print(f"Steps per episode: {steps_per_episode}")
        print(f"Save interval: {save_interval}")
        
        start_time = time.time()
        
        for episode in range(episodes):
            # Train one episode
            episode_stats = self.train_episode(steps_per_episode)
            
            # Store statistics
            self.training_stats['episodes'].append(episode)
            self.training_stats['rewards'].append(episode_stats['total_reward'])
            self.training_stats['scores'].append(episode_stats['overall_score'])
            self.training_stats['epsilon_values'].append(episode_stats['epsilon'])
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_stats['rewards'][-10:])
                avg_score = np.mean(self.training_stats['scores'][-10:])
                print(f"Episode {episode + 1}/{episodes} - "
                      f"Avg Reward: {avg_reward:.3f}, "
                      f"Avg Score: {avg_score:.3f}, "
                      f"Epsilon: {episode_stats['epsilon']:.3f}")
            
            # Save pattern periodically
            if (episode + 1) % save_interval == 0:
                self._save_training_pattern(episode + 1)
            
            # Update target network periodically
            if (episode + 1) % target_update_interval == 0:
                self.agent.update_target_network()
        
        training_time = time.time() - start_time
        print(f"Training completed in {training_time:.2f} seconds")
        
        # Save final model and statistics
        self._save_training_results()
    
    def _save_training_pattern(self, episode: int):
        """Save the current pattern from training."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.output_dir}/training_pattern_episode_{episode}_{timestamp}.png"
        self.canvas.save(filename)
    
    def _save_training_results(self):
        """Save training results and statistics."""
        # Save model
        model_filename = f"{self.output_dir}/trained_model.pth"
        self.agent.save_model(model_filename)
        
        # Save training statistics
        stats_filename = f"{self.output_dir}/training_stats.json"
        with open(stats_filename, 'w') as f:
            json.dump(self.training_stats, f, indent=2)
        
        # Create training plots
        self._create_training_plots()
        
        print(f"Training results saved to {self.output_dir}")
    
    def _create_training_plots(self):
        """Create plots of training progress."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Reward plot
        axes[0, 0].plot(self.training_stats['episodes'], self.training_stats['rewards'])
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')
        
        # Score plot
        axes[0, 1].plot(self.training_stats['episodes'], self.training_stats['scores'])
        axes[0, 1].set_title('Pattern Quality Scores')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Overall Score')
        
        # Epsilon plot
        axes[1, 0].plot(self.training_stats['episodes'], self.training_stats['epsilon_values'])
        axes[1, 0].set_title('Exploration Rate (Epsilon)')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Epsilon')
        
        # Moving average reward
        window = 10
        if len(self.training_stats['rewards']) >= window:
            moving_avg = np.convolve(self.training_stats['rewards'], 
                                   np.ones(window)/window, mode='valid')
            episodes_avg = self.training_stats['episodes'][window-1:]
            axes[1, 1].plot(episodes_avg, moving_avg)
            axes[1, 1].set_title(f'Moving Average Reward (window={window})')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Average Reward')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_dir}/training_plots.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def generate_sample_patterns(self, count: int = 10):
        """Generate sample patterns using the trained agent."""
        print(f"Generating {count} sample patterns...")
        
        for i in range(count):
            # Generate pattern
            self.agent.generate_pattern(steps=20)
            
            # Evaluate pattern
            scores = self.evaluator.evaluate_pattern(self.canvas)
            overall_score = self.evaluator.get_overall_score(scores)
            
            # Save pattern
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.output_dir}/sample_pattern_{i+1}_{timestamp}.png"
            self.canvas.save(filename)
            
            print(f"Generated pattern {i+1}: Score = {overall_score:.3f}")
            
            # Save metadata
            metadata = {
                'pattern_id': i + 1,
                'timestamp': timestamp,
                'scores': scores,
                'overall_score': overall_score,
                'agent_stats': self.agent.get_training_stats()
            }
            
            metadata_filename = f"{self.output_dir}/sample_pattern_{i+1}_{timestamp}_metadata.json"
            with open(metadata_filename, 'w') as f:
                json.dump(metadata, f, indent=2)


class InteractiveTrainer(PatternTrainer):
    """Interactive trainer that allows user feedback during training."""
    
    def __init__(self, canvas_size: int = 800, output_dir: str = "gallery"):
        super().__init__(canvas_size, output_dir)
        self.user_feedback_history = []
    
    def get_user_feedback(self, pattern_id: int) -> float:
        """Get user feedback for a pattern (0-1 scale)."""
        print(f"\nPattern {pattern_id} generated!")
        print("Please rate this pattern from 0 (worst) to 1 (best):")
        
        while True:
            try:
                feedback = float(input("Rating (0-1): "))
                if 0 <= feedback <= 1:
                    return feedback
                else:
                    print("Please enter a value between 0 and 1")
            except ValueError:
                print("Please enter a valid number")
    
    def train_with_feedback(self, episodes: int = 20, steps_per_episode: int = 15):
        """Train the agent with user feedback."""
        print("Starting interactive training...")
        print("You will be asked to rate patterns during training.")
        
        for episode in range(episodes):
            # Train one episode
            episode_stats = self.train_episode(steps_per_episode)
            
            # Get user feedback
            user_rating = self.get_user_feedback(episode + 1)
            
            # Store feedback
            feedback_data = {
                'episode': episode + 1,
                'user_rating': user_rating,
                'agent_score': episode_stats['overall_score'],
                'timestamp': datetime.now().isoformat()
            }
            self.user_feedback_history.append(feedback_data)
            
            # Adjust agent based on feedback
            self._adjust_agent_with_feedback(user_rating, episode_stats)
            
            # Save pattern
            self._save_training_pattern(episode + 1)
            
            print(f"Episode {episode + 1} completed. "
                  f"Your rating: {user_rating:.2f}, "
                  f"Agent score: {episode_stats['overall_score']:.3f}")
        
        # Save feedback history
        feedback_filename = f"{self.output_dir}/user_feedback_history.json"
        with open(feedback_filename, 'w') as f:
            json.dump(self.user_feedback_history, f, indent=2)
        
        print("Interactive training completed!")
    
    def _adjust_agent_with_feedback(self, user_rating: float, episode_stats: Dict[str, float]):
        """Adjust agent parameters based on user feedback."""
        # Simple adjustment: if user rating is much different from agent score,
        # adjust the reward calculation
        agent_score = episode_stats['overall_score']
        feedback_difference = abs(user_rating - agent_score)
        
        if feedback_difference >= 0.3:  # Significant difference
            # Adjust exploration rate
            if user_rating > agent_score:
                # User liked it more than agent expected - increase exploration
                self.agent.epsilon = min(1.0, self.agent.epsilon * 1.1)
            else:
                # User liked it less than agent expected - decrease exploration
                self.agent.epsilon = max(self.agent.epsilon_min, self.agent.epsilon * 0.9)


def main():
    """Main training function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Train PatternSmithAI agent")
    parser.add_argument("--episodes", type=int, default=100, help="Number of training episodes")
    parser.add_argument("--steps", type=int, default=15, help="Steps per episode")
    parser.add_argument("--interactive", action="store_true", help="Use interactive training")
    parser.add_argument("--output-dir", type=str, default="gallery", help="Output directory")
    parser.add_argument("--canvas-size", type=int, default=800, help="Canvas size")
    
    args = parser.parse_args()
    
    if args.interactive:
        trainer = InteractiveTrainer(args.canvas_size, args.output_dir)
        trainer.train_with_feedback(args.episodes, args.steps)
    else:
        trainer = PatternTrainer(args.canvas_size, args.output_dir)
        trainer.train(args.episodes, args.steps)
    
    # Generate sample patterns
    trainer.generate_sample_patterns(10)


if __name__ == "__main__":
    main()
