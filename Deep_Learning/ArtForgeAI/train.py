"""
Training module for ArtForgeAI - Training loop with reward mechanisms
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import time
from typing import Dict, List, Tuple, Any
import json
from datetime import datetime

from agent import PaintingAgent, PaintingEnvironment


class ArtForgeTrainer:
    """Trainer for the ArtForgeAI painting agent"""
    
    def __init__(self, canvas_width: int = 800, canvas_height: int = 600, 
                 max_strokes: int = 50, save_dir: str = "gallery"):
        """
        Initialize the trainer
        
        Args:
            canvas_width: Width of the canvas
            canvas_height: Height of the canvas
            max_strokes: Maximum strokes per episode
            save_dir: Directory to save artworks and models
        """
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.max_strokes = max_strokes
        self.save_dir = save_dir
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "artworks"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "models"), exist_ok=True)
        
        # Initialize environment and agent
        self.env = PaintingEnvironment(canvas_width, canvas_height, max_strokes)
        
        # Calculate state and action dimensions
        # State: flattened canvas image + metadata (coverage, color_diversity, stroke_count)
        state_dim = canvas_width * canvas_height * 3 + 3
        # Action: [stroke_type, x, y, angle, color_r, color_g, color_b, thickness]
        action_dim = 8
        
        self.agent = PaintingAgent(state_dim, action_dim)
        
        # Training statistics
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_coverages = []
        self.episode_color_diversities = []
        self.training_losses = []
        
        # Best performance tracking
        self.best_reward = float('-inf')
        self.best_coverage = 0.0
        
    def train_episode(self, render: bool = False, save_artwork: bool = False) -> Dict[str, Any]:
        """
        Train for one episode
        
        Args:
            render: Whether to display the canvas during training
            save_artwork: Whether to save the final artwork
        
        Returns:
            Episode statistics
        """
        state = self.env.reset()
        episode_reward = 0.0
        episode_length = 0
        
        # Store intermediate states for visualization
        intermediate_states = []
        
        for step in range(self.max_strokes):
            # Select action
            action = self.agent.select_action(state, add_noise=True)
            
            # Take step in environment
            next_state, reward, done, info = self.env.step(action)
            
            # Store transition
            self.agent.store_transition(state, action, reward, next_state, done)
            
            # Train agent
            if len(self.agent.memory) > 64:
                loss_info = self.agent.train(batch_size=64)
                if loss_info:
                    self.training_losses.append(loss_info)
            
            # Update state and statistics
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Store intermediate state for visualization
            if step % 10 == 0:  # Every 10 strokes
                intermediate_states.append(self.env.canvas.get_image().copy())
            
            # Render if requested
            if render and step % 5 == 0:
                self.env.canvas.display(f"Training - Step {step}")
                plt.pause(0.1)
            
            if done:
                break
        
        # Get final statistics
        final_coverage = self.env.canvas.get_coverage()
        final_color_diversity = self.env.canvas.get_color_diversity()
        
        # Save artwork if requested
        if save_artwork:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            artwork_path = os.path.join(self.save_dir, "artworks", f"episode_{len(self.episode_rewards)}_{timestamp}.png")
            self.env.canvas.save_image(artwork_path)
            
            # Save intermediate states as GIF (if matplotlib supports it)
            if len(intermediate_states) > 1:
                self._save_intermediate_states(intermediate_states, len(self.episode_rewards))
        
        # Update statistics
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)
        self.episode_coverages.append(final_coverage)
        self.episode_color_diversities.append(final_color_diversity)
        
        # Check for best performance
        if episode_reward > self.best_reward:
            self.best_reward = episode_reward
            self._save_best_model("best_reward")
        
        if final_coverage > self.best_coverage:
            self.best_coverage = final_coverage
            self._save_best_model("best_coverage")
        
        return {
            'episode': len(self.episode_rewards),
            'reward': episode_reward,
            'length': episode_length,
            'coverage': final_coverage,
            'color_diversity': final_color_diversity,
            'stroke_types': [stroke['type'] for stroke in self.env.canvas.stroke_history]
        }
    
    def _save_intermediate_states(self, states: List[np.ndarray], episode_num: int):
        """Save intermediate states as individual images"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, state in enumerate(states):
            img_path = os.path.join(self.save_dir, "artworks", 
                                   f"episode_{episode_num}_step_{i*10}_{timestamp}.png")
            from PIL import Image
            Image.fromarray(state).save(img_path)
    
    def _save_best_model(self, model_type: str):
        """Save the best model"""
        model_path = os.path.join(self.save_dir, "models", f"{model_type}_model.pth")
        self.agent.save_model(model_path)
        print(f"Saved {model_type} model with reward: {self.best_reward:.2f}")
    
    def train(self, num_episodes: int = 100, render_frequency: int = 10, 
              save_frequency: int = 5, checkpoint_frequency: int = 20):
        """
        Train the agent for multiple episodes
        
        Args:
            num_episodes: Number of episodes to train
            render_frequency: How often to render episodes
            save_frequency: How often to save artworks
            checkpoint_frequency: How often to save checkpoints
        """
        print(f"Starting training for {num_episodes} episodes...")
        print(f"Canvas size: {self.canvas_width}x{self.canvas_height}")
        print(f"Max strokes per episode: {self.max_strokes}")
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Determine if we should render or save this episode
            should_render = (episode + 1) % render_frequency == 0
            should_save = (episode + 1) % save_frequency == 0
            
            # Train episode
            episode_stats = self.train_episode(
                render=should_render,
                save_artwork=should_save
            )
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.episode_rewards[-10:])
                avg_coverage = np.mean(self.episode_coverages[-10:])
                print(f"Episode {episode + 1}/{num_episodes} - "
                      f"Avg Reward: {avg_reward:.2f}, "
                      f"Avg Coverage: {avg_coverage:.3f}")
            
            # Save checkpoint
            if (episode + 1) % checkpoint_frequency == 0:
                self._save_checkpoint(episode + 1)
        
        # Final save
        self._save_checkpoint(num_episodes, is_final=True)
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time:.2f} seconds")
        print(f"Best reward: {self.best_reward:.2f}")
        print(f"Best coverage: {self.best_coverage:.3f}")
        
        # Generate training plots
        self._plot_training_progress()
    
    def _save_checkpoint(self, episode: int, is_final: bool = False):
        """Save training checkpoint"""
        checkpoint_path = os.path.join(self.save_dir, "models", f"checkpoint_episode_{episode}.pth")
        self.agent.save_model(checkpoint_path)
        
        # Save training statistics
        stats = {
            'episode': episode,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'episode_coverages': self.episode_coverages,
            'episode_color_diversities': self.episode_color_diversities,
            'best_reward': self.best_reward,
            'best_coverage': self.best_coverage,
            'training_losses': self.training_losses
        }
        
        stats_path = os.path.join(self.save_dir, "models", f"training_stats_episode_{episode}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        if is_final:
            print(f"Final checkpoint saved at episode {episode}")
    
    def _plot_training_progress(self):
        """Plot training progress"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Episode coverages
        axes[0, 1].plot(self.episode_coverages)
        axes[0, 1].set_title('Canvas Coverage')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Coverage')
        axes[0, 1].grid(True)
        
        # Color diversity
        axes[1, 0].plot(self.episode_color_diversities)
        axes[1, 0].set_title('Color Diversity')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Diversity')
        axes[1, 0].grid(True)
        
        # Training losses
        if self.training_losses:
            critic_losses = [loss['critic_loss'] for loss in self.training_losses]
            actor_losses = [loss['actor_loss'] for loss in self.training_losses]
            
            axes[1, 1].plot(critic_losses, label='Critic Loss')
            axes[1, 1].plot(actor_losses, label='Actor Loss')
            axes[1, 1].set_title('Training Losses')
            axes[1, 1].set_xlabel('Training Step')
            axes[1, 1].set_ylabel('Loss')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plot_path = os.path.join(self.save_dir, "training_progress.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load a training checkpoint"""
        self.agent.load_model(checkpoint_path)
        print(f"Loaded checkpoint from {checkpoint_path}")
    
    def generate_artwork(self, num_strokes: int = None, render: bool = True) -> np.ndarray:
        """
        Generate artwork using the trained agent
        
        Args:
            num_strokes: Number of strokes to generate (default: max_strokes)
            render: Whether to display the artwork
        
        Returns:
            Final canvas image
        """
        if num_strokes is None:
            num_strokes = self.max_strokes
        
        # Temporarily set max_strokes
        original_max_strokes = self.env.max_strokes
        self.env.max_strokes = num_strokes
        
        # Generate artwork
        state = self.env.reset()
        
        for step in range(num_strokes):
            # Select action without noise for consistent results
            action = self.agent.select_action(state, add_noise=False)
            
            # Take step
            next_state, reward, done, info = self.env.step(action)
            state = next_state
            
            if done:
                break
        
        # Restore original max_strokes
        self.env.max_strokes = original_max_strokes
        
        # Get final artwork
        final_artwork = self.env.canvas.get_image()
        
        if render:
            self.env.canvas.display("Generated Artwork")
        
        return final_artwork


def main():
    """Main training function"""
    # Create trainer
    trainer = ArtForgeTrainer(
        canvas_width=600,
        canvas_height=400,
        max_strokes=30,
        save_dir="gallery"
    )
    
    # Train the agent
    trainer.train(
        num_episodes=50,
        render_frequency=10,
        save_frequency=5,
        checkpoint_frequency=10
    )
    
    # Generate final artwork
    print("\nGenerating final artwork...")
    final_artwork = trainer.generate_artwork(num_strokes=40)
    
    # Save final artwork
    final_path = os.path.join(trainer.save_dir, "artworks", "final_artwork.png")
    from PIL import Image
    Image.fromarray(final_artwork).save(final_path)
    print(f"Final artwork saved to {final_path}")


if __name__ == "__main__":
    main()
