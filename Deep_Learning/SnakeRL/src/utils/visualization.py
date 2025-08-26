import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import List, Dict, Any
import os

class TrainingVisualizer:
    """
    Visualization utilities for training progress and results
    """
    
    def __init__(self, save_dir: str = "plots"):
        """
        Initialize visualizer
        
        Args:
            save_dir: Directory to save plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
    
    def plot_training_progress(self, episodes: List[int], scores: List[float], 
                              rewards: List[float], agent_name: str = "Agent") -> None:
        """
        Plot training progress over episodes
        
        Args:
            episodes: List of episode numbers
            scores: List of scores per episode
            rewards: List of rewards per episode
            agent_name: Name of the agent for plot title
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Plot scores
        ax1.plot(episodes, scores, alpha=0.6, linewidth=1)
        ax1.set_title(f'{agent_name} - Training Progress')
        ax1.set_ylabel('Score')
        ax1.set_xlabel('Episode')
        ax1.grid(True, alpha=0.3)
        
        # Plot rewards
        ax2.plot(episodes, rewards, alpha=0.6, linewidth=1, color='orange')
        ax2.set_ylabel('Reward')
        ax2.set_xlabel('Episode')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{agent_name.lower()}_training_progress.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_average_scores(self, episodes: List[int], avg_scores: List[float], 
                           agent_name: str = "Agent") -> None:
        """
        Plot average scores over episodes
        
        Args:
            episodes: List of episode numbers
            avg_scores: List of average scores
            agent_name: Name of the agent for plot title
        """
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, avg_scores, linewidth=2, color='blue')
        plt.title(f'{agent_name} - Average Score Over Episodes')
        plt.xlabel('Episode')
        plt.ylabel('Average Score (last 100 episodes)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{agent_name.lower()}_average_scores.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_epsilon_decay(self, episodes: List[int], epsilons: List[float], 
                          agent_name: str = "Agent") -> None:
        """
        Plot epsilon decay over episodes
        
        Args:
            episodes: List of episode numbers
            epsilons: List of epsilon values
            agent_name: Name of the agent for plot title
        """
        plt.figure(figsize=(10, 6))
        plt.plot(episodes, epsilons, linewidth=2, color='red')
        plt.title(f'{agent_name} - Epsilon Decay')
        plt.xlabel('Episode')
        plt.ylabel('Epsilon (Exploration Rate)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{agent_name.lower()}_epsilon_decay.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_comparison(self, results: Dict[str, Dict[str, List[float]]]) -> None:
        """
        Plot comparison between different agents
        
        Args:
            results: Dictionary with agent names as keys and dicts containing
                    'episodes', 'scores', 'rewards' as values
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        colors = ['blue', 'red', 'green', 'orange', 'purple']
        
        for i, (agent_name, data) in enumerate(results.items()):
            color = colors[i % len(colors)]
            episodes = data['episodes']
            scores = data['scores']
            rewards = data['rewards']
            
            ax1.plot(episodes, scores, label=agent_name, color=color, alpha=0.7)
            ax2.plot(episodes, rewards, label=agent_name, color=color, alpha=0.7)
        
        ax1.set_title('Agent Comparison - Scores')
        ax1.set_ylabel('Score')
        ax1.set_xlabel('Episode')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.set_title('Agent Comparison - Rewards')
        ax2.set_ylabel('Reward')
        ax2.set_xlabel('Episode')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, 'agent_comparison.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_histogram(self, scores: List[float], agent_name: str = "Agent") -> None:
        """
        Plot histogram of scores
        
        Args:
            scores: List of scores
            agent_name: Name of the agent for plot title
        """
        plt.figure(figsize=(10, 6))
        plt.hist(scores, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        plt.title(f'{agent_name} - Score Distribution')
        plt.xlabel('Score')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{agent_name.lower()}_score_distribution.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_summary_plot(self, training_data: Dict[str, Any], agent_name: str = "Agent") -> None:
        """
        Create a comprehensive summary plot
        
        Args:
            training_data: Dictionary containing training metrics
            agent_name: Name of the agent
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = training_data['episodes']
        scores = training_data['scores']
        rewards = training_data['rewards']
        avg_scores = training_data.get('avg_scores', [])
        epsilons = training_data.get('epsilons', [])
        
        # Score progression
        ax1.plot(episodes, scores, alpha=0.6, linewidth=1)
        ax1.set_title('Score Progression')
        ax1.set_ylabel('Score')
        ax1.grid(True, alpha=0.3)
        
        # Average scores
        if avg_scores:
            ax2.plot(episodes, avg_scores, linewidth=2, color='blue')
            ax2.set_title('Average Score (100 episodes)')
            ax2.set_ylabel('Average Score')
            ax2.grid(True, alpha=0.3)
        
        # Reward progression
        ax3.plot(episodes, rewards, alpha=0.6, linewidth=1, color='orange')
        ax3.set_title('Reward Progression')
        ax3.set_ylabel('Reward')
        ax3.set_xlabel('Episode')
        ax3.grid(True, alpha=0.3)
        
        # Epsilon decay
        if epsilons:
            ax4.plot(episodes, epsilons, linewidth=2, color='red')
            ax4.set_title('Epsilon Decay')
            ax4.set_ylabel('Epsilon')
            ax4.set_xlabel('Episode')
            ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'{agent_name} - Training Summary', fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, f'{agent_name.lower()}_summary.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
