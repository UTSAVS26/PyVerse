import logging
import time
from typing import Dict, Any, List
import os

class TrainingLogger:
    """
    Logger for tracking training progress and metrics
    """
    
    def __init__(self, log_dir: str = "logs", agent_name: str = "Agent"):
        """
        Initialize logger
        
        Args:
            log_dir: Directory to save log files
            agent_name: Name of the agent for log file naming
        """
        self.log_dir = log_dir
        self.agent_name = agent_name
        os.makedirs(log_dir, exist_ok=True)
        
        # Setup logging
        self.logger = logging.getLogger(f"{agent_name}_training")
        self.logger.setLevel(logging.INFO)
        
        # Create handlers
        log_file = os.path.join(log_dir, f"{agent_name.lower()}_training.log")
        file_handler = logging.FileHandler(log_file)
        console_handler = logging.StreamHandler()
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(log_format)
        console_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
        
        # Training metrics
        self.episodes = []
        self.scores = []
        self.rewards = []
        self.avg_scores = []
        self.epsilons = []
        self.start_time = None
        
    def start_training(self, total_episodes: int, agent_info: Dict[str, Any]) -> None:
        """
        Log training start
        
        Args:
            total_episodes: Total number of episodes to train
            agent_info: Agent configuration information
        """
        self.start_time = time.time()
        self.logger.info(f"Starting training for {self.agent_name}")
        self.logger.info(f"Total episodes: {total_episodes}")
        self.logger.info(f"Agent configuration: {agent_info}")
        
    def log_episode(self, episode: int, score: float, reward: float, 
                   epsilon: float, avg_score: float = None) -> None:
        """
        Log episode results
        
        Args:
            episode: Episode number
            score: Score achieved in this episode
            reward: Total reward for this episode
            epsilon: Current exploration rate
            avg_score: Average score over last 100 episodes
        """
        # Store metrics
        self.episodes.append(episode)
        self.scores.append(score)
        self.rewards.append(reward)
        self.epsilons.append(epsilon)
        if avg_score is not None:
            self.avg_scores.append(avg_score)
        
        # Log every 100 episodes or at the end
        if episode % 100 == 0 or episode == len(self.episodes):
            self.logger.info(
                f"Episode {episode}: Score={score:.1f}, "
                f"Reward={reward:.1f}, Epsilon={epsilon:.3f}"
                + (f", Avg Score={avg_score:.1f}" if avg_score is not None else "")
            )
    
    def log_training_complete(self, best_score: float, final_avg_score: float) -> None:
        """
        Log training completion
        
        Args:
            best_score: Best score achieved during training
            final_avg_score: Final average score
        """
        training_time = time.time() - self.start_time
        self.logger.info(f"Training completed for {self.agent_name}")
        self.logger.info(f"Training time: {training_time:.2f} seconds")
        self.logger.info(f"Best score: {best_score}")
        self.logger.info(f"Final average score: {final_avg_score:.2f}")
        
    def get_training_data(self) -> Dict[str, List]:
        """
        Get training data for visualization
        
        Returns:
            Dictionary containing training metrics
        """
        data = {
            'episodes': self.episodes,
            'scores': self.scores,
            'rewards': self.rewards,
            'epsilons': self.epsilons
        }
        if self.avg_scores:
            data['avg_scores'] = self.avg_scores
        return data
    
    def log_agent_info(self, agent_info: Dict[str, Any]) -> None:
        """
        Log agent information
        
        Args:
            agent_info: Agent configuration and parameters
        """
        self.logger.info(f"Agent information: {agent_info}")
    
    def log_error(self, error_msg: str) -> None:
        """
        Log error message
        
        Args:
            error_msg: Error message to log
        """
        self.logger.error(f"Error: {error_msg}")
    
    def log_warning(self, warning_msg: str) -> None:
        """
        Log warning message
        
        Args:
            warning_msg: Warning message to log
        """
        self.logger.warning(f"Warning: {warning_msg}")
    
    def log_hyperparameters(self, hyperparams: Dict[str, Any]) -> None:
        """
        Log hyperparameters
        
        Args:
            hyperparams: Dictionary of hyperparameters
        """
        self.logger.info("Hyperparameters:")
        for key, value in hyperparams.items():
            self.logger.info(f"  {key}: {value}")
    
    def log_performance_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Log performance metrics
        
        Args:
            metrics: Dictionary of performance metrics
        """
        self.logger.info("Performance metrics:")
        for key, value in metrics.items():
            self.logger.info(f"  {key}: {value:.4f}")
    
    def close(self) -> None:
        """Close the logger"""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
