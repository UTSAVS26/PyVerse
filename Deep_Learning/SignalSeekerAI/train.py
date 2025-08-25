import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import time
import json
import os
from collections import deque

from spectrum import RadioSpectrum, SignalDetector
from agent import RandomAgent, SequentialAgent, AdaptiveAgent, DQNAgent


class TrainingEnvironment:
    """Environment for training reinforcement learning agents."""
    
    def __init__(self, 
                 freq_range: Tuple[float, float] = (1e6, 100e6),
                 num_channels: int = 1000,
                 noise_floor: float = -90,
                 detection_threshold: float = -70,
                 episode_length: int = 1000):
        """
        Initialize the training environment.
        
        Args:
            freq_range: Frequency range in Hz
            num_channels: Number of frequency channels
            noise_floor: Noise floor in dBm
            detection_threshold: Signal detection threshold in dBm
            episode_length: Length of each training episode
        """
        self.spectrum = RadioSpectrum(
            freq_range=freq_range,
            num_channels=num_channels,
            noise_floor=noise_floor
        )
        self.detector = SignalDetector(threshold=detection_threshold)
        
        self.num_channels = num_channels
        self.frequencies = self.spectrum.frequencies
        
        # Training parameters
        self.episode_length = episode_length
        self.reward_history = []
        self.detection_history = []
        
        # Reward parameters
        self.reward_params = {
            'signal_detection': 10.0,
            'false_positive': -5.0,
            'missed_signal': -2.0,
            'scanning_cost': -0.1,
            'collision_penalty': -1.0
        }
    
    def reset(self):
        """Reset the environment for a new episode."""
        self.spectrum.reset()
        self.reward_history = []
        self.detection_history = []
        return self.spectrum.spectrum_power.copy()
    
    def step(self, agent, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.
        
        Args:
            agent: The agent taking the action
            action: Frequency index to scan
            
        Returns:
            Tuple of (next_state, reward, done, info)
        """
        # Get current spectrum
        spectrum = self.spectrum.step()
        
        # Get signal information
        signal_info = self.spectrum.get_signal_info()
        active_signals = signal_info['signals']
        
        # Check if action detects a signal
        power_at_action = spectrum[action]
        frequency_at_action = self.frequencies[action]
        
        # Detect signals at the action location
        detected_signals = self.detector.detect_signals(spectrum, self.frequencies)
        
        # Check if there's a real signal at the action location
        real_signal_at_action = False
        for signal_id, signal_data in active_signals.items():
            if abs(signal_data['freq_idx'] - action) <= 5:  # Within 5 channels
                real_signal_at_action = True
                break
        
        # Calculate reward
        reward = self._calculate_reward(
            power_at_action, 
            real_signal_at_action, 
            detected_signals,
            action
        )
        
        # Store history first
        self.reward_history.append(reward)
        
        # Check if episode is done (after adding the current reward)
        done = len(self.reward_history) >= self.episode_length
        self.detection_history.append({
            'action': action,
            'power': power_at_action,
            'real_signal': real_signal_at_action,
            'detected_signals': len(detected_signals),
            'reward': reward
        })
        
        info = {
            'power': power_at_action,
            'frequency': frequency_at_action,
            'real_signal': real_signal_at_action,
            'detected_signals': detected_signals,
            'active_signals': len(active_signals)
        }
        
        return spectrum, reward, done, info
    
    def _calculate_reward(self, power: float, real_signal: bool, 
                         detected_signals: List[Dict], action: int) -> float:
        """
        Calculate reward based on the agent's action.
        
        Args:
            power: Power level at the scanned frequency
            real_signal: Whether there's a real signal at the location
            detected_signals: List of detected signals
            action: The action taken
            
        Returns:
            Reward value
        """
        reward = 0.0
        
        # Base scanning cost
        reward += self.reward_params['scanning_cost']
        
        # Signal detection reward
        if real_signal and power > self.detector.threshold:
            reward += self.reward_params['signal_detection']
        elif real_signal and power <= self.detector.threshold:
            reward += self.reward_params['missed_signal']
        elif not real_signal and power > self.detector.threshold:
            reward += self.reward_params['false_positive']
        
        # Additional reward for high-power signals
        if power > -50:  # Very strong signal
            reward += 5.0
        elif power > -60:  # Strong signal
            reward += 2.0
        
        return reward
    
    def get_episode_statistics(self) -> Dict:
        """Get statistics for the current episode."""
        if not self.reward_history:
            return {}
        
        total_reward = sum(self.reward_history)
        avg_reward = np.mean(self.reward_history)
        
        # Detection statistics
        detections = [d for d in self.detection_history if d['real_signal']]
        false_positives = [d for d in self.detection_history 
                          if not d['real_signal'] and d['power'] > self.detector.threshold]
        
        return {
            'total_reward': total_reward,
            'avg_reward': avg_reward,
            'total_detections': len(detections),
            'false_positives': len(false_positives),
            'detection_rate': len(detections) / max(1, len(self.detection_history)),
            'false_positive_rate': len(false_positives) / max(1, len(self.detection_history))
        }


class Trainer:
    """Main training class for reinforcement learning agents."""
    
    def __init__(self, env: TrainingEnvironment, agent: DQNAgent):
        """
        Initialize the trainer.
        
        Args:
            env: Training environment
            agent: DQN agent to train
        """
        self.env = env
        self.agent = agent
        
        # Training parameters
        self.num_episodes = 1000
        self.target_update_freq = 100
        self.save_freq = 200
        
        # Statistics
        self.episode_rewards = []
        self.episode_detection_rates = []
        self.training_history = []
    
    def train(self, num_episodes: int = None) -> Dict:
        """
        Train the agent.
        
        Args:
            num_episodes: Number of episodes to train (uses default if None)
            
        Returns:
            Training statistics
        """
        if num_episodes is None:
            num_episodes = self.num_episodes
        
        print(f"Starting training for {num_episodes} episodes...")
        
        # Create models directory if it doesn't exist
        os.makedirs("models", exist_ok=True)
        
        for episode in range(num_episodes):
            state = self.env.reset()
            episode_reward = 0
            episode_detections = 0
            
            for step in range(self.env.episode_length):
                # Select action
                action = self.agent.select_action(state, self.env.frequencies)
                
                # Take action
                next_state, reward, done, info = self.env.step(self.agent, action)
                
                # Store experience (only for DQN agents)
                if hasattr(self.agent, '_get_state_representation'):
                    state_rep = self.agent._get_state_representation(state)
                    next_state_rep = self.agent._get_state_representation(next_state)
                    self.agent.remember(state_rep, action, reward, next_state_rep, done)
                    
                    # Train the agent
                    self.agent.replay()
                
                episode_reward += reward
                if info['real_signal']:
                    episode_detections += 1
                
                state = next_state
                
                if done:
                    break
            
            # Update target network periodically (only for DQN agents)
            if hasattr(self.agent, 'update_target_network') and episode % self.target_update_freq == 0:
                self.agent.update_target_network()
            
            # Save model periodically (only for DQN agents)
            if hasattr(self.agent, 'save_model') and episode % self.save_freq == 0:
                self.agent.save_model(f"models/dqn_agent_episode_{episode}.pth")
            
            # Get episode statistics
            episode_stats = self.env.get_episode_statistics()
            self.episode_rewards.append(episode_reward)
            self.episode_detection_rates.append(episode_stats['detection_rate'])
            
            # Store training history
            history_entry = {
                'episode': episode,
                'reward': episode_reward,
                'detection_rate': episode_stats['detection_rate']
            }
            
            # Add epsilon if the agent has it (DQN agents)
            if hasattr(self.agent, 'epsilon'):
                history_entry['epsilon'] = self.agent.epsilon
            
            self.training_history.append(history_entry)
            
            # Print progress
            if episode % 50 == 0:
                avg_reward = np.mean(self.episode_rewards[-50:])
                avg_detection_rate = np.mean(self.episode_detection_rates[-50:])
                progress_msg = f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Detection Rate = {avg_detection_rate:.3f}"
                
                # Add epsilon info if available
                if hasattr(self.agent, 'epsilon'):
                    progress_msg += f", Epsilon = {self.agent.epsilon:.3f}"
                
                print(progress_msg)
        
        # Save final model (only for DQN agents)
        if hasattr(self.agent, 'save_model'):
            self.agent.save_model("models/dqn_agent_final.pth")
        
        return self._get_training_statistics()
    
    def _get_training_statistics(self) -> Dict:
        """Get comprehensive training statistics."""
        stats = {
            'episode_rewards': self.episode_rewards,
            'episode_detection_rates': self.episode_detection_rates,
            'training_history': self.training_history,
            'avg_final_reward': np.mean(self.episode_rewards[-100:]),
            'avg_final_detection_rate': np.mean(self.episode_detection_rates[-100:])
        }
        
        # Add final epsilon if the agent has it (DQN agents)
        if hasattr(self.agent, 'epsilon'):
            stats['final_epsilon'] = self.agent.epsilon
        
        return stats


def compare_agents(env: TrainingEnvironment, num_episodes: int = 100) -> Dict:
    """
    Compare different agent types.
    
    Args:
        env: Training environment
        num_episodes: Number of episodes to test each agent
        
    Returns:
        Comparison statistics
    """
    agents = {
        'Random': RandomAgent(env.num_channels),
        'Sequential': SequentialAgent(env.num_channels),
        'Adaptive': AdaptiveAgent(env.num_channels),
        'DQN': DQNAgent(env.num_channels)
    }
    
    results = {}
    
    for agent_name, agent in agents.items():
        print(f"Testing {agent_name} agent...")
        
        episode_rewards = []
        episode_detection_rates = []
        
        for episode in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            episode_detections = 0
            
            for step in range(env.episode_length):
                action = agent.select_action(state, env.frequencies)
                next_state, reward, done, info = env.step(agent, action)
                
                episode_reward += reward
                if info['real_signal']:
                    episode_detections += 1
                
                state = next_state
                
                if done:
                    break
            
            episode_rewards.append(episode_reward)
            episode_stats = env.get_episode_statistics()
            episode_detection_rates.append(episode_stats['detection_rate'])
        
        results[agent_name] = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_detection_rate': np.mean(episode_detection_rates),
            'std_detection_rate': np.std(episode_detection_rates),
            'episode_rewards': episode_rewards,
            'episode_detection_rates': episode_detection_rates
        }
    
    return results


def main():
    """Main training function."""
    # Create output directories
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Initialize environment
    env = TrainingEnvironment(
        freq_range=(1e6, 100e6),
        num_channels=1000,
        noise_floor=-90,
        detection_threshold=-70
    )
    
    # Initialize DQN agent
    dqn_agent = DQNAgent(
        num_channels=env.num_channels,
        learning_rate=0.001,
        gamma=0.99,
        epsilon=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995
    )
    
    # Train DQN agent
    trainer = Trainer(env, dqn_agent)
    training_stats = trainer.train(num_episodes=500)
    
    # Save training results
    with open("results/training_results.json", "w") as f:
        json.dump(training_stats, f, indent=2)
    
    # Compare agents
    print("\nComparing different agent types...")
    comparison_results = compare_agents(env, num_episodes=50)
    
    # Save comparison results
    with open("results/agent_comparison.json", "w") as f:
        # Convert numpy arrays to lists for JSON serialization
        for agent_name, stats in comparison_results.items():
            stats['episode_rewards'] = [float(x) for x in stats['episode_rewards']]
            stats['episode_detection_rates'] = [float(x) for x in stats['episode_detection_rates']]
        json.dump(comparison_results, f, indent=2)
    
    # Print final results
    print("\nTraining completed!")
    print(f"Final average reward: {training_stats['avg_final_reward']:.2f}")
    print(f"Final detection rate: {training_stats['avg_final_detection_rate']:.3f}")
    
    print("\nAgent Comparison Results:")
    for agent_name, stats in comparison_results.items():
        print(f"{agent_name}: Avg Reward = {stats['avg_reward']:.2f} ± {stats['std_reward']:.2f}, "
              f"Detection Rate = {stats['avg_detection_rate']:.3f} ± {stats['std_detection_rate']:.3f}")


if __name__ == "__main__":
    main()
