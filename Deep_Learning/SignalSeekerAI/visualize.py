import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import json
import os

from spectrum import RadioSpectrum, SignalDetector
from agent import RandomAgent, SequentialAgent, AdaptiveAgent, DQNAgent


class SpectrumVisualizer:
    """Visualization tools for radio spectrum analysis."""
    
    def __init__(self, spectrum: RadioSpectrum, detector: SignalDetector):
        """
        Initialize the spectrum visualizer.
        
        Args:
            spectrum: Radio spectrum object
            detector: Signal detector object
        """
        self.spectrum = spectrum
        self.detector = detector
        self.frequencies = spectrum.frequencies
        
        # Set up plotting style
        plt.style.use('dark_background')
        self.colors = {
            'spectrum': '#00ff88',
            'threshold': '#ff4444',
            'signals': '#ffaa00',
            'noise': '#444444',
            'background': '#1a1a1a'
        }
    
    def plot_spectrum(self, spectrum: np.ndarray, title: str = "Radio Spectrum", 
                     show_signals: bool = True, save_path: str = None):
        """
        Plot the current spectrum.
        
        Args:
            spectrum: Power spectrum in dBm
            title: Plot title
            show_signals: Whether to highlight detected signals
            save_path: Path to save the plot (optional)
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot spectrum
        ax.plot(self.frequencies / 1e6, spectrum, 
                color=self.colors['spectrum'], linewidth=1.5, alpha=0.8)
        
        # Plot detection threshold
        ax.axhline(y=self.detector.threshold, color=self.colors['threshold'], 
                   linestyle='--', alpha=0.7, label=f'Threshold ({self.detector.threshold} dBm)')
        
        # Highlight detected signals
        if show_signals:
            detected_signals = self.detector.detect_signals(spectrum, self.frequencies)
            for signal in detected_signals:
                ax.axvline(x=signal['frequency'] / 1e6, color=self.colors['signals'], 
                          alpha=0.6, linestyle=':')
                ax.text(signal['frequency'] / 1e6, signal['power'] + 5, 
                       f"{signal['power']:.1f} dBm", 
                       color=self.colors['signals'], fontsize=8, ha='center')
        
        # Formatting
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power (dBm)')
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Set background color
        ax.set_facecolor(self.colors['background'])
        fig.patch.set_facecolor(self.colors['background'])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.colors['background'])
        
        plt.show()
    
    def plot_spectrogram(self, spectrum_history: List[np.ndarray], 
                        time_steps: List[float], title: str = "Spectrogram",
                        save_path: str = None):
        """
        Create a spectrogram from spectrum history.
        
        Args:
            spectrum_history: List of spectrum snapshots
            time_steps: Corresponding time steps
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        if not spectrum_history:
            return
        
        # Convert to numpy array
        spectrogram_data = np.array(spectrum_history)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create custom colormap for radio spectrum
        colors = ['#000000', '#0000ff', '#00ff00', '#ffff00', '#ff0000']
        n_bins = 100
        cmap = LinearSegmentedColormap.from_list('radio_spectrum', colors, N=n_bins)
        
        # Plot spectrogram
        im = ax.imshow(spectrogram_data.T, aspect='auto', cmap=cmap,
                      extent=[time_steps[0], time_steps[-1], 
                              self.frequencies[0] / 1e6, self.frequencies[-1] / 1e6],
                      origin='lower')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Power (dBm)')
        
        # Formatting
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Frequency (MHz)')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_agent_scanning(self, agent, spectrum: np.ndarray, 
                           scan_history: List[Dict], title: str = "Agent Scanning",
                           save_path: str = None):
        """
        Visualize agent scanning behavior.
        
        Args:
            agent: The scanning agent
            spectrum: Current spectrum
            scan_history: History of agent scans
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot 1: Spectrum with agent position
        ax1.plot(self.frequencies / 1e6, spectrum, 
                color=self.colors['spectrum'], linewidth=1.5, alpha=0.8)
        ax1.axhline(y=self.detector.threshold, color=self.colors['threshold'], 
                   linestyle='--', alpha=0.7, label='Detection Threshold')
        
        # Mark current agent position
        if scan_history:
            current_pos = scan_history[-1]['position']
            current_freq = self.frequencies[current_pos] / 1e6
            current_power = spectrum[current_pos]
            ax1.plot(current_freq, current_power, 'ro', markersize=10, 
                    label=f'Agent Position ({current_power:.1f} dBm)')
        
        ax1.set_xlabel('Frequency (MHz)')
        ax1.set_ylabel('Power (dBm)')
        ax1.set_title(f"{title} - Current Spectrum")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_facecolor(self.colors['background'])
        
        # Plot 2: Agent scanning history
        if scan_history:
            positions = [scan['position'] for scan in scan_history]
            powers = [scan['power'] for scan in scan_history]
            frequencies = [self.frequencies[pos] / 1e6 for pos in positions]
            
            ax2.scatter(frequencies, powers, c=range(len(scan_history)), 
                       cmap='viridis', alpha=0.7, s=20)
            ax2.axhline(y=self.detector.threshold, color=self.colors['threshold'], 
                       linestyle='--', alpha=0.7, label='Detection Threshold')
            
            ax2.set_xlabel('Frequency (MHz)')
            ax2.set_ylabel('Power (dBm)')
            ax2.set_title(f"{title} - Scanning History")
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_facecolor(self.colors['background'])
        
        fig.patch.set_facecolor(self.colors['background'])
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                       facecolor=self.colors['background'])
        
        plt.show()
    
    def plot_heatmap(self, spectrum_history: List[np.ndarray], 
                    title: str = "Spectrum Heatmap", save_path: str = None):
        """
        Create a heatmap of spectrum over time.
        
        Args:
            spectrum_history: List of spectrum snapshots
            title: Plot title
            save_path: Path to save the plot (optional)
        """
        if not spectrum_history:
            return
        
        spectrogram_data = np.array(spectrum_history)
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Create heatmap
        sns.heatmap(spectrogram_data.T, cmap='viridis', ax=ax,
                   xticklabels=len(self.frequencies) // 10,
                   yticklabels=len(spectrogram_data) // 10)
        
        # Formatting
        ax.set_xlabel('Frequency Channel')
        ax.set_ylabel('Time Step')
        ax.set_title(title)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def animate_spectrum(self, spectrum_history: List[np.ndarray], 
                        interval: int = 100, save_path: str = None):
        """
        Create an animated spectrum visualization.
        
        Args:
            spectrum_history: List of spectrum snapshots
            interval: Animation interval in milliseconds
            save_path: Path to save the animation (optional)
        """
        if not spectrum_history:
            return
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Initial plot
        line, = ax.plot(self.frequencies / 1e6, spectrum_history[0], 
                       color=self.colors['spectrum'], linewidth=1.5)
        ax.axhline(y=self.detector.threshold, color=self.colors['threshold'], 
                   linestyle='--', alpha=0.7, label='Detection Threshold')
        
        ax.set_xlabel('Frequency (MHz)')
        ax.set_ylabel('Power (dBm)')
        ax.set_title('Animated Radio Spectrum')
        ax.grid(True, alpha=0.3)
        ax.legend()
        ax.set_facecolor(self.colors['background'])
        fig.patch.set_facecolor(self.colors['background'])
        
        def animate(frame):
            line.set_ydata(spectrum_history[frame])
            return line,
        
        anim = animation.FuncAnimation(fig, animate, frames=len(spectrum_history),
                                     interval=interval, blit=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=10)
        
        plt.show()
        return anim


class TrainingVisualizer:
    """Visualization tools for training results."""
    
    def __init__(self):
        """Initialize the training visualizer."""
        plt.style.use('default')
    
    def plot_training_progress(self, training_stats: Dict, save_path: str = None):
        """
        Plot training progress over episodes.
        
        Args:
            training_stats: Training statistics dictionary
            save_path: Path to save the plot (optional)
        """
        if not training_stats or 'episode_rewards' not in training_stats:
            print("No training statistics available.")
            return
        
        if len(training_stats['episode_rewards']) == 0:
            print("No training data available.")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        episodes = range(len(training_stats['episode_rewards']))
        
        # Plot 1: Episode rewards
        ax1.plot(episodes, training_stats['episode_rewards'], 'b-', alpha=0.7)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Episode Reward')
        ax1.set_title('Training Progress - Episode Rewards')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Moving average rewards
        window = min(50, len(episodes) // 2)  # Use smaller window for short datasets
        if window > 1:
            moving_avg = np.convolve(training_stats['episode_rewards'], 
                                    np.ones(window)/window, mode='valid')
            ax2.plot(episodes[window-1:], moving_avg, 'r-', linewidth=2)
            ax2.set_title(f'Training Progress - Moving Average (Window={window})')
        else:
            ax2.text(0.5, 0.5, 'Not enough data for moving average', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('Training Progress - Moving Average')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Moving Average Reward')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Detection rates
        ax3.plot(episodes, training_stats['episode_detection_rates'], 'g-', alpha=0.7)
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Detection Rate')
        ax3.set_title('Training Progress - Detection Rates')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Epsilon decay
        epsilons = [h['epsilon'] for h in training_stats['training_history']]
        ax4.plot(episodes, epsilons, 'm-', alpha=0.7)
        ax4.set_xlabel('Episode')
        ax4.set_ylabel('Epsilon')
        ax4.set_title('Training Progress - Epsilon Decay')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_agent_comparison(self, comparison_results: Dict, save_path: str = None):
        """
        Plot comparison of different agent types.
        
        Args:
            comparison_results: Agent comparison results
            save_path: Path to save the plot (optional)
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        agent_names = list(comparison_results.keys())
        avg_rewards = [comparison_results[name]['avg_reward'] for name in agent_names]
        std_rewards = [comparison_results[name]['std_reward'] for name in agent_names]
        avg_detection_rates = [comparison_results[name]['avg_detection_rate'] for name in agent_names]
        std_detection_rates = [comparison_results[name]['std_detection_rate'] for name in agent_names]
        
        # Plot 1: Average rewards with error bars
        ax1.bar(agent_names, avg_rewards, yerr=std_rewards, capsize=5, alpha=0.7)
        ax1.set_ylabel('Average Reward')
        ax1.set_title('Agent Comparison - Average Rewards')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Average detection rates with error bars
        ax2.bar(agent_names, avg_detection_rates, yerr=std_detection_rates, capsize=5, alpha=0.7)
        ax2.set_ylabel('Average Detection Rate')
        ax2.set_title('Agent Comparison - Detection Rates')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Reward distributions
        for name in agent_names:
            ax3.hist(comparison_results[name]['episode_rewards'], alpha=0.6, 
                    label=name, bins=20)
        ax3.set_xlabel('Episode Reward')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Agent Comparison - Reward Distributions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Detection rate distributions
        for name in agent_names:
            ax4.hist(comparison_results[name]['episode_detection_rates'], alpha=0.6, 
                    label=name, bins=20)
        ax4.set_xlabel('Detection Rate')
        ax4.set_ylabel('Frequency')
        ax4.set_title('Agent Comparison - Detection Rate Distributions')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_learning_curves(self, training_stats: Dict, save_path: str = None):
        """
        Plot detailed learning curves.
        
        Args:
            training_stats: Training statistics dictionary
            save_path: Path to save the plot (optional)
        """
        if not training_stats or 'episode_rewards' not in training_stats:
            print("No training statistics available.")
            return
        
        if len(training_stats['episode_rewards']) == 0:
            print("No training data available.")
            return
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        episodes = range(len(training_stats['episode_rewards']))
        
        # Plot rewards with trend line
        ax.plot(episodes, training_stats['episode_rewards'], 'b-', alpha=0.3, label='Episode Rewards')
        
        # Add trend line
        z = np.polyfit(episodes, training_stats['episode_rewards'], 1)
        p = np.poly1d(z)
        ax.plot(episodes, p(episodes), 'r-', linewidth=2, label='Trend Line')
        
        # Add moving average (adaptive window size)
        window = min(50, len(episodes) // 2)  # Use smaller window for short datasets
        if window > 1:
            moving_avg = np.convolve(training_stats['episode_rewards'], 
                                    np.ones(window)/window, mode='valid')
            ax.plot(episodes[window-1:], moving_avg, 'g-', linewidth=2, label=f'Moving Average (Window={window})')
        
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Learning Curves - Training Progress')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def create_dashboard(spectrum: RadioSpectrum, agent, num_steps: int = 100):
    """
    Create a real-time dashboard for spectrum monitoring.
    
    Args:
        spectrum: Radio spectrum object
        agent: Scanning agent
        num_steps: Number of steps to simulate
    """
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    visualizer = SpectrumVisualizer(spectrum, SignalDetector())
    
    # Collect data
    spectrum_history = []
    scan_history = []
    time_steps = []
    
    print("Creating real-time dashboard...")
    
    for step in range(num_steps):
        # Get current spectrum
        current_spectrum = spectrum.step()
        spectrum_history.append(current_spectrum.copy())
        
        # Agent scan
        scan_result = agent.scan_spectrum(current_spectrum, spectrum.frequencies)
        scan_history.append(scan_result)
        
        time_steps.append(spectrum.time)
        
        # Print progress
        if step % 10 == 0:
            print(f"Step {step}/{num_steps}")
    
    # Create visualizations
    print("Generating visualizations...")
    
    # Current spectrum
    visualizer.plot_spectrum(spectrum_history[-1], "Final Spectrum State", 
                           save_path="results/final_spectrum.png")
    
    # Agent scanning
    visualizer.plot_agent_scanning(agent, spectrum_history[-1], scan_history,
                                 "Agent Scanning Behavior", 
                                 save_path="results/agent_scanning.png")
    
    # Spectrogram
    visualizer.plot_spectrogram(spectrum_history, time_steps, "Spectrum Over Time",
                               save_path="results/spectrogram.png")
    
    # Heatmap
    visualizer.plot_heatmap(spectrum_history, "Spectrum Heatmap",
                           save_path="results/heatmap.png")
    
    print("Dashboard created successfully!")


def main():
    """Main visualization function."""
    # Create output directory
    os.makedirs("results", exist_ok=True)
    
    # Initialize spectrum and agent
    spectrum = RadioSpectrum(
        freq_range=(1e6, 100e6),
        num_channels=1000,
        noise_floor=-90
    )
    
    agent = DQNAgent(num_channels=1000)
    
    # Create dashboard
    create_dashboard(spectrum, agent, num_steps=200)
    
    # Load and plot training results if available
    if os.path.exists("results/training_results.json"):
        with open("results/training_results.json", "r") as f:
            training_stats = json.load(f)
        
        training_viz = TrainingVisualizer()
        training_viz.plot_training_progress(training_stats, 
                                          save_path="results/training_progress.png")
        training_viz.plot_learning_curves(training_stats, 
                                        save_path="results/learning_curves.png")
    
    # Load and plot agent comparison if available
    if os.path.exists("results/agent_comparison.json"):
        with open("results/agent_comparison.json", "r") as f:
            comparison_results = json.load(f)
        
        training_viz = TrainingVisualizer()
        training_viz.plot_agent_comparison(comparison_results, 
                                         save_path="results/agent_comparison.png")


if __name__ == "__main__":
    main()
