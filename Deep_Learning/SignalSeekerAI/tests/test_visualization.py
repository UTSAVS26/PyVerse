import pytest
import numpy as np
import tempfile
import os
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for testing
import matplotlib.pyplot as plt
from visualize import SpectrumVisualizer, TrainingVisualizer, create_dashboard
from spectrum import RadioSpectrum, SignalDetector
from agent import DQNAgent


class TestSpectrumVisualizer:
    """Test cases for SpectrumVisualizer class."""
    
    def test_initialization(self):
        """Test spectrum visualizer initialization."""
        spectrum = RadioSpectrum(num_channels=100)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        assert visualizer.spectrum == spectrum
        assert visualizer.detector == detector
        assert visualizer.frequencies.shape == (100,)
        assert 'spectrum' in visualizer.colors
        assert 'threshold' in visualizer.colors
        assert 'signals' in visualizer.colors
    
    def test_plot_spectrum_basic(self):
        """Test basic spectrum plotting."""
        spectrum = RadioSpectrum(num_channels=100)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        current_spectrum = spectrum.step()
        
        # Test plotting without saving
        visualizer.plot_spectrum(current_spectrum, "Test Spectrum")
        
        # Check that plot was created
        assert plt.get_fignums()  # Should have at least one figure
        plt.close('all')  # Clean up
    
    def test_plot_spectrum_with_save(self):
        """Test spectrum plotting with save functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            spectrum = RadioSpectrum(num_channels=100)
            detector = SignalDetector(threshold=-70)
            visualizer = SpectrumVisualizer(spectrum, detector)
            
            current_spectrum = spectrum.step()
            save_path = os.path.join(temp_dir, "test_spectrum.png")
            
            # Test plotting with save
            visualizer.plot_spectrum(current_spectrum, "Test Spectrum", save_path=save_path)
            
            # Check that file was created
            assert os.path.exists(save_path)
            plt.close('all')  # Clean up
    
    def test_plot_spectrum_with_signals(self):
        """Test spectrum plotting with signal detection."""
        spectrum = RadioSpectrum(num_channels=100)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        # Create spectrum with strong signal
        current_spectrum = spectrum.step()
        current_spectrum[50] = -50  # Add strong signal
        
        # Test plotting with signal detection
        visualizer.plot_spectrum(current_spectrum, "Test Spectrum with Signals", show_signals=True)
        
        # Check that plot was created
        assert plt.get_fignums()
        plt.close('all')  # Clean up
    
    def test_plot_spectrogram(self):
        """Test spectrogram plotting."""
        spectrum = RadioSpectrum(num_channels=100)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        # Create spectrum history
        spectrum_history = []
        time_steps = []
        
        for i in range(10):
            current_spectrum = spectrum.step()
            spectrum_history.append(current_spectrum.copy())
            time_steps.append(spectrum.time)
        
        # Test spectrogram plotting
        visualizer.plot_spectrogram(spectrum_history, time_steps, "Test Spectrogram")
        
        # Check that plot was created
        assert plt.get_fignums()
        plt.close('all')  # Clean up
    
    def test_plot_spectrogram_with_save(self):
        """Test spectrogram plotting with save functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            spectrum = RadioSpectrum(num_channels=100)
            detector = SignalDetector(threshold=-70)
            visualizer = SpectrumVisualizer(spectrum, detector)
            
            # Create spectrum history
            spectrum_history = []
            time_steps = []
            
            for i in range(10):
                current_spectrum = spectrum.step()
                spectrum_history.append(current_spectrum.copy())
                time_steps.append(spectrum.time)
            
            save_path = os.path.join(temp_dir, "test_spectrogram.png")
            visualizer.plot_spectrogram(spectrum_history, time_steps, "Test Spectrogram", save_path=save_path)
            
            # Check that file was created
            assert os.path.exists(save_path)
            plt.close('all')  # Clean up
    
    def test_plot_agent_scanning(self):
        """Test agent scanning visualization."""
        spectrum = RadioSpectrum(num_channels=100)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        agent = DQNAgent(num_channels=100)
        
        current_spectrum = spectrum.step()
        
        # Create scan history
        scan_history = []
        for i in range(5):
            scan_result = agent.scan_spectrum(current_spectrum, spectrum.frequencies)
            scan_history.append(scan_result)
        
        # Test agent scanning visualization
        visualizer.plot_agent_scanning(agent, current_spectrum, scan_history, "Test Agent Scanning")
        
        # Check that plot was created
        assert plt.get_fignums()
        plt.close('all')  # Clean up
    
    def test_plot_heatmap(self):
        """Test heatmap plotting."""
        spectrum = RadioSpectrum(num_channels=100)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        # Create spectrum history
        spectrum_history = []
        for i in range(10):
            current_spectrum = spectrum.step()
            spectrum_history.append(current_spectrum.copy())
        
        # Test heatmap plotting
        visualizer.plot_heatmap(spectrum_history, "Test Heatmap")
        
        # Check that plot was created
        assert plt.get_fignums()
        plt.close('all')  # Clean up
    
    def test_plot_heatmap_with_save(self):
        """Test heatmap plotting with save functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            spectrum = RadioSpectrum(num_channels=100)
            detector = SignalDetector(threshold=-70)
            visualizer = SpectrumVisualizer(spectrum, detector)
            
            # Create spectrum history
            spectrum_history = []
            for i in range(10):
                current_spectrum = spectrum.step()
                spectrum_history.append(current_spectrum.copy())
            
            save_path = os.path.join(temp_dir, "test_heatmap.png")
            visualizer.plot_heatmap(spectrum_history, "Test Heatmap", save_path=save_path)
            
            # Check that file was created
            assert os.path.exists(save_path)
            plt.close('all')  # Clean up
    
    def test_animate_spectrum(self):
        """Test spectrum animation."""
        spectrum = RadioSpectrum(num_channels=100)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        # Create spectrum history
        spectrum_history = []
        for i in range(10):
            current_spectrum = spectrum.step()
            spectrum_history.append(current_spectrum.copy())
        
        # Test animation
        anim = visualizer.animate_spectrum(spectrum_history, interval=100)
        
        # Check that animation was created
        assert anim is not None
        plt.close('all')  # Clean up
    
    def test_animate_spectrum_with_save(self):
        """Test spectrum animation with save functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            spectrum = RadioSpectrum(num_channels=100)
            detector = SignalDetector(threshold=-70)
            visualizer = SpectrumVisualizer(spectrum, detector)
            
            # Create spectrum history
            spectrum_history = []
            for i in range(5):  # Use fewer frames for faster test
                current_spectrum = spectrum.step()
                spectrum_history.append(current_spectrum.copy())
            
            save_path = os.path.join(temp_dir, "test_animation.gif")
            anim = visualizer.animate_spectrum(spectrum_history, interval=100, save_path=save_path)
            
            # Check that animation was created
            assert anim is not None
            plt.close('all')  # Clean up
    
    def test_empty_spectrum_history(self):
        """Test visualization with empty spectrum history."""
        spectrum = RadioSpectrum(num_channels=100)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        # Test with empty history
        visualizer.plot_spectrogram([], [], "Empty Spectrogram")
        visualizer.plot_heatmap([], "Empty Heatmap")
        visualizer.animate_spectrum([], interval=100)
        
        # Should not crash
        plt.close('all')  # Clean up


class TestTrainingVisualizer:
    """Test cases for TrainingVisualizer class."""
    
    def test_initialization(self):
        """Test training visualizer initialization."""
        visualizer = TrainingVisualizer()
        assert visualizer is not None
    
    def test_plot_training_progress(self):
        """Test training progress plotting."""
        visualizer = TrainingVisualizer()
        
        # Create mock training statistics
        training_stats = {
            'episode_rewards': [10, 15, 20, 18, 25],
            'episode_detection_rates': [0.5, 0.6, 0.7, 0.65, 0.8],
            'training_history': [
                {'episode': 0, 'reward': 10, 'detection_rate': 0.5, 'epsilon': 1.0},
                {'episode': 1, 'reward': 15, 'detection_rate': 0.6, 'epsilon': 0.9},
                {'episode': 2, 'reward': 20, 'detection_rate': 0.7, 'epsilon': 0.8},
                {'episode': 3, 'reward': 18, 'detection_rate': 0.65, 'epsilon': 0.7},
                {'episode': 4, 'reward': 25, 'detection_rate': 0.8, 'epsilon': 0.6}
            ]
        }
        
        # Test plotting
        visualizer.plot_training_progress(training_stats)
        
        # Check that plot was created
        assert plt.get_fignums()
        plt.close('all')  # Clean up
    
    def test_plot_training_progress_with_save(self):
        """Test training progress plotting with save functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrainingVisualizer()
            
            # Create mock training statistics
            training_stats = {
                'episode_rewards': [10, 15, 20, 18, 25],
                'episode_detection_rates': [0.5, 0.6, 0.7, 0.65, 0.8],
                'training_history': [
                    {'episode': 0, 'reward': 10, 'detection_rate': 0.5, 'epsilon': 1.0},
                    {'episode': 1, 'reward': 15, 'detection_rate': 0.6, 'epsilon': 0.9},
                    {'episode': 2, 'reward': 20, 'detection_rate': 0.7, 'epsilon': 0.8},
                    {'episode': 3, 'reward': 18, 'detection_rate': 0.65, 'epsilon': 0.7},
                    {'episode': 4, 'reward': 25, 'detection_rate': 0.8, 'epsilon': 0.6}
                ]
            }
            
            save_path = os.path.join(temp_dir, "test_training_progress.png")
            visualizer.plot_training_progress(training_stats, save_path=save_path)
            
            # Check that file was created
            assert os.path.exists(save_path)
            plt.close('all')  # Clean up
    
    def test_plot_agent_comparison(self):
        """Test agent comparison plotting."""
        visualizer = TrainingVisualizer()
        
        # Create mock comparison results
        comparison_results = {
            'Random': {
                'avg_reward': 10.0,
                'std_reward': 2.0,
                'avg_detection_rate': 0.3,
                'std_detection_rate': 0.1,
                'episode_rewards': [8, 10, 12, 9, 11],
                'episode_detection_rates': [0.2, 0.3, 0.4, 0.25, 0.35]
            },
            'Sequential': {
                'avg_reward': 15.0,
                'std_reward': 3.0,
                'avg_detection_rate': 0.5,
                'std_detection_rate': 0.15,
                'episode_rewards': [12, 15, 18, 14, 16],
                'episode_detection_rates': [0.4, 0.5, 0.6, 0.45, 0.55]
            },
            'Adaptive': {
                'avg_reward': 20.0,
                'std_reward': 4.0,
                'avg_detection_rate': 0.7,
                'std_detection_rate': 0.2,
                'episode_rewards': [16, 20, 24, 18, 22],
                'episode_detection_rates': [0.6, 0.7, 0.8, 0.65, 0.75]
            },
            'DQN': {
                'avg_reward': 25.0,
                'std_reward': 5.0,
                'avg_detection_rate': 0.8,
                'std_detection_rate': 0.25,
                'episode_rewards': [20, 25, 30, 22, 28],
                'episode_detection_rates': [0.7, 0.8, 0.9, 0.75, 0.85]
            }
        }
        
        # Test plotting
        visualizer.plot_agent_comparison(comparison_results)
        
        # Check that plot was created
        assert plt.get_fignums()
        plt.close('all')  # Clean up
    
    def test_plot_agent_comparison_with_save(self):
        """Test agent comparison plotting with save functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrainingVisualizer()
            
            # Create mock comparison results
            comparison_results = {
                'Random': {
                    'avg_reward': 10.0,
                    'std_reward': 2.0,
                    'avg_detection_rate': 0.3,
                    'std_detection_rate': 0.1,
                    'episode_rewards': [8, 10, 12, 9, 11],
                    'episode_detection_rates': [0.2, 0.3, 0.4, 0.25, 0.35]
                },
                'Sequential': {
                    'avg_reward': 15.0,
                    'std_reward': 3.0,
                    'avg_detection_rate': 0.5,
                    'std_detection_rate': 0.15,
                    'episode_rewards': [12, 15, 18, 14, 16],
                    'episode_detection_rates': [0.4, 0.5, 0.6, 0.45, 0.55]
                }
            }
            
            save_path = os.path.join(temp_dir, "test_agent_comparison.png")
            visualizer.plot_agent_comparison(comparison_results, save_path=save_path)
            
            # Check that file was created
            assert os.path.exists(save_path)
            plt.close('all')  # Clean up
    
    def test_plot_learning_curves(self):
        """Test learning curves plotting."""
        visualizer = TrainingVisualizer()
        
        # Create mock training statistics
        training_stats = {
            'episode_rewards': [10, 15, 20, 18, 25, 22, 28, 30, 27, 35]
        }
        
        # Test plotting
        visualizer.plot_learning_curves(training_stats)
        
        # Check that plot was created
        assert plt.get_fignums()
        plt.close('all')  # Clean up
    
    def test_plot_learning_curves_with_save(self):
        """Test learning curves plotting with save functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            visualizer = TrainingVisualizer()
            
            # Create mock training statistics
            training_stats = {
                'episode_rewards': [10, 15, 20, 18, 25, 22, 28, 30, 27, 35]
            }
            
            save_path = os.path.join(temp_dir, "test_learning_curves.png")
            visualizer.plot_learning_curves(training_stats, save_path=save_path)
            
            # Check that file was created
            assert os.path.exists(save_path)
            plt.close('all')  # Clean up
    
    def test_empty_training_stats(self):
        """Test visualization with empty training statistics."""
        visualizer = TrainingVisualizer()
        
        # Test with empty statistics
        empty_stats = {
            'episode_rewards': [],
            'episode_detection_rates': [],
            'training_history': []
        }
        
        # Should not crash
        visualizer.plot_training_progress(empty_stats)
        visualizer.plot_learning_curves(empty_stats)
        
        plt.close('all')  # Clean up


class TestDashboard:
    """Test cases for dashboard functionality."""
    
    def test_create_dashboard(self):
        """Test dashboard creation."""
        spectrum = RadioSpectrum(num_channels=50)
        agent = DQNAgent(num_channels=50)
        
        # Test dashboard creation with small number of steps
        create_dashboard(spectrum, agent, num_steps=5)
        
        # Check that results directory was created
        assert os.path.exists("results")
        
        # Check that output files were created
        expected_files = [
            "results/final_spectrum.png",
            "results/agent_scanning.png",
            "results/spectrogram.png",
            "results/heatmap.png"
        ]
        
        for file_path in expected_files:
            if os.path.exists(file_path):
                # Clean up
                os.remove(file_path)
        
        # Clean up results directory if empty
        if os.path.exists("results") and not os.listdir("results"):
            os.rmdir("results")
    
    def test_create_dashboard_with_custom_directory(self):
        """Test dashboard creation with custom directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Change to temporary directory
            original_dir = os.getcwd()
            os.chdir(temp_dir)
            
            try:
                spectrum = RadioSpectrum(num_channels=50)
                agent = DQNAgent(num_channels=50)
                
                # Test dashboard creation
                create_dashboard(spectrum, agent, num_steps=3)
                
                # Check that results directory was created
                assert os.path.exists("results")
                
                # Check that output files were created
                expected_files = [
                    "results/final_spectrum.png",
                    "results/agent_scanning.png",
                    "results/spectrogram.png",
                    "results/heatmap.png"
                ]
                
                for file_path in expected_files:
                    if os.path.exists(file_path):
                        # Clean up
                        os.remove(file_path)
                
                # Clean up results directory if empty
                if os.path.exists("results") and not os.listdir("results"):
                    os.rmdir("results")
            
            finally:
                # Restore original directory
                os.chdir(original_dir)


class TestVisualizationIntegration:
    """Integration tests for visualization system."""
    
    def test_spectrum_visualizer_integration(self):
        """Test spectrum visualizer with real spectrum data."""
        spectrum = RadioSpectrum(num_channels=100)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        # Get real spectrum data
        current_spectrum = spectrum.step()
        
        # Test all visualization methods
        visualizer.plot_spectrum(current_spectrum, "Real Spectrum")
        
        # Create spectrum history
        spectrum_history = []
        for i in range(5):
            current_spectrum = spectrum.step()
            spectrum_history.append(current_spectrum.copy())
        
        visualizer.plot_spectrogram(spectrum_history, list(range(5)), "Real Spectrogram")
        visualizer.plot_heatmap(spectrum_history, "Real Heatmap")
        
        # Test with agent
        agent = DQNAgent(num_channels=100)
        scan_history = []
        for i in range(3):
            scan_result = agent.scan_spectrum(current_spectrum, spectrum.frequencies)
            scan_history.append(scan_result)
        
        visualizer.plot_agent_scanning(agent, current_spectrum, scan_history, "Real Agent Scanning")
        
        plt.close('all')  # Clean up
    
    def test_training_visualizer_integration(self):
        """Test training visualizer with realistic training data."""
        visualizer = TrainingVisualizer()
        
        # Create realistic training statistics
        episodes = 20
        training_stats = {
            'episode_rewards': list(range(10, 10 + episodes)),
            'episode_detection_rates': [0.3 + 0.02 * i for i in range(episodes)],
            'training_history': [
                {
                    'episode': i,
                    'reward': 10 + i,
                    'detection_rate': 0.3 + 0.02 * i,
                    'epsilon': max(0.01, 1.0 - 0.05 * i)
                }
                for i in range(episodes)
            ]
        }
        
        # Test all visualization methods
        visualizer.plot_training_progress(training_stats)
        visualizer.plot_learning_curves(training_stats)
        
        # Create realistic comparison results
        comparison_results = {
            'Random': {
                'avg_reward': 12.0,
                'std_reward': 3.0,
                'avg_detection_rate': 0.35,
                'std_detection_rate': 0.1,
                'episode_rewards': list(range(10, 15)),
                'episode_detection_rates': [0.3, 0.35, 0.4, 0.35, 0.4]
            },
            'DQN': {
                'avg_reward': 25.0,
                'std_reward': 5.0,
                'avg_detection_rate': 0.75,
                'std_detection_rate': 0.15,
                'episode_rewards': list(range(20, 30, 2)),
                'episode_detection_rates': [0.6, 0.7, 0.8, 0.75, 0.85]
            }
        }
        
        visualizer.plot_agent_comparison(comparison_results)
        
        plt.close('all')  # Clean up


class TestVisualizationEdgeCases:
    """Test edge cases in visualization system."""
    
    def test_single_point_spectrum(self):
        """Test visualization with single point spectrum."""
        spectrum = RadioSpectrum(num_channels=1)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        current_spectrum = spectrum.step()
        
        # Should not crash
        visualizer.plot_spectrum(current_spectrum, "Single Point Spectrum")
        plt.close('all')  # Clean up
    
    def test_very_large_spectrum(self):
        """Test visualization with very large spectrum."""
        spectrum = RadioSpectrum(num_channels=10000)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        current_spectrum = spectrum.step()
        
        # Should not crash (though it might be slow)
        visualizer.plot_spectrum(current_spectrum, "Large Spectrum")
        plt.close('all')  # Clean up
    
    def test_extreme_power_values(self):
        """Test visualization with extreme power values."""
        spectrum = RadioSpectrum(num_channels=100)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        # Create spectrum with extreme values
        current_spectrum = np.full(100, -120)  # Very low power
        current_spectrum[50] = 0  # Very high power
        
        # Should not crash
        visualizer.plot_spectrum(current_spectrum, "Extreme Power Spectrum")
        plt.close('all')  # Clean up
    
    def test_nan_values(self):
        """Test visualization with NaN values."""
        spectrum = RadioSpectrum(num_channels=100)
        detector = SignalDetector(threshold=-70)
        visualizer = SpectrumVisualizer(spectrum, detector)
        
        # Create spectrum with NaN values
        current_spectrum = np.random.uniform(-90, -30, 100)
        current_spectrum[50] = np.nan
        
        # Should handle NaN values gracefully
        visualizer.plot_spectrum(current_spectrum, "NaN Spectrum")
        plt.close('all')  # Clean up


if __name__ == "__main__":
    pytest.main([__file__])
