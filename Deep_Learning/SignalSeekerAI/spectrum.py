import numpy as np
import scipy.signal as signal
from typing import Dict, List, Tuple, Optional
import random


class RadioSpectrum:
    """
    Simulates a radio spectrum environment with signals, noise, and interference.
    """
    
    def __init__(self, 
                 freq_range: Tuple[float, float] = (1e6, 100e6),  # 1-100 MHz
                 num_channels: int = 1000,
                 sample_rate: float = 200e6,  # 200 MHz sampling
                 noise_floor: float = -90,  # dBm
                 signal_types: List[str] = None):
        """
        Initialize the radio spectrum environment.
        
        Args:
            freq_range: Frequency range in Hz (min, max)
            num_channels: Number of frequency channels to simulate
            sample_rate: Sampling rate in Hz
            noise_floor: Noise floor in dBm
            signal_types: Types of signals to generate
        """
        self.freq_range = freq_range
        self.num_channels = num_channels
        self.sample_rate = sample_rate
        self.noise_floor = noise_floor
        self.signal_types = signal_types or ['carrier', 'am', 'fm', 'digital']
        
        # Frequency resolution
        self.freq_resolution = (freq_range[1] - freq_range[0]) / num_channels
        self.frequencies = np.linspace(freq_range[0], freq_range[1], num_channels)
        
        # Spectrum state
        self.spectrum_power = np.full(num_channels, noise_floor, dtype=np.float64)
        self.active_signals = {}
        self.interference_sources = {}
        
        # Time tracking
        self.time = 0.0
        self.time_step = 1e-3  # 1ms time steps
        
        # Signal parameters
        self.signal_params = {
            'carrier': {'power': -30, 'bandwidth': 1e3, 'duration': 1.0},
            'am': {'power': -40, 'bandwidth': 10e3, 'duration': 0.5, 'mod_freq': 1e3},
            'fm': {'power': -35, 'bandwidth': 20e3, 'duration': 0.3, 'deviation': 5e3},
            'digital': {'power': -45, 'bandwidth': 50e3, 'duration': 0.1, 'symbol_rate': 10e3}
        }
        
        # Initialize random signals and interference
        self._generate_initial_signals()
        self._generate_interference_sources()
    
    def _generate_initial_signals(self, num_signals: int = 5):
        """Generate initial random signals in the spectrum."""
        for i in range(num_signals):
            signal_type = random.choice(self.signal_types)
            freq_idx = random.randint(0, self.num_channels - 1)
            freq = self.frequencies[freq_idx]
            
            signal_id = f"signal_{i}_{signal_type}"
            self.active_signals[signal_id] = {
                'type': signal_type,
                'frequency': freq,
                'freq_idx': freq_idx,
                'power': self.signal_params[signal_type]['power'],
                'bandwidth': self.signal_params[signal_type]['bandwidth'],
                'start_time': self.time,
                'duration': self.signal_params[signal_type]['duration'],
                'params': self.signal_params[signal_type].copy()
            }
    
    def _generate_interference_sources(self, num_sources: int = 3):
        """Generate interference sources that change over time."""
        for i in range(num_sources):
            source_id = f"interference_{i}"
            freq_idx = random.randint(0, self.num_channels - 1)
            freq = self.frequencies[freq_idx]
            
            self.interference_sources[source_id] = {
                'frequency': freq,
                'freq_idx': freq_idx,
                'power': random.uniform(-60, -20),
                'bandwidth': random.uniform(1e3, 50e3),
                'change_rate': random.uniform(0.1, 2.0),  # Hz
                'phase': random.uniform(0, 2 * np.pi)
            }
    
    def step(self, dt: float = None) -> np.ndarray:
        """
        Advance the simulation by one time step and return current spectrum.
        
        Args:
            dt: Time step in seconds (uses default if None)
            
        Returns:
            Current spectrum power levels in dBm
        """
        if dt is None:
            dt = self.time_step
        
        self.time += dt
        
        # Reset spectrum to noise floor
        self.spectrum_power = np.full(self.num_channels, self.noise_floor, dtype=np.float64)
        
        # Update and add active signals
        self._update_signals(dt)
        
        # Update and add interference
        self._update_interference(dt)
        
        # Add thermal noise
        self._add_thermal_noise()
        
        return self.spectrum_power.copy()
    
    def _update_signals(self, dt: float):
        """Update active signals and remove expired ones."""
        expired_signals = []
        
        for signal_id, signal_info in self.active_signals.items():
            # Check if signal has expired
            if self.time - signal_info['start_time'] > signal_info['duration']:
                expired_signals.append(signal_id)
                continue
            
            # Add signal power to spectrum
            self._add_signal_to_spectrum(signal_info)
        
        # Remove expired signals
        for signal_id in expired_signals:
            del self.active_signals[signal_id]
        
        # Randomly generate new signals
        if random.random() < 0.1:  # 10% chance per time step
            self._generate_new_signal()
    
    def _update_interference(self, dt: float):
        """Update interference sources."""
        for source_id, source_info in self.interference_sources.items():
            # Update phase
            source_info['phase'] += 2 * np.pi * source_info['change_rate'] * dt
            
            # Vary power slightly
            power_variation = 5 * np.sin(source_info['phase'])
            current_power = source_info['power'] + power_variation
            
            # Add interference to spectrum
            self._add_interference_to_spectrum(source_info, current_power)
    
    def _add_signal_to_spectrum(self, signal_info: Dict):
        """Add a signal to the spectrum."""
        freq_idx = signal_info['freq_idx']
        power = signal_info['power']
        bandwidth = signal_info['bandwidth']
        
        # Calculate bandwidth in channel indices
        bw_channels = int(bandwidth / self.freq_resolution)
        start_idx = max(0, freq_idx - bw_channels // 2)
        end_idx = min(self.num_channels, freq_idx + bw_channels // 2)
        
        # Add signal power to affected channels
        for i in range(start_idx, end_idx):
            # Simple gaussian-like power distribution
            distance = abs(i - freq_idx)
            if distance <= bw_channels // 2:
                power_reduction = (distance / (bw_channels // 2)) ** 2
                channel_power = power - 3 * power_reduction
                self.spectrum_power[i] = self._combine_powers(
                    self.spectrum_power[i], channel_power
                )
    
    def _add_interference_to_spectrum(self, source_info: Dict, power: float):
        """Add interference to the spectrum."""
        freq_idx = source_info['freq_idx']
        bandwidth = source_info['bandwidth']
        
        # Calculate bandwidth in channel indices
        bw_channels = int(bandwidth / self.freq_resolution)
        start_idx = max(0, freq_idx - bw_channels // 2)
        end_idx = min(self.num_channels, freq_idx + bw_channels // 2)
        
        # Add interference power to affected channels
        for i in range(start_idx, end_idx):
            distance = abs(i - freq_idx)
            if distance <= bw_channels // 2:
                power_reduction = (distance / (bw_channels // 2)) ** 2
                channel_power = power - 3 * power_reduction
                self.spectrum_power[i] = self._combine_powers(
                    self.spectrum_power[i], channel_power
                )
    
    def _add_thermal_noise(self):
        """Add thermal noise to the spectrum."""
        # Add random noise variation
        noise_variation = np.random.normal(0, 2, self.num_channels)
        self.spectrum_power += noise_variation
    
    def _combine_powers(self, power1: float, power2: float) -> float:
        """Combine two power levels in dBm."""
        # Convert to linear scale, add, convert back to dBm
        linear1 = 10 ** (power1 / 10)
        linear2 = 10 ** (power2 / 10)
        combined_linear = linear1 + linear2
        return 10 * np.log10(combined_linear)
    
    def _generate_new_signal(self):
        """Generate a new random signal."""
        signal_type = random.choice(self.signal_types)
        freq_idx = random.randint(0, self.num_channels - 1)
        freq = self.frequencies[freq_idx]
        
        signal_id = f"signal_{len(self.active_signals)}_{signal_type}_{int(self.time)}"
        self.active_signals[signal_id] = {
            'type': signal_type,
            'frequency': freq,
            'freq_idx': freq_idx,
            'power': self.signal_params[signal_type]['power'],
            'bandwidth': self.signal_params[signal_type]['bandwidth'],
            'start_time': self.time,
            'duration': self.signal_params[signal_type]['duration'],
            'params': self.signal_params[signal_type].copy()
        }
    
    def get_spectrum_slice(self, start_freq: float, end_freq: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get a slice of the spectrum between specified frequencies.
        
        Args:
            start_freq: Start frequency in Hz
            end_freq: End frequency in Hz
            
        Returns:
            Tuple of (frequencies, power_levels) for the specified range
        """
        start_idx = np.searchsorted(self.frequencies, start_freq)
        end_idx = np.searchsorted(self.frequencies, end_freq)
        
        return (self.frequencies[start_idx:end_idx], 
                self.spectrum_power[start_idx:end_idx])
    
    def get_signal_info(self) -> Dict:
        """Get information about currently active signals."""
        return {
            'num_signals': len(self.active_signals),
            'signals': self.active_signals.copy(),
            'num_interference': len(self.interference_sources),
            'time': self.time
        }
    
    def reset(self):
        """Reset the spectrum to initial state."""
        self.time = 0.0
        self.spectrum_power = np.full(self.num_channels, self.noise_floor, dtype=np.float64)
        self.active_signals = {}
        self.interference_sources = {}
        self._generate_initial_signals()
        self._generate_interference_sources()


class SignalDetector:
    """
    Detects signals in the spectrum using various methods.
    """
    
    def __init__(self, threshold: float = -70):
        """
        Initialize the signal detector.
        
        Args:
            threshold: Detection threshold in dBm
        """
        self.threshold = threshold
    
    def detect_signals(self, spectrum: np.ndarray, frequencies: np.ndarray) -> List[Dict]:
        """
        Detect signals in the spectrum above the threshold.
        
        Args:
            spectrum: Power spectrum in dBm
            frequencies: Corresponding frequencies in Hz
            
        Returns:
            List of detected signals with frequency and power information
        """
        detected_signals = []
        
        # Handle empty spectrum
        if len(spectrum) == 0:
            return detected_signals
        
        # Find peaks above threshold
        if len(spectrum) == 1:
            # Handle single point spectrum
            if spectrum[0] > self.threshold:
                peaks = [0]
            else:
                peaks = []
        else:
            # Find peaks with minimum distance (ensure distance >= 1)
            distance = max(1, min(5, len(spectrum)//2))
            peaks, _ = signal.find_peaks(spectrum, height=self.threshold, distance=distance)
        
        for peak_idx in peaks:
            signal_info = {
                'frequency': frequencies[peak_idx],
                'freq_idx': peak_idx,
                'power': spectrum[peak_idx],
                'bandwidth': self._estimate_bandwidth(spectrum, peak_idx, frequencies)
            }
            detected_signals.append(signal_info)
        
        return detected_signals
    
    def _estimate_bandwidth(self, spectrum: np.ndarray, peak_idx: int, frequencies: np.ndarray) -> float:
        """Estimate the bandwidth of a detected signal."""
        peak_power = spectrum[peak_idx]
        threshold = peak_power - 3  # 3dB down point
        
        # Find left edge
        left_idx = peak_idx
        while left_idx > 0 and spectrum[left_idx] > threshold:
            left_idx -= 1
        
        # Find right edge
        right_idx = peak_idx
        while right_idx < len(spectrum) - 1 and spectrum[right_idx] > threshold:
            right_idx += 1
        
        return frequencies[right_idx] - frequencies[left_idx]
