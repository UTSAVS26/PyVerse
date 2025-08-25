"""
Online Trainer for HoneypotAI
Enables continuous learning from new data streams
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Callable
from datetime import datetime, timedelta
import threading
import time
import queue
import structlog

from .threat_detector import ThreatDetector
from .feature_extractor import FeatureExtractor

logger = structlog.get_logger(__name__)

class OnlineTrainer:
    """Online learning system for continuous model updates"""
    
    def __init__(self, threat_detector: ThreatDetector):
        self.threat_detector = threat_detector
        self.feature_extractor = FeatureExtractor()
        
        self.logger = structlog.get_logger("ml.online_trainer")
        
        # Training configuration
        self.batch_size = 100
        self.retrain_interval = 3600  # 1 hour
        self.min_samples_for_retrain = 50
        self.max_samples_in_memory = 10000
        
        # Training state
        self.is_running = False
        self.training_thread = None
        self.data_queue = queue.Queue()
        self.sample_buffer = []
        self.last_retrain_time = None
        
        # Statistics
        self.training_stats = {
            "total_samples_processed": 0,
            "retrain_count": 0,
            "last_retrain_duration": 0,
            "start_time": None,
            "uptime": 0
        }
        
        # Callbacks
        self.callbacks = {
            "on_retrain": [],
            "on_sample_added": [],
            "on_error": []
        }
    
    def start_online_training(self):
        """Start the online training process"""
        if self.is_running:
            self.logger.warning("Online training is already running")
            return False
        
        try:
            self.is_running = True
            self.training_stats["start_time"] = datetime.now()
            self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
            self.training_thread.start()
            
            self.logger.info("Started online training")
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting online training: {e}")
            self.is_running = False
            return False
    
    def stop_online_training(self):
        """Stop the online training process"""
        if not self.is_running:
            return
        
        self.is_running = False
        if self.training_thread:
            self.training_thread.join(timeout=10)
        
        self.logger.info("Stopped online training")
    
    def add_sample(self, log: Dict[str, Any]):
        """Add a new sample to the training queue"""
        try:
            self.data_queue.put(log)
            self.training_stats["total_samples_processed"] += 1
            
            # Trigger callback
            self._trigger_callbacks("on_sample_added", log)
            
        except Exception as e:
            self.logger.error(f"Error adding sample: {e}")
            self._trigger_callbacks("on_error", str(e))
    
    def add_samples_batch(self, logs: List[Dict[str, Any]]):
        """Add multiple samples to the training queue"""
        for log in logs:
            self.add_sample(log)
    
    def _training_loop(self):
        """Main training loop"""
        while self.is_running:
            try:
                # Collect samples from queue
                self._collect_samples()
                
                # Check if retraining is needed
                if self._should_retrain():
                    self._perform_retrain()
                
                # Sleep for a short interval
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Error in training loop: {e}")
                self._trigger_callbacks("on_error", str(e))
                time.sleep(30)  # Wait longer on error
    
    def _collect_samples(self):
        """Collect samples from the queue"""
        try:
            while not self.data_queue.empty() and len(self.sample_buffer) < self.max_samples_in_memory:
                sample = self.data_queue.get_nowait()
                self.sample_buffer.append(sample)
                
        except queue.Empty:
            pass
        except Exception as e:
            self.logger.error(f"Error collecting samples: {e}")
    
    def _should_retrain(self) -> bool:
        """Check if retraining should be performed"""
        # Check if we have enough samples
        if len(self.sample_buffer) < self.min_samples_for_retrain:
            return False
        
        # Check if enough time has passed since last retrain
        if self.last_retrain_time:
            time_since_retrain = (datetime.now() - self.last_retrain_time).total_seconds()
            if time_since_retrain < self.retrain_interval:
                return False
        
        return True
    
    def _perform_retrain(self):
        """Perform model retraining"""
        try:
            start_time = time.time()
            
            # Get current batch of samples
            batch = self.sample_buffer[:self.batch_size]
            self.sample_buffer = self.sample_buffer[self.batch_size:]
            
            # Extract features
            features = self.feature_extractor.extract_features(batch)
            if features.empty:
                self.logger.warning("No features extracted for retraining")
                return
            
            # Normalize features
            features_normalized = self.feature_extractor.normalize_features(features)
            
            # Retrain models
            self._retrain_anomaly_detector(features_normalized)
            self._retrain_attack_classifier(features_normalized, batch)
            
            # Update statistics
            self.training_stats["retrain_count"] += 1
            self.training_stats["last_retrain_duration"] = time.time() - start_time
            self.last_retrain_time = datetime.now()
            
            self.logger.info(f"Retrained models with {len(batch)} samples in {self.training_stats['last_retrain_duration']:.2f}s")
            
            # Trigger callback
            self._trigger_callbacks("on_retrain", {
                "samples_used": len(batch),
                "duration": self.training_stats["last_retrain_duration"],
                "timestamp": self.last_retrain_time
            })
            
        except Exception as e:
            self.logger.error(f"Error during retraining: {e}")
            self._trigger_callbacks("on_error", str(e))
    
    def _retrain_anomaly_detector(self, features: pd.DataFrame):
        """Retrain the anomaly detector"""
        try:
            if not self.threat_detector.anomaly_detector.is_trained:
                # Initial training
                self.threat_detector.anomaly_detector.train(features)
            else:
                # Online update (simplified - in practice you'd use incremental learning)
                # For now, we'll retrain from scratch with accumulated data
                self.threat_detector.anomaly_detector.train(features)
                
        except Exception as e:
            self.logger.error(f"Error retraining anomaly detector: {e}")
    
    def _retrain_attack_classifier(self, features: pd.DataFrame, logs: List[Dict[str, Any]]):
        """Retrain the attack classifier"""
        try:
            if not self.threat_detector.attack_classifier.is_trained:
                # Initial training
                self.threat_detector.attack_classifier.train(features, logs)
            else:
                # Online update (simplified - in practice you'd use incremental learning)
                # For now, we'll retrain from scratch with accumulated data
                self.threat_detector.attack_classifier.train(features, logs)
                
        except Exception as e:
            self.logger.error(f"Error retraining attack classifier: {e}")
    
    def register_callback(self, event: str, callback: Callable):
        """Register a callback for training events"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
        else:
            self.logger.warning(f"Unknown callback event: {event}")
    
    def _trigger_callbacks(self, event: str, data: Any):
        """Trigger registered callbacks"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    callback(data)
                except Exception as e:
                    self.logger.error(f"Error in callback: {e}")
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get online training statistics"""
        stats = self.training_stats.copy()
        
        # Calculate uptime
        if stats["start_time"]:
            stats["uptime"] = (datetime.now() - stats["start_time"]).total_seconds()
        
        # Add current state
        stats["is_running"] = self.is_running
        stats["queue_size"] = self.data_queue.qsize()
        stats["buffer_size"] = len(self.sample_buffer)
        stats["last_retrain_time"] = self.last_retrain_time.isoformat() if self.last_retrain_time else None
        
        return stats
    
    def set_batch_size(self, batch_size: int):
        """Set the batch size for training"""
        if batch_size > 0:
            self.batch_size = batch_size
            self.logger.info(f"Set batch size to {batch_size}")
        else:
            self.logger.error(f"Invalid batch size: {batch_size}")
    
    def set_retrain_interval(self, interval_seconds: int):
        """Set the retraining interval"""
        if interval_seconds > 0:
            self.retrain_interval = interval_seconds
            self.logger.info(f"Set retrain interval to {interval_seconds} seconds")
        else:
            self.logger.error(f"Invalid retrain interval: {interval_seconds}")
    
    def set_min_samples_for_retrain(self, min_samples: int):
        """Set minimum samples required for retraining"""
        if min_samples > 0:
            self.min_samples_for_retrain = min_samples
            self.logger.info(f"Set minimum samples for retrain to {min_samples}")
        else:
            self.logger.error(f"Invalid minimum samples: {min_samples}")
    
    def clear_buffer(self):
        """Clear the sample buffer"""
        self.sample_buffer.clear()
        self.logger.info("Cleared sample buffer")
    
    def force_retrain(self):
        """Force immediate retraining"""
        if self.is_running and self.sample_buffer:
            self.logger.info("Forcing immediate retraining")
            self._perform_retrain()
        else:
            self.logger.warning("Cannot force retrain - not running or no samples")
    
    def is_running(self) -> bool:
        """Check if online training is running"""
        return self.is_running
