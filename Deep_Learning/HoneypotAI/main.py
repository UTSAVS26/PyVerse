#!/usr/bin/env python3
"""
HoneypotAI - Advanced Adaptive Cybersecurity Intelligence Platform
Main Application Entry Point
"""

import argparse
import sys
import os
import signal
import time
import threading
from typing import Dict, Any
import structlog

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from honeypot import HoneypotManager
from ml import ThreatDetector, OnlineTrainer
from adapt import AdaptiveResponse

# Configure logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

class HoneypotAI:
    """Main HoneypotAI application class"""
    
    def __init__(self):
        self.honeypot_manager = HoneypotManager()
        self.threat_detector = ThreatDetector()
        self.online_trainer = OnlineTrainer(self.threat_detector)
        self.adaptive_response = AdaptiveResponse()
        
        self.running = False
        self.shutdown_event = threading.Event()
        
        # Register signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, shutting down...")
        self.shutdown_event.set()
        self.stop()
    
    def start(self, config: Dict[str, Any] = None) -> bool:
        """Start the HoneypotAI system"""
        try:
            logger.info("Starting HoneypotAI Advanced Cybersecurity Intelligence Platform")
            
            # Initialize configuration
            if config is None:
                config = self._get_default_config()
            
            # Start honeypot services
            logger.info("Deploying honeypot services...")
            if not self.honeypot_manager.deploy_all_services():
                logger.error("Failed to deploy honeypot services")
                return False
            
            # Configure threat detection
            logger.info("Configuring threat detection...")
            self.threat_detector.setup_anomaly_detection(
                sensitivity=config.get('anomaly_sensitivity', 0.8)
            )
            self.threat_detector.setup_attack_classification(
                confidence_threshold=config.get('classification_confidence', 0.9)
            )
            
            # Configure adaptive response
            logger.info("Configuring adaptive response...")
            self.adaptive_response.set_blocking_strategy(config.get('blocking_strategy', 'dynamic'))
            self.adaptive_response.set_throttling_enabled(config.get('throttling_enabled', True))
            self.adaptive_response.set_decoy_responses(config.get('decoy_responses', True))
            
            # Register callbacks
            self.honeypot_manager.register_callback("connection", self._on_connection)
            self.online_trainer.register_callback("on_retrain", self._on_model_retrain)
            
            # Start online training
            logger.info("Starting online training...")
            if not self.online_trainer.start_online_training():
                logger.error("Failed to start online training")
                return False
            
            # Start adaptive defense
            logger.info("Starting adaptive defense...")
            if not self.adaptive_response.start_adaptive_defense():
                logger.error("Failed to start adaptive defense")
                return False
            
            self.running = True
            logger.info("HoneypotAI system started successfully")
            
            # Start monitoring thread
            self._start_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"Error starting HoneypotAI: {e}")
            return False
    
    def stop(self):
        """Stop the HoneypotAI system"""
        try:
            logger.info("Stopping HoneypotAI system...")
            
            self.running = False
            self.shutdown_event.set()
            
            # Stop online training
            self.online_trainer.stop_online_training()
            
            # Stop adaptive defense
            self.adaptive_response.stop_adaptive_defense()
            
            # Stop honeypot services
            self.honeypot_manager.stop_all_services()
            
            logger.info("HoneypotAI system stopped")
            
        except Exception as e:
            logger.error(f"Error stopping HoneypotAI: {e}")
    
    def _start_monitoring(self):
        """Start the monitoring thread"""
        def monitor():
            while self.running and not self.shutdown_event.is_set():
                try:
                    # Get system status
                    honeypot_stats = self.honeypot_manager.get_overall_stats()
                    ml_stats = self.threat_detector.get_detection_stats()
                    training_stats = self.online_trainer.get_training_stats()
                    
                    # Log status periodically
                    logger.info("System Status", 
                              honeypot_connections=honeypot_stats.get('total_connections', 0),
                              ml_detections=ml_stats.get('total_detections', 0),
                              training_samples=training_stats.get('total_samples_processed', 0))
                    
                    time.sleep(60)  # Log every minute
                    
                except Exception as e:
                    logger.error(f"Error in monitoring: {e}")
                    time.sleep(30)
        
        monitor_thread = threading.Thread(target=monitor, daemon=True)
        monitor_thread.start()
    
    def _on_connection(self, connection_log):
        """Handle new connection events"""
        try:
            # Add to online training
            self.online_trainer.add_sample(connection_log)
            
            # Detect threats
            threats = self.threat_detector.detect_threats([connection_log])
            
            # Trigger adaptive response for threats
            for threat in threats:
                self.adaptive_response.handle_threat(threat)
                
        except Exception as e:
            logger.error(f"Error handling connection: {e}")
    
    def _on_model_retrain(self, retrain_data):
        """Handle model retraining events"""
        try:
            logger.info("Models retrained", 
                       samples_used=retrain_data.get('samples_used', 0),
                       duration=retrain_data.get('duration', 0))
            
            # Save updated models
            self.threat_detector.save_models("models/")
            
        except Exception as e:
            logger.error(f"Error handling model retrain: {e}")
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'anomaly_sensitivity': 0.8,
            'classification_confidence': 0.9,
            'blocking_strategy': 'dynamic',
            'throttling_enabled': True,
            'decoy_responses': True,
            'online_training': {
                'batch_size': 100,
                'retrain_interval': 3600,
                'min_samples': 50
            }
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get system status"""
        return {
            'running': self.running,
            'honeypot': self.honeypot_manager.get_overall_stats(),
            'threat_detection': self.threat_detector.get_detection_stats(),
            'online_training': self.online_trainer.get_training_stats(),
            'adaptive_response': self.adaptive_response.get_status()
        }
    
    def wait_for_shutdown(self):
        """Wait for shutdown signal"""
        try:
            while self.running and not self.shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            self.stop()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="HoneypotAI - Advanced Adaptive Cybersecurity Intelligence Platform"
    )
    parser.add_argument(
        '--start-honeypot',
        action='store_true',
        help='Start the honeypot environment'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to configuration file'
    )
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    parser.add_argument(
        '--daemon',
        action='store_true',
        help='Run as daemon'
    )
    
    args = parser.parse_args()
    
    # Set log level
    import logging
    logging.basicConfig(level=getattr(logging, args.log_level))
    
    # Create and start HoneypotAI
    honeypot_ai = HoneypotAI()
    
    try:
        if args.start_honeypot:
            if honeypot_ai.start():
                logger.info("HoneypotAI started successfully")
                if args.daemon:
                    honeypot_ai.wait_for_shutdown()
                else:
                    logger.info("Press Ctrl+C to stop")
                    honeypot_ai.wait_for_shutdown()
            else:
                logger.error("Failed to start HoneypotAI")
                sys.exit(1)
        else:
            logger.info("Use --start-honeypot to start the honeypot environment")
            logger.info("Use --help for more options")
    
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        honeypot_ai.stop()

if __name__ == "__main__":
    main()
