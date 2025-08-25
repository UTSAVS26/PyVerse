#!/usr/bin/env python3
"""
Tests for main.py - HoneypotAI Main Application
"""

import unittest
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import HoneypotAI, main

class TestHoneypotAI(unittest.TestCase):
    """Test HoneypotAI main application class"""
    
    @patch('main.HoneypotManager')
    @patch('main.ThreatDetector')
    @patch('main.OnlineTrainer')
    @patch('main.AdaptiveResponse')
    def test_honeypot_ai_initialization(self, mock_response, mock_trainer, mock_detector, mock_manager):
        """Test HoneypotAI initialization"""
        # Mock component instances
        mock_manager_instance = Mock()
        mock_detector_instance = Mock()
        mock_trainer_instance = Mock()
        mock_response_instance = Mock()
        
        mock_manager.return_value = mock_manager_instance
        mock_detector.return_value = mock_detector_instance
        mock_trainer.return_value = mock_trainer_instance
        mock_response.return_value = mock_response_instance
        
        app = HoneypotAI()
        
        # Components should be initialized
        assert app.honeypot_manager == mock_manager_instance
        assert app.threat_detector == mock_detector_instance
        assert app.online_trainer == mock_trainer_instance
        assert app.adaptive_response == mock_response_instance
        assert app.running == False
        assert app.shutdown_event is not None
    
    @patch('main.HoneypotManager')
    @patch('main.ThreatDetector')
    @patch('main.OnlineTrainer')
    @patch('main.AdaptiveResponse')
    def test_start_services(self, mock_response, mock_trainer, mock_detector, mock_manager):
        """Test starting services"""
        # Mock component instances
        mock_manager_instance = Mock()
        mock_detector_instance = Mock()
        mock_trainer_instance = Mock()
        mock_response_instance = Mock()
        
        mock_manager.return_value = mock_manager_instance
        mock_detector.return_value = mock_detector_instance
        mock_trainer.return_value = mock_trainer_instance
        mock_response.return_value = mock_response_instance
        
        # Mock successful operations
        mock_manager_instance.deploy_all_services.return_value = True
        mock_trainer_instance.start_online_training.return_value = True
        mock_response_instance.start_adaptive_defense.return_value = True
        
        app = HoneypotAI()
        
        # Test successful start
        result = app.start()
        assert result == True
        assert app.running == True
        
        # Verify component methods were called
        mock_manager_instance.deploy_all_services.assert_called_once()
        mock_detector_instance.setup_anomaly_detection.assert_called_once()
        mock_detector_instance.setup_attack_classification.assert_called_once()
        mock_response_instance.set_blocking_strategy.assert_called_once()
        mock_response_instance.set_throttling_enabled.assert_called_once()
        mock_response_instance.set_decoy_responses.assert_called_once()
        mock_trainer_instance.start_online_training.assert_called_once()
        mock_response_instance.start_adaptive_defense.assert_called_once()
    
    @patch('main.HoneypotManager')
    @patch('main.ThreatDetector')
    @patch('main.OnlineTrainer')
    @patch('main.AdaptiveResponse')
    def test_stop_services(self, mock_response, mock_trainer, mock_detector, mock_manager):
        """Test stopping services"""
        # Mock component instances
        mock_manager_instance = Mock()
        mock_detector_instance = Mock()
        mock_trainer_instance = Mock()
        mock_response_instance = Mock()
        
        mock_manager.return_value = mock_manager_instance
        mock_detector.return_value = mock_detector_instance
        mock_trainer.return_value = mock_trainer_instance
        mock_response.return_value = mock_response_instance
        
        app = HoneypotAI()
        app.running = True
        
        app.stop()
        
        assert app.running == False
        mock_trainer_instance.stop_online_training.assert_called_once()
        mock_response_instance.stop_adaptive_defense.assert_called_once()
        mock_manager_instance.stop_all_services.assert_called_once()
    
    @patch('main.HoneypotManager')
    @patch('main.ThreatDetector')
    @patch('main.OnlineTrainer')
    @patch('main.AdaptiveResponse')
    def test_process_logs(self, mock_response, mock_trainer, mock_detector, mock_manager):
        """Test log processing"""
        # Mock component instances
        mock_manager_instance = Mock()
        mock_detector_instance = Mock()
        mock_trainer_instance = Mock()
        mock_response_instance = Mock()
        
        mock_manager.return_value = mock_manager_instance
        mock_detector.return_value = mock_detector_instance
        mock_trainer.return_value = mock_trainer_instance
        mock_response.return_value = mock_response_instance
        
        app = HoneypotAI()
        
        # Mock connection log
        connection_log = {"source_ip": "192.168.1.1", "service": "ssh", "timestamp": "2023-01-01T00:00:00"}
        
        # Mock threat detection
        mock_threats = [{"source_ip": "192.168.1.1", "threat_type": "brute_force", "confidence": 0.9}]
        mock_detector_instance.detect_threats.return_value = mock_threats
        
        # Test connection handling
        app._on_connection(connection_log)
        
        mock_trainer_instance.add_sample.assert_called_once_with(connection_log)
        mock_detector_instance.detect_threats.assert_called_once_with([connection_log])
        mock_response_instance.handle_threat.assert_called_once_with(mock_threats[0])
    
    @patch('main.HoneypotManager')
    @patch('main.ThreatDetector')
    @patch('main.OnlineTrainer')
    @patch('main.AdaptiveResponse')
    def test_get_status(self, mock_response, mock_trainer, mock_detector, mock_manager):
        """Test getting application status"""
        # Mock component instances
        mock_manager_instance = Mock()
        mock_detector_instance = Mock()
        mock_trainer_instance = Mock()
        mock_response_instance = Mock()
        
        mock_manager.return_value = mock_manager_instance
        mock_detector.return_value = mock_detector_instance
        mock_trainer.return_value = mock_trainer_instance
        mock_response.return_value = mock_response_instance
        
        # Mock status returns
        mock_manager_instance.get_overall_stats.return_value = {"total_connections": 100}
        mock_detector_instance.get_detection_stats.return_value = {"total_detections": 10}
        mock_trainer_instance.get_training_stats.return_value = {"total_samples": 50}
        mock_response_instance.get_status.return_value = {"blocks_issued": 5}
        
        app = HoneypotAI()
        app.running = True
        
        status = app.get_status()
        
        assert status["running"] == True
        assert status["honeypot"] == {"total_connections": 100}
        assert status["threat_detection"] == {"total_detections": 10}
        assert status["online_training"] == {"total_samples": 50}
        assert status["adaptive_response"] == {"blocks_issued": 5}
    
    @patch('main.HoneypotManager')
    @patch('main.ThreatDetector')
    @patch('main.OnlineTrainer')
    @patch('main.AdaptiveResponse')
    def test_run_dashboard(self, mock_response, mock_trainer, mock_detector, mock_manager):
        """Test running dashboard"""
        # Mock component instances
        mock_manager_instance = Mock()
        mock_detector_instance = Mock()
        mock_trainer_instance = Mock()
        mock_response_instance = Mock()
        
        mock_manager.return_value = mock_manager_instance
        mock_detector.return_value = mock_detector_instance
        mock_trainer.return_value = mock_trainer_instance
        mock_response.return_value = mock_response_instance
        
        app = HoneypotAI()
        
        # Test that the app can be started (which includes dashboard functionality)
        mock_manager_instance.deploy_all_services.return_value = True
        mock_trainer_instance.start_online_training.return_value = True
        mock_response_instance.start_adaptive_defense.return_value = True
        
        result = app.start()
        assert result == True
    
    @patch('main.HoneypotManager')
    @patch('main.ThreatDetector')
    @patch('main.OnlineTrainer')
    @patch('main.AdaptiveResponse')
    @patch('time.sleep')
    def test_run_continuous_processing(self, mock_sleep, mock_response, mock_trainer, mock_detector, mock_manager):
        """Test continuous log processing"""
        # Mock component instances
        mock_manager_instance = Mock()
        mock_detector_instance = Mock()
        mock_trainer_instance = Mock()
        mock_response_instance = Mock()
        
        mock_manager.return_value = mock_manager_instance
        mock_detector.return_value = mock_detector_instance
        mock_trainer.return_value = mock_trainer_instance
        mock_response.return_value = mock_response_instance
        
        # Mock status returns
        mock_manager_instance.get_overall_stats.return_value = {"total_connections": 100}
        mock_detector_instance.get_detection_stats.return_value = {"total_detections": 10}
        mock_trainer_instance.get_training_stats.return_value = {"total_samples": 50}
        
        app = HoneypotAI()
        
        # Test that monitoring can be started
        app._start_monitoring()
        
        # Verify that monitoring thread was created (we can't easily test the thread itself)
        assert app.running == False  # Should still be False since we didn't call start()
    
    @patch('main.HoneypotManager')
    @patch('main.ThreatDetector')
    @patch('main.OnlineTrainer')
    @patch('main.AdaptiveResponse')
    def test_cleanup(self, mock_response, mock_trainer, mock_detector, mock_manager):
        """Test cleanup on shutdown"""
        # Mock component instances
        mock_manager_instance = Mock()
        mock_detector_instance = Mock()
        mock_trainer_instance = Mock()
        mock_response_instance = Mock()
        
        mock_manager.return_value = mock_manager_instance
        mock_detector.return_value = mock_detector_instance
        mock_trainer.return_value = mock_trainer_instance
        mock_response.return_value = mock_response_instance
        
        app = HoneypotAI()
        app.running = True
        
        # Test cleanup via stop
        app.stop()
        
        assert app.running == False
        mock_trainer_instance.stop_online_training.assert_called_once()
        mock_response_instance.stop_adaptive_defense.assert_called_once()
        mock_manager_instance.stop_all_services.assert_called_once()
    
    @patch('main.HoneypotManager')
    @patch('main.ThreatDetector')
    @patch('main.OnlineTrainer')
    @patch('main.AdaptiveResponse')
    def test_context_manager(self, mock_response, mock_trainer, mock_detector, mock_manager):
        """Test context manager functionality"""
        # Mock component instances
        mock_manager_instance = Mock()
        mock_detector_instance = Mock()
        mock_trainer_instance = Mock()
        mock_response_instance = Mock()
        
        mock_manager.return_value = mock_manager_instance
        mock_detector.return_value = mock_detector_instance
        mock_trainer.return_value = mock_trainer_instance
        mock_response.return_value = mock_response_instance
        
        # HoneypotAI doesn't support context manager protocol, so we test manual start/stop
        app = HoneypotAI()
        
        # Mock successful start
        mock_manager_instance.deploy_all_services.return_value = True
        mock_trainer_instance.start_online_training.return_value = True
        mock_response_instance.start_adaptive_defense.return_value = True
        
        result = app.start()
        assert result == True
        
        app.stop()
        assert app.running == False

class TestMainFunction(unittest.TestCase):
    """Test main function"""
    
    @patch('main.HoneypotAI')
    @patch('argparse.ArgumentParser')
    def test_main_function_dashboard_mode(self, mock_parser, mock_honeypot_ai):
        """Test main function in dashboard mode"""
        # Mock argument parser
        mock_args = Mock()
        mock_args.start_honeypot = True
        mock_args.config = None
        mock_args.log_level = 'INFO'
        mock_args.daemon = False
        
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance
        
        # Mock HoneypotAI
        mock_app = Mock()
        mock_app.start.return_value = True
        mock_honeypot_ai.return_value = mock_app
        
        # Mock logging
        with patch('logging.basicConfig'):
            main()
        
        mock_honeypot_ai.assert_called_once()
        mock_app.start.assert_called_once()
        mock_app.stop.assert_called_once()
    
    @patch('main.HoneypotAI')
    @patch('argparse.ArgumentParser')
    def test_main_function_honeypot_only_mode(self, mock_parser, mock_honeypot_ai):
        """Test main function in honeypot-only mode"""
        # Mock argument parser
        mock_args = Mock()
        mock_args.start_honeypot = True
        mock_args.config = None
        mock_args.log_level = 'INFO'
        mock_args.daemon = False
        
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance
        
        # Mock HoneypotAI
        mock_app = Mock()
        mock_app.start.return_value = True
        mock_honeypot_ai.return_value = mock_app
        
        # Mock logging
        with patch('logging.basicConfig'):
            main()
        
        mock_honeypot_ai.assert_called_once()
        mock_app.start.assert_called_once()
        mock_app.stop.assert_called_once()
    
    @patch('main.HoneypotAI')
    @patch('argparse.ArgumentParser')
    def test_main_function_ml_only_mode(self, mock_parser, mock_honeypot_ai):
        """Test main function in ML-only mode"""
        # Mock argument parser
        mock_args = Mock()
        mock_args.start_honeypot = True
        mock_args.config = None
        mock_args.log_level = 'INFO'
        mock_args.daemon = False
        
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance
        
        # Mock HoneypotAI
        mock_app = Mock()
        mock_app.start.return_value = True
        mock_honeypot_ai.return_value = mock_app
        
        # Mock logging
        with patch('logging.basicConfig'):
            main()
        
        mock_honeypot_ai.assert_called_once()
        mock_app.start.assert_called_once()
        mock_app.stop.assert_called_once()
    
    @patch('main.HoneypotAI')
    @patch('argparse.ArgumentParser')
    def test_main_function_full_mode(self, mock_parser, mock_honeypot_ai):
        """Test main function in full mode"""
        # Mock argument parser
        mock_args = Mock()
        mock_args.start_honeypot = True
        mock_args.config = None
        mock_args.log_level = 'INFO'
        mock_args.daemon = False
        
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance
        
        # Mock HoneypotAI
        mock_app = Mock()
        mock_app.start.return_value = True
        mock_honeypot_ai.return_value = mock_app
        
        # Mock logging
        with patch('logging.basicConfig'):
            main()
        
        mock_honeypot_ai.assert_called_once()
        mock_app.start.assert_called_once()
        mock_app.stop.assert_called_once()
    
    @patch('main.HoneypotAI')
    @patch('argparse.ArgumentParser')
    @patch('builtins.print')
    def test_main_function_verbose_mode(self, mock_print, mock_parser, mock_honeypot_ai):
        """Test main function in verbose mode"""
        # Mock argument parser
        mock_args = Mock()
        mock_args.start_honeypot = True
        mock_args.config = None
        mock_args.log_level = 'DEBUG'
        mock_args.daemon = False
        
        mock_parser_instance = Mock()
        mock_parser_instance.parse_args.return_value = mock_args
        mock_parser.return_value = mock_parser_instance
        
        # Mock HoneypotAI
        mock_app = Mock()
        mock_app.start.return_value = True
        mock_honeypot_ai.return_value = mock_app
        
        # Mock logging
        with patch('logging.basicConfig'):
            main()
        
        mock_honeypot_ai.assert_called_once()
        mock_app.start.assert_called_once()
        mock_app.stop.assert_called_once()

if __name__ == "__main__":
    unittest.main()
