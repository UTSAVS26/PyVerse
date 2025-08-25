"""
Feature Extractor for HoneypotAI
Extracts and processes features from connection logs for ML models
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib
import re
import structlog

logger = structlog.get_logger(__name__)

class FeatureExtractor:
    """Extracts features from connection logs for machine learning models"""
    
    def __init__(self):
        self.logger = structlog.get_logger("ml.feature_extractor")
        
        # Feature categories
        self.basic_features = [
            'payload_size', 'connection_duration', 'success'
        ]
        
        self.network_features = [
            'source_port', 'service_type', 'protocol'
        ]
        
        self.temporal_features = [
            'hour_of_day', 'day_of_week', 'is_weekend', 'time_since_midnight'
        ]
        
        self.behavioral_features = [
            'request_frequency', 'unique_ips', 'payload_entropy', 'command_complexity'
        ]
        
        self.security_features = [
            'attack_type', 'confidence', 'suspicious_patterns'
        ]
    
    def extract_features(self, logs: List[Dict[str, Any]]) -> pd.DataFrame:
        """Extract features from connection logs"""
        if not logs:
            return pd.DataFrame()
        
        features = []
        for log in logs:
            feature_vector = self._extract_single_log_features(log)
            features.append(feature_vector)
        
        return pd.DataFrame(features)
    
    def _extract_single_log_features(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract features from a single connection log"""
        features = {}
        
        # Basic features
        features.update(self._extract_basic_features(log))
        
        # Network features
        features.update(self._extract_network_features(log))
        
        # Temporal features
        features.update(self._extract_temporal_features(log))
        
        # Behavioral features
        features.update(self._extract_behavioral_features(log))
        
        # Security features
        features.update(self._extract_security_features(log))
        
        return features
    
    def _extract_basic_features(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract basic connection features"""
        features = {}
        
        # Payload size
        features['payload_size'] = log.get('payload_size', 0)
        features['payload_size_log'] = np.log1p(features['payload_size'])
        
        # Connection duration
        features['connection_duration'] = log.get('connection_duration', 0.0)
        features['connection_duration_log'] = np.log1p(features['connection_duration'])
        
        # Success flag
        features['success'] = 1 if log.get('success', False) else 0
        
        return features
    
    def _extract_network_features(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract network-related features"""
        features = {}
        
        # Source port
        features['source_port'] = log.get('source_port', 0)
        features['is_privileged_port'] = 1 if features['source_port'] < 1024 else 0
        features['is_common_port'] = 1 if features['source_port'] in [80, 443, 22, 21, 25, 53] else 0
        
        # Service type encoding
        service = log.get('service', 'unknown').lower()
        features['service_ssh'] = 1 if service == 'ssh' else 0
        features['service_http'] = 1 if service == 'http' else 0
        features['service_ftp'] = 1 if service == 'ftp' else 0
        
        # Protocol (derived from service)
        features['protocol_tcp'] = 1  # All our services use TCP
        
        return features
    
    def _extract_temporal_features(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract temporal features from timestamp"""
        features = {}
        
        try:
            timestamp = datetime.fromisoformat(log.get('timestamp', ''))
            
            # Hour of day (0-23)
            features['hour_of_day'] = timestamp.hour
            features['hour_sin'] = np.sin(2 * np.pi * timestamp.hour / 24)
            features['hour_cos'] = np.cos(2 * np.pi * timestamp.hour / 24)
            
            # Day of week (0-6, Monday=0)
            features['day_of_week'] = timestamp.weekday()
            features['day_sin'] = np.sin(2 * np.pi * timestamp.weekday() / 7)
            features['day_cos'] = np.cos(2 * np.pi * timestamp.weekday() / 7)
            
            # Weekend flag
            features['is_weekend'] = 1 if timestamp.weekday() >= 5 else 0
            
            # Time since midnight (seconds)
            midnight = timestamp.replace(hour=0, minute=0, second=0, microsecond=0)
            features['time_since_midnight'] = (timestamp - midnight).total_seconds()
            
        except (ValueError, TypeError):
            # Default values if timestamp parsing fails
            features['hour_of_day'] = 12
            features['hour_sin'] = 0
            features['hour_cos'] = 1
            features['day_of_week'] = 0
            features['day_sin'] = 0
            features['day_cos'] = 1
            features['is_weekend'] = 0
            features['time_since_midnight'] = 43200  # 12 hours in seconds
        
        return features
    
    def _extract_behavioral_features(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract behavioral features"""
        features = {}
        
        # Payload entropy (if payload hash is available)
        payload_hash = log.get('payload_hash', '')
        features['payload_entropy'] = self._calculate_entropy(payload_hash)
        
        # Command complexity (for SSH/FTP)
        service = log.get('service', '').lower()
        if service in ['ssh', 'ftp']:
            features['command_complexity'] = self._calculate_command_complexity(log)
        else:
            features['command_complexity'] = 0
        
        # Request frequency (simplified - would need context in real implementation)
        features['request_frequency'] = 1.0  # Default value
        
        # Unique IPs (simplified - would need context in real implementation)
        features['unique_ips'] = 1.0  # Default value
        
        return features
    
    def _extract_security_features(self, log: Dict[str, Any]) -> Dict[str, Any]:
        """Extract security-related features"""
        features = {}
        
        # Attack type encoding
        attack_type = log.get('attack_type', 'none').lower()
        attack_types = ['brute_force', 'sql_injection', 'xss', 'path_traversal', 
                       'command_injection', 'scanning', 'dos', 'anonymous_access']
        
        for attack in attack_types:
            features[f'attack_{attack}'] = 1 if attack_type == attack else 0
        
        # Confidence score
        features['confidence'] = log.get('confidence', 0.0)
        
        # Suspicious patterns
        features['suspicious_patterns'] = self._detect_suspicious_patterns(log)
        
        # Is attack flag
        features['is_attack'] = 1 if attack_type != 'none' else 0
        
        return features
    
    def _calculate_entropy(self, text: str) -> float:
        """Calculate Shannon entropy of a string"""
        if not text:
            return 0.0
        
        # Count character frequencies
        char_counts = {}
        for char in text:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        # Calculate entropy
        length = len(text)
        entropy = 0.0
        
        for count in char_counts.values():
            probability = count / length
            if probability > 0:
                entropy -= probability * np.log2(probability)
        
        return entropy
    
    def _calculate_command_complexity(self, log: Dict[str, Any]) -> float:
        """Calculate complexity of commands in SSH/FTP logs"""
        # This is a simplified implementation
        # In a real system, you'd analyze the actual commands
        
        # Simulate complexity based on payload size and service
        payload_size = log.get('payload_size', 0)
        service = log.get('service', '').lower()
        
        if service == 'ssh':
            # SSH commands tend to be more complex
            return min(payload_size / 100.0, 10.0)
        elif service == 'ftp':
            # FTP commands are usually simple
            return min(payload_size / 50.0, 5.0)
        else:
            return 0.0
    
    def _detect_suspicious_patterns(self, log: Dict[str, Any]) -> int:
        """Detect suspicious patterns in the connection"""
        patterns = 0
        
        # Check for suspicious payload sizes
        payload_size = log.get('payload_size', 0)
        if payload_size > 10000:  # Very large payload
            patterns += 1
        
        # Check for very short connections (potential scanning)
        duration = log.get('connection_duration', 0)
        if duration < 0.1:  # Very short connection
            patterns += 1
        
        # Check for non-standard ports
        source_port = log.get('source_port', 0)
        if source_port not in [80, 443, 22, 21, 25, 53, 8080, 8443]:
            patterns += 1
        
        # Check for failed connections
        if not log.get('success', False):
            patterns += 1
        
        return patterns
    
    def get_feature_names(self) -> List[str]:
        """Get list of all feature names"""
        # This would return all possible feature names
        # In practice, you'd generate this dynamically based on your feature extraction
        return [
            'payload_size', 'payload_size_log', 'connection_duration', 'connection_duration_log',
            'success', 'source_port', 'is_privileged_port', 'is_common_port',
            'service_ssh', 'service_http', 'service_ftp', 'protocol_tcp',
            'hour_of_day', 'hour_sin', 'hour_cos', 'day_of_week', 'day_sin', 'day_cos',
            'is_weekend', 'time_since_midnight', 'payload_entropy', 'command_complexity',
            'request_frequency', 'unique_ips', 'confidence', 'suspicious_patterns',
            'is_attack'
        ] + [f'attack_{attack}' for attack in ['brute_force', 'sql_injection', 'xss', 
                                               'path_traversal', 'command_injection', 
                                               'scanning', 'dos', 'anonymous_access']]
    
    def normalize_features(self, features: pd.DataFrame) -> pd.DataFrame:
        """Normalize features for better ML performance"""
        normalized = features.copy()
        
        # Normalize numerical features
        numerical_features = [
            'payload_size', 'payload_size_log', 'connection_duration', 'connection_duration_log',
            'source_port', 'hour_of_day', 'time_since_midnight', 'payload_entropy',
            'command_complexity', 'request_frequency', 'unique_ips', 'confidence',
            'suspicious_patterns'
        ]
        
        for feature in numerical_features:
            if feature in normalized.columns:
                # Min-max normalization
                min_val = normalized[feature].min()
                max_val = normalized[feature].max()
                if max_val > min_val:
                    normalized[feature] = (normalized[feature] - min_val) / (max_val - min_val)
        
        return normalized
    
    def create_aggregated_features(self, logs: List[Dict[str, Any]], 
                                 window_minutes: int = 60) -> pd.DataFrame:
        """Create aggregated features over time windows"""
        if not logs:
            return pd.DataFrame()
        
        # Convert logs to DataFrame
        df = pd.DataFrame(logs)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Create time windows
        df['window'] = df['timestamp'].dt.floor(f'{window_minutes}min')
        
        # Aggregate features
        aggregated = df.groupby('window').agg({
            'source_ip': 'nunique',  # Unique IPs
            'payload_size': ['mean', 'std', 'sum'],
            'connection_duration': ['mean', 'std'],
            'success': 'sum',
            'attack_type': lambda x: (x != 'none').sum(),  # Attack count
            'confidence': 'mean'
        }).reset_index()
        
        # Flatten column names
        aggregated.columns = ['window', 'unique_ips', 'avg_payload_size', 
                            'std_payload_size', 'total_payload_size', 'avg_duration',
                            'std_duration', 'successful_connections', 'attack_count',
                            'avg_confidence']
        
        return aggregated
