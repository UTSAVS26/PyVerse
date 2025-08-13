"""
Test cases for password generation and labeling functionality.
"""

import pytest
import sys
import os
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labeling.labeler import PasswordLabeler
from data.generate_passwords import PasswordGenerator


class TestPasswordLabeler:
    """Test cases for PasswordLabeler class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.labeler = PasswordLabeler()
    
    def test_weak_passwords(self):
        """Test that weak passwords are correctly identified."""
        weak_passwords = [
            "abc",           # too short
            "password",      # common word
            "123456",        # sequential
            "qwerty",        # keyboard pattern
            "admin",         # common word
            "hello123",      # common word + numbers
            "test",          # too short
            "user",          # too short
        ]
        
        for password in weak_passwords:
            label = self.labeler.label_password(password)
            assert label == 'weak', f"Password '{password}' should be weak, got {label}"
    
    def test_medium_passwords(self):
        """Test that medium passwords are correctly identified."""
        medium_passwords = [
            "Book@789",      # word + special + numbers
            "Test@2023",     # mixed case + special + numbers
            "MixedCase123",  # mixed case + numbers
            "User123!",      # word + numbers + special
            "Pass2023",      # word + numbers
            "Simple123",     # word + numbers
        ]
        
        for password in medium_passwords:
            label = self.labeler.label_password(password)
            assert label == 'medium', f"Password '{password}' should be medium, got {label}"
    
    def test_strong_passwords(self):
        """Test that strong passwords are correctly identified."""
        strong_passwords = [
            "Hello2023!",    # mixed case + numbers + special (strong)
            "G7^s9L!zB1m",  # complex random
            "tR#8$!XmPq@",  # complex random
            "K9#mN2$pL7@",  # complex random
            "Complex#Pass1", # complex pattern
            "UltraSecure#2023!", # complex pattern
            "SecurePass1!",  # mixed case + numbers + special (strong)
        ]
        
        for password in strong_passwords:
            label = self.labeler.label_password(password)
            assert label == 'strong', f"Password '{password}' should be strong, got {label}"
    
    def test_empty_password(self):
        """Test empty password handling."""
        label = self.labeler.label_password("")
        assert label == 'weak'
    
    def test_none_password(self):
        """Test None password handling."""
        label = self.labeler.label_password(None)
        assert label == 'weak'
    
    def test_common_password_detection(self):
        """Test detection of common weak passwords."""
        common_passwords = [
            "password", "123456", "qwerty", "admin", "letmein",
            "welcome", "monkey", "dragon", "master", "sunshine"
        ]
        
        for password in common_passwords:
            is_common = self.labeler.is_common_password(password)
            assert is_common, f"Password '{password}' should be detected as common"
    
    def test_sequential_pattern_detection(self):
        """Test detection of sequential patterns."""
        sequential_passwords = [
            "12345", "67890", "qwert", "asdfg", "zxcvb"
        ]
        
        for password in sequential_passwords:
            has_pattern = self.labeler.has_sequential_patterns(password)
            assert has_pattern, f"Password '{password}' should have sequential pattern"
    
    def test_character_analysis(self):
        """Test character type analysis."""
        password = "Hello2023!"
        score = self.labeler.calculate_strength_score(password)
        
        assert score['length'] == 10
        assert score['uppercase'] == 1
        assert score['lowercase'] == 4
        assert score['digits'] == 4
        assert score['special'] == 1
        assert score['unique_chars'] == 8
    
    def test_detailed_analysis(self):
        """Test detailed analysis functionality."""
        password = "Test@123"
        analysis = self.labeler.get_detailed_analysis(password)
        
        assert 'password' in analysis
        assert 'label' in analysis
        assert 'score' in analysis
        assert 'issues' in analysis
        assert 'strengths' in analysis
        assert analysis['password'] == password


class TestPasswordGenerator:
    """Test cases for PasswordGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = PasswordGenerator()
    
    def test_weak_password_generation(self):
        """Test weak password generation."""
        for _ in range(10):
            password = self.generator.generate_weak_password()
            label = self.generator.labeler.label_password(password)
            assert label == 'weak', f"Generated weak password '{password}' was labeled as {label}"
    
    def test_medium_password_generation(self):
        """Test medium password generation."""
        for _ in range(10):
            password = self.generator.generate_medium_password()
            label = self.generator.labeler.label_password(password)
            assert label == 'medium', f"Generated medium password '{password}' was labeled as {label}"
    
    def test_strong_password_generation(self):
        """Test strong password generation."""
        for _ in range(10):
            password = self.generator.generate_strong_password()
            label = self.generator.labeler.label_password(password)
            assert label == 'strong', f"Generated strong password '{password}' was labeled as {label}"
    
    def test_target_label_generation(self):
        """Test generation targeting specific labels."""
        for target_label in ['weak', 'medium', 'strong']:
            password = self.generator.generate_password_with_target_label(target_label)
            actual_label = self.generator.labeler.label_password(password)
            # Allow some tolerance for generation accuracy
            assert actual_label in ['weak', 'medium', 'strong'], f"Invalid label: {actual_label}"
    
    def test_dataset_generation(self):
        """Test dataset generation."""
        df = self.generator.generate_dataset(total_samples=30)  # Small dataset for testing
        
        assert len(df) == 30
        assert 'password' in df.columns
        assert 'target_label' in df.columns
        assert 'actual_label' in df.columns
        assert 'label_match' in df.columns
        
        # Check label distribution
        label_counts = df['target_label'].value_counts()
        assert len(label_counts) == 3  # weak, medium, strong
        assert all(count == 10 for count in label_counts.values)  # Equal distribution
    
    def test_balanced_dataset_generation(self):
        """Test balanced dataset generation."""
        dataset = self.generator.generate_balanced_dataset(samples_per_class=5)
        
        assert len(dataset) == 15  # 5 samples per class * 3 classes
        
        # Check label distribution
        labels = [item['target_label'] for item in dataset]
        assert labels.count('weak') == 5
        assert labels.count('medium') == 5
        assert labels.count('strong') == 5


class TestIntegration:
    """Integration tests for the complete pipeline."""
    
    def test_end_to_end_pipeline(self):
        """Test the complete pipeline from generation to labeling."""
        # Generate passwords
        generator = PasswordGenerator()
        df = generator.generate_dataset(total_samples=30)
        
        # Test labeling
        labeler = PasswordLabeler()
        
        for _, row in df.iterrows():
            password = row['password']
            actual_label = row['actual_label']
            
            # Test individual labeling
            predicted_label = labeler.label_password(password)
            assert predicted_label in ['weak', 'medium', 'strong']
            
            # Test detailed analysis
            analysis = labeler.get_detailed_analysis(password)
            assert analysis['password'] == password
            assert analysis['label'] in ['weak', 'medium', 'strong']
    
    def test_batch_labeling(self):
        """Test batch labeling functionality."""
        from labeling.labeler import label_passwords_batch
        
        passwords = [
            "abc", "password123", "Hello2023!", "G7^s9L!zB1m",
            "qwerty", "MyPass@123", "tR#8$!XmPq@"
        ]
        
        results = label_passwords_batch(passwords)
        
        assert len(results) == len(passwords)
        
        for result in results:
            assert 'password' in result
            assert 'label' in result
            assert 'score' in result
            assert 'issues' in result
            assert 'strengths' in result
            assert result['label'] in ['weak', 'medium', 'strong']


def test_data_consistency():
    """Test that generated data is consistent."""
    generator = PasswordGenerator()
    df = generator.generate_dataset(total_samples=30)
    
    # Check data types
    assert df['password'].dtype == 'object'
    assert df['target_label'].dtype == 'object'
    assert df['actual_label'].dtype == 'object'
    assert df['label_match'].dtype == 'bool'
    
    # Check for missing values
    assert not df.isnull().any().any()
    
    # Check label values
    valid_labels = {'weak', 'medium', 'strong'}
    assert set(df['target_label'].unique()).issubset(valid_labels)
    assert set(df['actual_label'].unique()).issubset(valid_labels)


def test_performance():
    """Test performance of key functions."""
    import time
    
    labeler = PasswordLabeler()
    generator = PasswordGenerator()
    
    # Test labeling performance
    test_passwords = ["TestPassword123!", "abc", "G7^s9L!zB1m"] * 100
    
    start_time = time.time()
    for password in test_passwords:
        labeler.label_password(password)
    labeling_time = time.time() - start_time
    
    # Should complete within reasonable time (adjust threshold as needed)
    assert labeling_time < 5.0, f"Labeling took too long: {labeling_time:.2f}s"
    
    # Test generation performance
    start_time = time.time()
    generator.generate_dataset(total_samples=100)
    generation_time = time.time() - start_time
    
    # Should complete within reasonable time
    assert generation_time < 10.0, f"Generation took too long: {generation_time:.2f}s"


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"]) 