"""
Password Generator

This module generates synthetic passwords with varying complexity levels
for training the password strength classifier.
"""

import random
import string
import pandas as pd
from typing import List, Dict
import sys
import os

# Add parent directory to path to import labeling module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from labeling.labeler import PasswordLabeler


class PasswordGenerator:
    """Generates synthetic passwords with varying complexity levels."""
    
    def __init__(self):
        self.labeler = PasswordLabeler()
        
        # Character sets
        self.lowercase = string.ascii_lowercase
        self.uppercase = string.ascii_uppercase
        self.digits = string.digits
        self.special_chars = "!@#$%^&*()_+-=[]{}|;:,.<>?"
        
        # Common words for weak passwords
        self.common_words = [
            'password', 'admin', 'user', 'test', 'hello', 'world',
            'welcome', 'login', 'guest', 'temp', 'demo', 'user123',
            'test123', 'pass123', 'abc123', 'qwerty', '123456'
        ]
        
        # Common names and words
        self.names = [
            'john', 'jane', 'mike', 'sarah', 'david', 'lisa',
            'chris', 'emma', 'alex', 'anna', 'tom', 'mary'
        ]
        
        # Common years and numbers
        self.years = ['2023', '2022', '2021', '2020', '2019', '2018']
        self.common_numbers = ['123', '456', '789', '000', '111', '999']
    
    def generate_weak_password(self) -> str:
        """Generate a weak password."""
        patterns = [
            # Very short passwords
            lambda: ''.join(random.choices(self.lowercase, k=random.randint(3, 5))),
            
            # Common words
            lambda: random.choice(self.common_words),
            
            # Common words with numbers
            lambda: random.choice(self.common_words) + random.choice(self.common_numbers),
            
            # Names with numbers
            lambda: random.choice(self.names) + random.choice(self.years),
            
            # Sequential patterns
            lambda: ''.join(random.choices('123456789', k=random.randint(4, 6))),
            
            # Keyboard patterns
            lambda: random.choice(['qwerty', 'asdfgh', 'zxcvbn']),
            
            # Simple letter patterns
            lambda: ''.join(random.choices(self.lowercase, k=random.randint(4, 6))),
            
            # Common word with simple substitution
            lambda: random.choice(self.common_words).replace('a', '4').replace('e', '3'),
            
            # Very short with numbers
            lambda: ''.join(random.choices(self.lowercase, k=3)) + ''.join(random.choices(self.digits, k=2)),
            
            # Simple names
            lambda: random.choice(self.names),
        ]
        
        return random.choice(patterns)()
    
    def generate_medium_password(self) -> str:
        """Generate a medium strength password."""
        patterns = [
            # Mixed case with numbers
            lambda: ''.join([
                random.choice(self.uppercase),
                ''.join(random.choices(self.lowercase, k=random.randint(4, 7))),
                ''.join(random.choices(self.digits, k=random.randint(2, 4)))
            ]),
            
            # Word with special character and numbers
            lambda: random.choice(self.names).capitalize() + random.choice(self.special_chars) + random.choice(self.years),
            
            # Random mixed characters
            lambda: ''.join(random.choices(
                self.lowercase + self.uppercase + self.digits,
                k=random.randint(8, 12)
            )),
            
            # Name with special chars
            lambda: random.choice(self.names).capitalize() + random.choice(self.special_chars) + random.choice(self.common_numbers),
            
            # Mixed case word with numbers
            lambda: ''.join([
                random.choice(self.uppercase),
                ''.join(random.choices(self.lowercase, k=random.randint(3, 6))),
                random.choice(self.special_chars),
                ''.join(random.choices(self.digits, k=random.randint(2, 3)))
            ]),
            
            # Random with some special chars
            lambda: ''.join(random.choices(
                self.lowercase + self.uppercase + self.digits + self.special_chars[:5],
                k=random.randint(8, 10)
            )),
            
            # Simple mixed case with numbers
            lambda: ''.join([
                random.choice(self.uppercase),
                ''.join(random.choices(self.lowercase, k=random.randint(3, 5))),
                ''.join(random.choices(self.digits, k=random.randint(2, 3)))
            ]),
            
            # Word with numbers and special char
            lambda: random.choice(self.names).capitalize() + random.choice(self.digits) + random.choice(self.special_chars),
        ]
        
        return random.choice(patterns)()
    
    def generate_strong_password(self) -> str:
        """Generate a strong password."""
        patterns = [
            # Complex random password
            lambda: ''.join(random.choices(
                self.lowercase + self.uppercase + self.digits + self.special_chars,
                k=random.randint(12, 16)
            )),
            
            # Structured strong password
            lambda: ''.join([
                random.choice(self.uppercase),
                random.choice(self.lowercase),
                random.choice(self.digits),
                random.choice(self.special_chars),
                ''.join(random.choices(
                    self.lowercase + self.uppercase + self.digits + self.special_chars,
                    k=random.randint(8, 12)
                ))
            ]),
            
            # High entropy random
            lambda: ''.join(random.choices(
                self.lowercase + self.uppercase + self.digits + self.special_chars,
                k=random.randint(14, 18)
            )),
            
            # Complex pattern
            lambda: ''.join([
                ''.join(random.choices(self.uppercase, k=2)),
                ''.join(random.choices(self.lowercase, k=3)),
                ''.join(random.choices(self.digits, k=3)),
                ''.join(random.choices(self.special_chars, k=2)),
                ''.join(random.choices(self.uppercase + self.lowercase, k=4))
            ]),
        ]
        
        return random.choice(patterns)()
    
    def generate_password_with_target_label(self, target_label: str) -> str:
        """Generate a password targeting a specific strength label."""
        if target_label == 'weak':
            return self.generate_weak_password()
        elif target_label == 'medium':
            return self.generate_medium_password()
        elif target_label == 'strong':
            return self.generate_strong_password()
        else:
            raise ValueError(f"Invalid target label: {target_label}")
    
    def generate_balanced_dataset(self, samples_per_class: int = 1000) -> List[Dict]:
        """Generate a balanced dataset with equal samples per class."""
        dataset = []
        
        for label in ['weak', 'medium', 'strong']:
            print(f"Generating {samples_per_class} {label} passwords...")
            
            for _ in range(samples_per_class):
                password = self.generate_password_with_target_label(label)
                
                # Verify the label using our labeler
                actual_label = self.labeler.label_password(password)
                
                # If the generated password doesn't match the target label,
                # try a few more times, then accept it
                attempts = 0
                attempts = 0
                while actual_label != label and attempts < 5:
                    password = self.generate_password_with_target_label(label)
                    actual_label = self.labeler.label_password(password)
                    attempts += 1

                if actual_label != label:
                    print(f"Warning: Failed to generate {label} password after {attempts} attempts. "
                          f"Generated '{password}' was labeled as '{actual_label}'")
                
                dataset.append({
                    'password': password,
                    'target_label': label,
                    'actual_label': actual_label,
                    'label_match': actual_label == label
                })
        
        return dataset
    
    def generate_dataset(self, total_samples: int = 3000) -> pd.DataFrame:
        """Generate a complete dataset for training."""
        print(f"Generating {total_samples} passwords...")
        
        samples_per_class = total_samples // 3
        dataset = self.generate_balanced_dataset(samples_per_class)
        
        df = pd.DataFrame(dataset)
        
        # Add some statistics
        print(f"\nDataset Statistics:")
        print(f"Total samples: {len(df)}")
        print(f"Target label distribution:")
        print(df['target_label'].value_counts())
        print(f"\nActual label distribution:")
        print(df['actual_label'].value_counts())
        print(f"\nLabel match rate: {df['label_match'].mean():.2%}")
        
        return df


def main():
    """Generate password dataset and save to CSV."""
    generator = PasswordGenerator()
    
    # Generate dataset
    df = generator.generate_dataset(total_samples=3000)
    
    # Save to CSV
    # Save to CSV
    output_file = 'data/password_dataset.csv'
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    df.to_csv(output_file, index=False)
    print(f"\nDataset saved to: {output_file}")
    
    # Show some examples
    print(f"\nSample passwords from dataset:")
    print("=" * 50)
    
    for label in ['weak', 'medium', 'strong']:
        sample = df[df['target_label'] == label].iloc[0]
        print(f"{label.upper()}: {sample['password']} (actual: {sample['actual_label']})")
    
    return df


if __name__ == "__main__":
    main() 