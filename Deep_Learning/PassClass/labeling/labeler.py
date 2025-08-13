"""
Password Strength Labeler

This module contains heuristic rules to automatically label password strength
as weak, medium, or strong based on various security criteria.
"""

import re
import string
from typing import Dict, List, Tuple


class PasswordLabeler:
    """Password strength classifier using heuristic rules."""
    
    def __init__(self):
        # Common weak passwords and patterns
        self.common_passwords = {
            'password', '123456', '123456789', 'qwerty', 'abc123',
            'password123', 'admin', 'letmein', 'welcome', 'monkey',
            'dragon', 'master', 'sunshine', 'princess', 'qwerty123',
            'football', 'baseball', 'superman', 'trustno1', 'hello123'
        }
        
        # Common words that make passwords weak
        self.common_words = {
            'hello', 'world', 'test', 'user', 'login', 'welcome',
            'password', 'admin', 'user123', 'test123', 'demo',
            'guest', 'temp', 'temp123', 'newuser', 'user1'
        }
        
        # Sequential patterns
        self.sequential_patterns = [
            r'12345', r'67890', r'qwert', r'asdfg', r'zxcvb',
            r'abcde', r'fghij', r'klmno', r'pqrst', r'uvwxy'
        ]
    
    def calculate_strength_score(self, password: str) -> Dict[str, int]:
        """Calculate various strength metrics for a password."""
        score = {
            'length': len(password),
            'uppercase': sum(1 for c in password if c.isupper()),
            'lowercase': sum(1 for c in password if c.islower()),
            'digits': sum(1 for c in password if c.isdigit()),
            'special': sum(1 for c in password if c in string.punctuation),
            'unique_chars': len(set(password)),
            'repeating_patterns': 0,
            'common_patterns': 0
        }
        
        # Check for repeating patterns
        for i in range(2, len(password) // 2 + 1):
            for j in range(len(password) - i + 1):
                pattern = password[j:j+i]
                if password.count(pattern) > 1:
                    score['repeating_patterns'] += 1
        
        # Check for sequential patterns
        for pattern in self.sequential_patterns:
            if re.search(pattern, password.lower()):
                score['common_patterns'] += 1
        
        return score
    
    def is_common_password(self, password: str) -> bool:
        """Check if password is in common weak password list."""
        return password.lower() in self.common_passwords
    
    def contains_common_words(self, password: str) -> bool:
        """Check if password contains common weak words."""
        password_lower = password.lower()
        # Check for whole words only, not substrings
        for word in self.common_words:
            if word in password_lower:
                # Check if it's a whole word (not part of another word)
                # Use regex for proper word boundary detection
                import re
                if re.search(r'\b' + re.escape(word) + r'\b', password_lower):
                    return True
        return False
    
    def has_sequential_patterns(self, password: str) -> bool:
        """Check for sequential keyboard patterns."""
        password_lower = password.lower()
        return any(re.search(pattern, password_lower) for pattern in self.sequential_patterns)
    
    def label_password(self, password: str) -> str:
        """
        Label password strength as 'weak', 'medium', or 'strong'.
        
        Args:
            password: The password to evaluate
            
        Returns:
            str: 'weak', 'medium', or 'strong'
        """
        if not password:
            return 'weak'
        
        score = self.calculate_strength_score(password)
        
        # Immediate weak conditions
        if (self.is_common_password(password) or 
            self.has_sequential_patterns(password) or
            score['length'] < 6):
            return 'weak'
        
        # Check for common words (more strict)
        if self.contains_common_words(password) and score['length'] < 8:
            return 'weak'
        
        # Calculate composite score
        composite_score = 0
        
        # Length contribution (0-25 points)
        if score['length'] >= 12:
            composite_score += 25
        elif score['length'] >= 8:
            composite_score += 15
        elif score['length'] >= 6:
            composite_score += 8
        
        # Character variety contribution (0-35 points)
        char_types = 0
        if score['uppercase'] > 0:
            char_types += 1
        if score['lowercase'] > 0:
            char_types += 1
        if score['digits'] > 0:
            char_types += 1
        if score['special'] > 0:
            char_types += 1
        
        composite_score += char_types * 6
        
        # Uniqueness contribution (0-20 points)
        uniqueness_ratio = score['unique_chars'] / score['length']
        composite_score += int(uniqueness_ratio * 20)
        
        # Bonus for mixed case and special characters
        if score['uppercase'] > 0 and score['lowercase'] > 0:
            composite_score += 3
        if score['special'] > 0:
            composite_score += 3
        
        # Penalty for patterns
        composite_score -= score['repeating_patterns'] * 3
        composite_score -= score['common_patterns'] * 5
        
        # Ensure score is within bounds
        composite_score = max(0, min(100, composite_score))
        
        # Classification based on composite score
        if composite_score >= 65:
            return 'strong'
        elif composite_score >= 35:
            return 'medium'
        else:
            return 'weak'
    
    def get_detailed_analysis(self, password: str) -> Dict[str, any]:
        """
        Get detailed analysis of password strength.
        
        Args:
            password: The password to analyze
            
        Returns:
            Dict containing detailed analysis
        """
        score = self.calculate_strength_score(password)
        label = self.label_password(password)
        
        analysis = {
            'password': password,
            'label': label,
            'score': score,
            'issues': [],
            'strengths': []
        }
        
        # Identify issues
        if score['length'] < 8:
            analysis['issues'].append('Too short (recommend at least 8 characters)')
        
        if score['uppercase'] == 0:
            analysis['issues'].append('No uppercase letters')
        
        if score['lowercase'] == 0:
            analysis['issues'].append('No lowercase letters')
        
        if score['digits'] == 0:
            analysis['issues'].append('No digits')
        
        if score['special'] == 0:
            analysis['issues'].append('No special characters')
        
        if self.is_common_password(password):
            analysis['issues'].append('Common weak password')
        
        if self.contains_common_words(password):
            analysis['issues'].append('Contains common words')
        
        if self.has_sequential_patterns(password):
            analysis['issues'].append('Contains sequential patterns')
        
        # Identify strengths
        if score['length'] >= 12:
            analysis['strengths'].append('Good length')
        
        if score['uppercase'] > 0 and score['lowercase'] > 0:
            analysis['strengths'].append('Mixed case')
        
        if score['digits'] > 0:
            analysis['strengths'].append('Contains digits')
        
        if score['special'] > 0:
            analysis['strengths'].append('Contains special characters')
        
        if score['unique_chars'] / score['length'] > 0.8:
            analysis['strengths'].append('High character variety')
        
        return analysis


def label_passwords_batch(passwords: List[str]) -> List[Dict[str, any]]:
    """
    Label a batch of passwords.
    
    Args:
        passwords: List of passwords to label
        
    Returns:
        List of dictionaries with password and label
    """
    labeler = PasswordLabeler()
    results = []
    
    for password in passwords:
        analysis = labeler.get_detailed_analysis(password)
        results.append({
            'password': password,
            'label': analysis['label'],
            'score': analysis['score'],
            'issues': analysis['issues'],
            'strengths': analysis['strengths']
        })
    
    return results


if __name__ == "__main__":
    # Test the labeler
    labeler = PasswordLabeler()
    
    test_passwords = [
        "abc",           # weak
        "password123",   # weak
        "Hello2023!",    # medium
        "G7^s9L!zB1m",  # strong
        "qwerty",       # weak
        "MyPass@123",   # medium
        "tR#8$!XmPq@",  # strong
        "123456",       # weak
        "Book@789",     # medium
        "K9#mN2$pL7@",  # strong
    ]
    
    print("Password Strength Analysis:")
    print("=" * 50)
    
    for password in test_passwords:
        analysis = labeler.get_detailed_analysis(password)
        print(f"\nPassword: {password}")
        print(f"Label: {analysis['label'].upper()}")
        if analysis['issues']:
            print(f"Issues: {', '.join(analysis['issues'])}")
        if analysis['strengths']:
            print(f"Strengths: {', '.join(analysis['strengths'])}")
        print("-" * 30) 