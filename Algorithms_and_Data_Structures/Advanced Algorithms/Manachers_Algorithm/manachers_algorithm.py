import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

class ManachersAlgorithm:
    """Manacher's Algorithm for finding longest palindromic substring"""
    
    def __init__(self, text: str):
        self.text = text
        self.processed_text = self._preprocess(text)
        self.palindrome_radius = []
        self.operation_log = []
        self._compute_palindromes()
    
    def _preprocess(self, text: str) -> str:
        """Add special characters for even-length palindromes"""
        if not text:
            return ""
        
        processed = "#"
        for char in text:
            processed += char + "#"
        return processed
    
    def _compute_palindromes(self) -> None:
        """Compute palindrome radius for each position"""
        if not self.processed_text:
            return
        
        n = len(self.processed_text)
        self.palindrome_radius = [0] * n
        
        center = 0
        right = 0
        
        for i in range(n):
            # Use mirror property if i is within current palindrome
            if i < right:
                mirror = 2 * center - i
                self.palindrome_radius[i] = min(right - i, self.palindrome_radius[mirror])
            
            # Expand palindrome
            left = i - self.palindrome_radius[i] - 1
            right_expand = i + self.palindrome_radius[i] + 1
            
            while (left >= 0 and right_expand < n and 
                   self.processed_text[left] == self.processed_text[right_expand]):
                self.palindrome_radius[i] += 1
                left -= 1
                right_expand += 1
            
            # Update center and right boundary if needed
            if i + self.palindrome_radius[i] > right:
                center = i
                right = i + self.palindrome_radius[i]
        
        # Log operation
        self.operation_log.append({
            'operation': 'compute_palindromes',
            'text_length': len(self.text),
            'processed_length': n,
            'timestamp': time.time()
        })
    
    def get_longest_palindrome(self) -> str:
        """Get longest palindromic substring"""
        if not self.palindrome_radius:
            return ""
        
        max_radius = max(self.palindrome_radius)
        max_center = self.palindrome_radius.index(max_radius)
        
        start = (max_center - max_radius) // 2
        end = (max_center + max_radius) // 2
        
        return self.text[start:end]
    
    def get_all_palindromes(self) -> List[Tuple[int, int, str]]:
        """Get all palindromic substrings"""
        palindromes = []
        
        for i, radius in enumerate(self.palindrome_radius):
            if radius > 0:
                start = (i - radius) // 2
                end = (i + radius) // 2
                palindrome = self.text[start:end]
                palindromes.append((start, end, palindrome))
        
        return palindromes
    
    def count_palindromes(self) -> int:
    def count_palindromes(self) -> int:
        """Count total number of palindromic substrings"""
        count = 0
        for radius in self.palindrome_radius:
            count += (radius + 1) // 2
        return count
    
    def is_palindrome(self, start: int, end: int) -> bool:
        """Check if substring is palindrome"""
        if start < 0 or end > len(self.text) or start >= end:
            return False
        
        substring = self.text[start:end]
        return substring == substring[::-1]
    
    def get_palindrome_statistics(self) -> Dict[str, Any]:
        """Get statistics about palindromes"""
        if not self.palindrome_radius:
            return {}
        
        max_radius = max(self.palindrome_radius)
        total_palindromes = self.count_palindromes()
        longest_palindrome = self.get_longest_palindrome()
        
        return {
            'text_length': len(self.text),
            'max_radius': max_radius,
            'total_palindromes': total_palindromes,
            'longest_palindrome': longest_palindrome,
            'longest_palindrome_length': len(longest_palindrome)
        }
    
    def get_palindromes_of_length(self, length: int) -> List[str]:
        """Get all palindromes of specific length"""
        palindromes = []
        
        for i, radius in enumerate(self.palindrome_radius):
            if radius >= length:
                start = (i - radius) // 2
                end = (i + radius) // 2
                palindrome = self.text[start:end]
                if len(palindrome) == length and palindrome not in palindromes:
                    palindromes.append(palindrome)
        
        return palindromes
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the algorithm"""
        return {
            'text_length': len(self.text),
            'processed_length': len(self.processed_text),
            'total_operations': len(self.operation_log),
            'palindrome_radius_length': len(self.palindrome_radius)
        }
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        if not self.operation_log:
            return {}
        
        operations = [op['operation'] for op in self.operation_log]
        timestamps = [op['timestamp'] for op in self.operation_log]
        
        # Calculate operation frequencies
        op_counts = defaultdict(int)
        for op in operations:
            op_counts[op] += 1
        
        # Calculate time intervals
        intervals = []
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i] - timestamps[i-1])
        
        return {
            'total_operations': len(self.operation_log),
            'operation_counts': dict(op_counts),
            'average_interval': np.mean(intervals) if intervals else 0,
            'total_time': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'algorithm_statistics': self.get_statistics()
        }

def build_manachers_algorithm(text: str) -> ManachersAlgorithm:
    """Build Manacher's algorithm from text"""
    return ManachersAlgorithm(text)

def find_longest_palindrome(text: str) -> str:
    """Find longest palindromic substring"""
    ma = ManachersAlgorithm(text)
    return ma.get_longest_palindrome()

def generate_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for Manacher's Algorithm"""
    test_cases = []
    
    # Test case 1: Basic operations
    text1 = "babad"
    ma1 = ManachersAlgorithm(text1)
    
    test_cases.append({
        'name': 'Basic Operations',
        'algorithm': ma1,
        'text': text1,
        'expected_longest': "bab",  # or "aba"
        'expected_count': 5
    })
    
    # Test case 2: Multiple palindromes
    text2 = "racecar"
    ma2 = ManachersAlgorithm(text2)
    
    test_cases.append({
        'name': 'Multiple Palindromes',
        'algorithm': ma2,
        'text': text2,
        'expected_longest': "racecar",
        'expected_count': 7
    })
    
    # Test case 3: Edge cases
    text3 = "a"
    ma3 = ManachersAlgorithm(text3)
    
    test_cases.append({
        'name': 'Single Character',
        'algorithm': ma3,
        'text': text3,
        'expected_longest': "a",
        'expected_count': 1
    })
    
    return test_cases

def visualize_manachers_algorithm(ma: ManachersAlgorithm, show_plot: bool = True) -> None:
    """Visualize the Manacher's algorithm process"""
    if not ma.text:
        print("Empty text")
        return
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Original text with palindrome highlighting
    text = ma.text
    longest_palindrome = ma.get_longest_palindrome()
    
    # Find position of longest palindrome
    start_pos = text.find(longest_palindrome)
    end_pos = start_pos + len(longest_palindrome)
    
    # Create color array
    colors = ['lightblue'] * len(text)
    for i in range(start_pos, end_pos):
        colors[i] = 'red'
    
    bars = ax1.bar(range(len(text)), [1] * len(text), color=colors, alpha=0.7)
    ax1.set_xlabel('Position')
    ax1.set_ylabel('Character')
    ax1.set_title('Original Text with Longest Palindrome')
    ax1.set_xticks(range(len(text)))
    ax1.set_xticklabels(list(text))
    
    # Add character labels
    for i, char in enumerate(text):
        ax1.text(i, 0.5, char, ha='center', va='center', fontweight='bold')
    
    # Plot 2: Palindrome radius array
    if ma.palindrome_radius:
        indices = list(range(len(ma.palindrome_radius)))
        radii = ma.palindrome_radius
        
        ax2.bar(indices, radii, color='skyblue', alpha=0.7)
        ax2.set_xlabel('Position in Processed Text')
        ax2.set_ylabel('Palindrome Radius')
        ax2.set_title('Palindrome Radius Array')
        
        # Highlight maximum radius
        max_radius = max(radii)
        max_index = radii.index(max_radius)
        ax2.bar(max_index, max_radius, color='red', alpha=0.8)
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def visualize_performance_metrics(ma: ManachersAlgorithm, show_plot: bool = True) -> None:
    """Visualize performance metrics"""
    if not ma.operation_log:
        print("No operations to analyze")
        return
    
    # Extract data
    operations = [op['operation'] for op in ma.operation_log]
    timestamps = [op['timestamp'] for op in ma.operation_log]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Operation frequency
    op_counts = defaultdict(int)
    for op in operations:
        op_counts[op] += 1
    
    ax1.bar(op_counts.keys(), op_counts.values(), color='skyblue')
    ax1.set_title('Operation Frequency')
    ax1.set_ylabel('Count')
    
    # Plot 2: Operations over time
    ax2.plot(range(len(timestamps)), timestamps, 'b-', marker='o')
    ax2.set_title('Operations Timeline')
    ax2.set_xlabel('Operation Index')
    ax2.set_ylabel('Timestamp')
    
    # Plot 3: Text length vs processed length
    text_lengths = [op.get('text_length', 0) for op in ma.operation_log]
    processed_lengths = [op.get('processed_length', 0) for op in ma.operation_log]
    
    ax3.plot(range(len(text_lengths)), text_lengths, 'g-', marker='s', label='Text Length')
    ax3.plot(range(len(processed_lengths)), processed_lengths, 'r-', marker='o', label='Processed Length')
    ax3.set_title('Text Length vs Processed Length')
    ax3.set_xlabel('Operation Index')
    ax3.set_ylabel('Length')
    ax3.legend()
    
    # Plot 4: Palindrome statistics
    stats = ma.get_palindrome_statistics()
    metrics = ['Text Length', 'Max Radius', 'Total Palindromes', 'Longest Length']
    values = [stats.get('text_length', 0), stats.get('max_radius', 0), 
              stats.get('total_palindromes', 0), stats.get('longest_palindrome_length', 0)]
    
    ax4.bar(metrics, values, color=['orange', 'red', 'purple', 'green'])
    ax4.set_title('Palindrome Statistics')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def main():
    """Main function to demonstrate Manacher's Algorithm"""
    print("=== Manacher's Algorithm Demo ===\n")
    
    # Create Manacher's algorithm instance
    text = "babad"
    ma = ManachersAlgorithm(text)
    
    print("1. Basic Operations:")
    print(f"   Text: {text}")
    print(f"   Text length: {len(text)}")
    print(f"   Processed text: {ma.processed_text}")
    
    print("\n2. Longest Palindrome:")
    longest = ma.get_longest_palindrome()
    print(f"   Longest palindrome: '{longest}'")
    
    print("\n3. All Palindromes:")
    all_palindromes = ma.get_all_palindromes()
    print(f"   All palindromes: {[p[2] for p in all_palindromes]}")
    
    print("\n4. Palindrome Count:")
    count = ma.count_palindromes()
    print(f"   Total palindromes: {count}")
    
    print("\n5. Palindrome Statistics:")
    stats = ma.get_palindrome_statistics()
    print(f"   Text length: {stats['text_length']}")
    print(f"   Max radius: {stats['max_radius']}")
    print(f"   Total palindromes: {stats['total_palindromes']}")
    print(f"   Longest palindrome: '{stats['longest_palindrome']}'")
    print(f"   Longest palindrome length: {stats['longest_palindrome_length']}")
    
    print("\n6. Performance Analysis:")
    perf = ma.analyze_performance()
    print(f"   Text length: {perf.get('algorithm_statistics', {}).get('text_length', 0)}")
    print(f"   Total operations: {perf.get('total_operations', 0)}")
    print(f"   Operation counts: {perf.get('operation_counts', {})}")
    
    print("\n7. Algorithm Statistics:")
    alg_stats = ma.get_statistics()
    print(f"   Text length: {alg_stats['text_length']}")
    print(f"   Processed length: {alg_stats['processed_length']}")
    print(f"   Palindrome radius length: {alg_stats['palindrome_radius_length']}")
    print(f"   Total operations: {alg_stats['total_operations']}")
    
    print("\n8. Visualization:")
    print("   Generating visualizations...")
    
    # Generate visualizations
    visualize_manachers_algorithm(ma, show_plot=False)
    visualize_performance_metrics(ma, show_plot=False)
    
    print("   Visualizations created successfully!")
    
    print("\n9. Test Cases:")
    test_cases = generate_test_cases()
    print(f"   Generated {len(test_cases)} test cases")
    
    for i, test_case in enumerate(test_cases):
        print(f"   Test case {i+1}: {test_case['name']}")
    
    print("\n=== Demo Complete ===")
    
    # Show one visualization
    visualize_manachers_algorithm(ma)

if __name__ == "__main__":
    main() 