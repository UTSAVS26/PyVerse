import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

class Query:
    """Query class for Mo's algorithm"""
    
    def __init__(self, left: int, right: int, index: int):
        self.left = left
        self.right = right
        self.index = index

class MosAlgorithm:
    """Mo's Algorithm for efficient offline range queries"""
    
    def __init__(self, array: List[int]):
        self.array = array
        self.n = len(array)
        self.block_size = int(self.n ** 0.5)
        self.operation_log = []
    
    def process_queries(self, queries: List[Query]) -> List[int]:
        """Process range queries using Mo's algorithm"""
        if not queries:
            return []
        
        # Sort queries using Mo's ordering
        sorted_queries = self._sort_queries(queries)
        
        # Initialize current range
        current_left = 0
        current_right = -1
        current_sum = 0
        results = [0] * len(queries)
        
        for query in sorted_queries:
            # Expand/contract range to reach query range
            while current_left > query.left:
                current_left -= 1
                current_sum += self.array[current_left]
            
            while current_right < query.right:
                current_right += 1
                current_sum += self.array[current_right]
            
            while current_left < query.left:
                current_sum -= self.array[current_left]
                current_left += 1
            
            while current_right > query.right:
                current_sum -= self.array[current_right]
                current_right -= 1
            
            results[query.index] = current_sum
        
        # Log operation
        self.operation_log.append({
            'operation': 'process_queries',
            'array_size': self.n,
            'query_count': len(queries),
            'timestamp': time.time()
        })
        
        return results
    
    def _sort_queries(self, queries: List[Query]) -> List[Query]:
        """Sort queries using Mo's ordering"""
        def get_block(query):
            return query.left // self.block_size
        
        def compare_queries(q1, q2):
            block1 = get_block(q1)
            block2 = get_block(q2)
            
            if block1 != block2:
                return block1 - block2
            
            # Even blocks: sort by right, odd blocks: sort by -right
            if block1 % 2 == 0:
                return q1.right - q2.right
            else:
                return q2.right - q1.right
        
        return sorted(queries, key=lambda q: (get_block(q), 
                                             q.right if get_block(q) % 2 == 0 else -q.right))
    
    def range_sum_query(self, queries: List[Query]) -> List[int]:
        """Process range sum queries"""
        return self.process_queries(queries)
    
    def range_min_query(self, queries: List[Query]) -> List[int]:
        """Process range minimum queries"""
        if not queries:
            return []
        
        # Sort queries
        sorted_queries = self._sort_queries(queries)
        
        # Initialize current range
        current_left = 0
        current_right = -1
        current_min = float('inf')
        results = [0] * len(queries)
        
        for query in sorted_queries:
            # Expand/contract range to reach query range
            while current_left > query.left:
                current_left -= 1
                current_min = min(current_min, self.array[current_left])
            
            while current_right < query.right:
                current_right += 1
                current_min = min(current_min, self.array[current_right])
            
            while current_left < query.left:
                if self.array[current_left] == current_min:
                    # Recalculate min
                    current_min = min(self.array[current_left + 1:current_right + 1])
                current_left += 1
            
            while current_right > query.right:
                if self.array[current_right] == current_min:
                    # Recalculate min
                    current_min = min(self.array[current_left:current_right])
                current_right -= 1
            
            results[query.index] = current_min
        
        return results
    
    def range_max_query(self, queries: List[Query]) -> List[int]:
        """Process range maximum queries"""
        if not queries:
            return []
        
        # Sort queries
        sorted_queries = self._sort_queries(queries)
        
        # Initialize current range
        current_left = 0
        current_right = -1
        current_max = float('-inf')
        results = [0] * len(queries)
        
        for query in sorted_queries:
            # Expand/contract range to reach query range
            while current_left > query.left:
                current_left -= 1
                current_max = max(current_max, self.array[current_left])
            
            while current_right < query.right:
                current_right += 1
                current_max = max(current_max, self.array[current_right])
            
            while current_left < query.left:
                if self.array[current_left] == current_max:
                    # Recalculate max
                    current_max = max(self.array[current_left + 1:current_right + 1])
                current_left += 1
            
            while current_right > query.right:
                if self.array[current_right] == current_max:
                    # Recalculate max
                    current_max = max(self.array[current_left:current_right])
                current_right -= 1
            
            results[query.index] = current_max
        
        return results
    
    def count_frequencies(self, queries: List[Query]) -> List[Dict[int, int]]:
        """Count element frequencies in ranges"""
        if not queries:
            return []
        
        # Sort queries
        sorted_queries = self._sort_queries(queries)
        
        # Initialize current range
        current_left = 0
        current_right = -1
        frequency = defaultdict(int)
        results = [{}] * len(queries)
        
        for query in sorted_queries:
            # Expand/contract range to reach query range
            while current_left > query.left:
                current_left -= 1
                frequency[self.array[current_left]] += 1
            
            while current_right < query.right:
                current_right += 1
                frequency[self.array[current_right]] += 1
            
            while current_left < query.left:
                frequency[self.array[current_left]] -= 1
                if frequency[self.array[current_left]] == 0:
                    del frequency[self.array[current_left]]
                current_left += 1
            
            while current_right > query.right:
                frequency[self.array[current_right]] -= 1
                if frequency[self.array[current_right]] == 0:
                    del frequency[self.array[current_right]]
                current_right -= 1
            
            results[query.index] = dict(frequency)
        
        return results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the algorithm"""
        return {
            'array_size': self.n,
            'block_size': self.block_size,
            'total_operations': len(self.operation_log),
            'array': self.array
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

def build_mos_algorithm(array: List[int]) -> MosAlgorithm:
    """Build Mo's algorithm from array"""
    return MosAlgorithm(array)

def range_sum_query(array: List[int], queries: List[Tuple[int, int]]) -> List[int]:
    """Perform range sum queries on array"""
    mos = MosAlgorithm(array)
    query_objects = [Query(left, right, i) for i, (left, right) in enumerate(queries)]
    return mos.range_sum_query(query_objects)

def generate_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for Mo's Algorithm"""
    test_cases = []
    
    # Test case 1: Basic operations
    array1 = [1, 2, 3, 4, 5]
    mos1 = MosAlgorithm(array1)
    queries1 = [Query(0, 2, 0), Query(1, 3, 1), Query(0, 4, 2)]
    
    test_cases.append({
        'name': 'Basic Operations',
        'algorithm': mos1,
        'array': array1,
        'queries': queries1,
        'expected_sums': [6, 9, 15]
    })
    
    # Test case 2: Range min/max
    array2 = [3, 1, 4, 1, 5, 9, 2, 6]
    mos2 = MosAlgorithm(array2)
    queries2 = [Query(2, 5, 0), Query(1, 4, 1)]
    
    test_cases.append({
        'name': 'Range Min/Max',
        'algorithm': mos2,
        'array': array2,
        'queries': queries2,
        'expected_mins': [1, 1],
        'expected_maxs': [9, 5]
    })
    
    # Test case 3: Edge cases
    array3 = [5]
    mos3 = MosAlgorithm(array3)
    queries3 = [Query(0, 0, 0)]
    
    test_cases.append({
        'name': 'Single Element',
        'algorithm': mos3,
        'array': array3,
        'queries': queries3,
        'expected_sums': [5]
    })
    
    return test_cases

def visualize_mos_algorithm(mos: MosAlgorithm, show_plot: bool = True) -> None:
    """Visualize the Mo's algorithm process"""
    if not mos.array:
        print("Empty array")
        return
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Original array
    indices = list(range(len(mos.array)))
    values = mos.array
    
    bars = ax1.bar(indices, values, color='skyblue', alpha=0.7)
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.set_title('Original Array')
    ax1.set_xticks(indices)
    ax1.set_xticklabels(indices)
    
    # Add value labels on bars
    for i, v in enumerate(values):
        ax1.text(i, v + 0.1, str(v), ha='center', va='bottom')
    
    # Plot 2: Block structure
    block_indices = list(range(0, len(mos.array), mos.block_size))
    block_sizes = [mos.block_size] * len(block_indices)
    
    ax2.bar(block_indices, block_sizes, color='lightgreen', alpha=0.7)
    ax2.set_xlabel('Array Index')
    ax2.set_ylabel('Block Size')
    ax2.set_title('Block Structure')
    ax2.set_xticks(range(0, len(mos.array), max(1, len(mos.array)//10)))
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def visualize_performance_metrics(mos: MosAlgorithm, show_plot: bool = True) -> None:
    """Visualize performance metrics"""
    if not mos.operation_log:
        print("No operations to analyze")
        return
    
    # Extract data
    operations = [op['operation'] for op in mos.operation_log]
    timestamps = [op['timestamp'] for op in mos.operation_log]
    
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
    
    # Plot 3: Array size vs query count
    array_sizes = [op.get('array_size', 0) for op in mos.operation_log]
    query_counts = [op.get('query_count', 0) for op in mos.operation_log]
    
    ax3.plot(range(len(array_sizes)), array_sizes, 'g-', marker='s', label='Array Size')
    ax3.plot(range(len(query_counts)), query_counts, 'r-', marker='o', label='Query Count')
    ax3.set_title('Array Size vs Query Count')
    ax3.set_xlabel('Operation Index')
    ax3.set_ylabel('Count')
    ax3.legend()
    
    # Plot 4: Algorithm statistics
    stats = mos.get_statistics()
    metrics = ['Array Size', 'Block Size', 'Operations', 'Array Sum']
    values = [stats['array_size'], stats['block_size'], 
              stats['total_operations'], sum(stats['array'])]
    
    ax4.bar(metrics, values, color=['orange', 'red', 'purple', 'green'])
    ax4.set_title('Mo\'s Algorithm Statistics')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def main():
    """Main function to demonstrate Mo's Algorithm"""
    print("=== Mo's Algorithm Demo ===\n")
    
    # Create Mo's algorithm instance
    array = [1, 2, 3, 4, 5, 6, 7, 8]
    mos = MosAlgorithm(array)
    
    print("1. Basic Operations:")
    print(f"   Array: {array}")
    print(f"   Array size: {mos.n}")
    print(f"   Block size: {mos.block_size}")
    
    print("\n2. Range Sum Queries:")
    queries = [
        Query(0, 3, 0),  # Sum of [1,2,3,4]
        Query(2, 5, 1),  # Sum of [3,4,5,6]
        Query(1, 4, 2)   # Sum of [2,3,4,5]
    ]
    
    results = mos.range_sum_query(queries)
    for i, (query, result) in enumerate(zip(queries, results)):
        print(f"   Sum[{query.left}, {query.right}] = {result}")
    
    print("\n3. Range Minimum Queries:")
    min_results = mos.range_min_query(queries)
    for i, (query, result) in enumerate(zip(queries, min_results)):
        print(f"   Min[{query.left}, {query.right}] = {result}")
    
    print("\n4. Range Maximum Queries:")
    max_results = mos.range_max_query(queries)
    for i, (query, result) in enumerate(zip(queries, max_results)):
        print(f"   Max[{query.left}, {query.right}] = {result}")
    
    print("\n5. Frequency Counting:")
    freq_results = mos.count_frequencies(queries)
    for i, (query, freq) in enumerate(zip(queries, freq_results)):
        print(f"   Freq[{query.left}, {query.right}] = {freq}")
    
    print("\n6. Performance Analysis:")
    perf = mos.analyze_performance()
    print(f"   Array size: {perf.get('algorithm_statistics', {}).get('array_size', 0)}")
    print(f"   Total operations: {perf.get('total_operations', 0)}")
    print(f"   Operation counts: {perf.get('operation_counts', {})}")
    
    print("\n7. Algorithm Statistics:")
    stats = mos.get_statistics()
    print(f"   Array size: {stats['array_size']}")
    print(f"   Block size: {stats['block_size']}")
    print(f"   Total operations: {stats['total_operations']}")
    print(f"   Array sum: {sum(stats['array'])}")
    
    print("\n8. Visualization:")
    print("   Generating visualizations...")
    
    # Generate visualizations
    visualize_mos_algorithm(mos, show_plot=False)
    visualize_performance_metrics(mos, show_plot=False)
    
    print("   Visualizations created successfully!")
    
    print("\n9. Test Cases:")
    test_cases = generate_test_cases()
    print(f"   Generated {len(test_cases)} test cases")
    
    for i, test_case in enumerate(test_cases):
        print(f"   Test case {i+1}: {test_case['name']}")
    
    print("\n=== Demo Complete ===")
    
    # Show one visualization
    visualize_mos_algorithm(mos)

if __name__ == "__main__":
    main() 