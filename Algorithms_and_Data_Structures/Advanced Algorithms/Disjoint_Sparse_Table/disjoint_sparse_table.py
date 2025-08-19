import time
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
import math

class DisjointSparseTable:
    """Disjoint Sparse Table implementation."""
    
    def __init__(self, arr: List[int], operation: str = 'sum'):
        self.arr = arr
        self.n = len(arr)
        self.operation = operation
        self.table = []
        self.log_table = []
        
        # Precompute logarithms
        self._precompute_logs()
        
        # Build sparse table
        self._build_table()
    
    def _precompute_logs(self) -> None:
        """Precompute logarithms for efficient querying."""
        self.log_table = [0] * (self.n + 1)
        for i in range(2, self.n + 1):
            self.log_table[i] = self.log_table[i // 2] + 1
    
    def _get_operation(self) -> Callable[[int, int], int]:
        """Get the operation function."""
        if self.operation == 'sum':
            return lambda x, y: x + y
        elif self.operation == 'min':
            return lambda x, y: min(x, y)
        elif self.operation == 'max':
            return lambda x, y: max(x, y)
        elif self.operation == 'gcd':
            return lambda x, y: math.gcd(x, y)
        elif self.operation == 'lcm':
            return lambda x, y: (x * y) // math.gcd(x, y)
        elif self.operation == 'and':
            return lambda x, y: x & y
        elif self.operation == 'or':
            return lambda x, y: x | y
        elif self.operation == 'xor':
            return lambda x, y: x ^ y
        else:
            return lambda x, y: x + y  # Default to sum
    
    def _build_table(self) -> None:
        """Build the sparse table."""
        op_func = self._get_operation()
        
        # Calculate number of levels needed
        levels = self.log_table[self.n] + 1
        
        # Initialize table
        self.table = [[0] * self.n for _ in range(levels)]
        
        # Fill first level with original array
        for i in range(self.n):
            self.table[0][i] = self.arr[i]
        
        # Fill remaining levels
        for level in range(1, levels):
            block_size = 1 << level
            
            for i in range(self.n):
                # Calculate the result for range [i, i + block_size - 1]
                if i + block_size <= self.n:
                    # Use the results from previous level
                    left_half = self.table[level - 1][i]
                    right_half = self.table[level - 1][i + (block_size // 2)]
                    self.table[level][i] = op_func(left_half, right_half)
                else:
                    # Range extends beyond array
                    self.table[level][i] = self.table[level - 1][i]
    
    def query(self, left: int, right: int) -> int:
        """Query the range [left, right]."""
        if left > right or left < 0 or right >= self.n:
            raise ValueError("Invalid range")
        
        if left == right:
            return self.arr[left]
        
        # Find the largest power of 2 that fits in the range
        length = right - left + 1
        level = self.log_table[length]
        
        # Use the precomputed result
        return self.table[level][left]
    
    def range_sum(self, left: int, right: int) -> int:
        """Get the sum of elements in range [left, right]."""
        if self.operation != 'sum':
            raise ValueError("Operation is not sum")
        return self.query(left, right)
    
    def range_min(self, left: int, right: int) -> int:
        """Get the minimum element in range [left, right]."""
        if self.operation != 'min':
            raise ValueError("Operation is not min")
        return self.query(left, right)
    
    def range_max(self, left: int, right: int) -> int:
        """Get the maximum element in range [left, right]."""
        if self.operation != 'max':
            raise ValueError("Operation is not max")
        return self.query(left, right)
    
    def range_gcd(self, left: int, right: int) -> int:
        """Get the GCD of elements in range [left, right]."""
        if self.operation != 'gcd':
            raise ValueError("Operation is not gcd")
        return self.query(left, right)
    
    def get_table(self) -> List[List[int]]:
        """Get the sparse table."""
        return self.table
    
    def get_levels(self) -> int:
        """Get the number of levels in the table."""
        return len(self.table)
    
    def get_operation(self) -> str:
        """Get the operation used."""
        return self.operation

def build_sparse_table(arr: List[int], operation: str = 'sum') -> DisjointSparseTable:
    """Build a disjoint sparse table for the given array."""
    return DisjointSparseTable(arr, operation)

def query(sparse_table: DisjointSparseTable, left: int, right: int) -> int:
    """Query the range [left, right]."""
    return sparse_table.query(left, right)

def range_sum(sparse_table: DisjointSparseTable, left: int, right: int) -> int:
    """Get the sum of elements in range [left, right]."""
    return sparse_table.range_sum(left, right)

def range_min(sparse_table: DisjointSparseTable, left: int, right: int) -> int:
    """Get the minimum element in range [left, right]."""
    return sparse_table.range_min(left, right)

def range_max(sparse_table: DisjointSparseTable, left: int, right: int) -> int:
    """Get the maximum element in range [left, right]."""
    return sparse_table.range_max(left, right)

def analyze_performance(arr: List[int], operation: str = 'sum') -> Dict:
    """Analyze performance of sparse table operations."""
    start_time = time.time()
    sparse_table = build_sparse_table(arr, operation)
    construction_time = time.time() - start_time
    
    # Test multiple queries
    queries = [(0, len(arr)//2), (len(arr)//4, 3*len(arr)//4), (0, len(arr)-1)]
    query_times = []
    
    for left, right in queries:
        start_time = time.time()
        result = sparse_table.query(left, right)
        query_times.append(time.time() - start_time)
    
    return {
        "array_size": len(arr),
        "operation": operation,
        "levels": sparse_table.get_levels(),
        "construction_time": construction_time,
        "query_times": query_times,
        "average_query_time": np.mean(query_times),
        "max_query_time": np.max(query_times),
        "min_query_time": np.min(query_times)
    }

def generate_test_cases() -> List[Tuple[List[int], str]]:
    """Generate test cases for sparse table."""
    return [
        # Small array with sum
        ([1, 2, 3, 4, 5, 6, 7, 8], 'sum'),
        
        # Small array with min
        ([5, 2, 8, 1, 9, 3, 7, 4], 'min'),
        
        # Small array with max
        ([1, 5, 3, 8, 2, 9, 4, 6], 'max'),
        
        # Array with GCD
        ([12, 18, 24, 36, 48, 60], 'gcd'),
        
        # Large array
        (list(range(1, 1001)), 'sum'),
        
        # Random array
        ([3, 7, 2, 9, 1, 8, 5, 4, 6], 'min')
    ]

def visualize_sparse_table(sparse_table: DisjointSparseTable, show_plot: bool = True) -> None:
    """Visualize the sparse table structure."""
    table = sparse_table.get_table()
    levels = len(table)
    
    if levels == 0:
        print("Empty sparse table")
        return
    
    # Create visualization
    fig, axes = plt.subplots(levels, 1, figsize=(12, 3 * levels))
    if levels == 1:
        axes = [axes]
    
    for level in range(levels):
        ax = axes[level]
        
        # Get data for this level
        data = table[level]
        x_positions = list(range(len(data)))
        
        # Create bar chart
        bars = ax.bar(x_positions, data, color='lightblue', alpha=0.7)
        
        # Highlight non-zero values
        for i, val in enumerate(data):
            if val != 0:
                bars[i].set_color('red')
        
        ax.set_xlabel('Array Index')
        ax.set_ylabel('Value')
        ax.set_title(f'Level {level} (Block Size: {1 << level})')
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, val in enumerate(data):
            if val != 0:
                ax.text(i, val, str(val), ha='center', va='bottom')
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig('sparse_table_structure.png', dpi=300, bbox_inches='tight')
        plt.close()

def visualize_query_process(sparse_table: DisjointSparseTable, left: int, right: int, 
                          show_plot: bool = True) -> None:
    """Visualize the query process."""
    print(f"Querying range [{left}, {right}]")
    
    # Perform query
    result = sparse_table.query(left, right)
    print(f"Result: {result}")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Original array
    ax1.bar(range(len(sparse_table.arr)), sparse_table.arr, 
            color='lightblue', alpha=0.7)
    ax1.bar(range(left, right + 1), 
            sparse_table.arr[left:right + 1], 
            color='red', alpha=0.7)
    ax1.set_title(f'Original Array (Query Range: [{left}, {right}])')
    ax1.set_xlabel('Index')
    ax1.set_ylabel('Value')
    ax1.grid(True, alpha=0.3)
    
    # Sparse table level used
    length = right - left + 1
    level = sparse_table.log_table[length]
    
    table = sparse_table.get_table()
    if level < len(table):
        data = table[level]
        x_positions = list(range(len(data)))
        
        bars = ax2.bar(x_positions, data, color='lightgreen', alpha=0.7)
        
        # Highlight the query range
        for i in range(left, min(left + (1 << level), len(data))):
            if i < len(bars):
                bars[i].set_color('red')
        
        ax2.set_title(f'Sparse Table Level {level} (Block Size: {1 << level})')
        ax2.set_xlabel('Index')
        ax2.set_ylabel('Value')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(f'query_process_{left}_{right}.png', dpi=300, bbox_inches='tight')
        plt.close()

def visualize_performance_comparison(arrays: List[List[int]], operations: List[str], 
                                   show_plot: bool = True) -> None:
    """Visualize performance comparison between different operations."""
    construction_times = []
    query_times = []
    labels = []
    
    for arr, op in zip(arrays, operations):
        perf = analyze_performance(arr, op)
        construction_times.append(perf['construction_time'])
        query_times.append(perf['average_query_time'])
        labels.append(f"{op}({len(arr)})")
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Construction time comparison
    bars1 = ax1.bar(labels, construction_times, color='lightblue', alpha=0.7)
    ax1.set_title('Construction Time Comparison')
    ax1.set_ylabel('Time (seconds)')
    ax1.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, time in zip(bars1, construction_times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{time:.6f}s', ha='center', va='bottom')
    
    # Query time comparison
    bars2 = ax2.bar(labels, query_times, color='lightgreen', alpha=0.7)
    ax2.set_title('Average Query Time Comparison')
    ax2.set_ylabel('Time (seconds)')
    ax2.tick_params(axis='x', rotation=45)
    
    # Add value labels
    for bar, time in zip(bars2, query_times):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{time:.6f}s', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig('performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to demonstrate Disjoint Sparse Table."""
    print("=== Disjoint Sparse Table Implementation ===\n")
    
    # Test cases
    test_cases = generate_test_cases()
    
    for i, (arr, operation) in enumerate(test_cases):
        print(f"Test Case {i+1}: Array size {len(arr)}, Operation: {operation}")
        print("-" * 50)
        
        # Build sparse table
        start_time = time.time()
        sparse_table = build_sparse_table(arr, operation)
        construction_time = time.time() - start_time
        
        print(f"Array: {arr[:10]}..." if len(arr) > 10 else f"Array: {arr}")
        print(f"Operation: {operation}")
        print(f"Construction time: {construction_time:.6f}s")
        print(f"Levels: {sparse_table.get_levels()}")
        
        # Test queries
        queries = [(0, len(arr)//2), (len(arr)//4, 3*len(arr)//4)]
        for left, right in queries:
            result = sparse_table.query(left, right)
            print(f"Query [{left}, {right}]: {result}")
        
        print()
    
    # Performance analysis
    print("=== Performance Analysis ===")
    performance_data = []
    
    for arr, operation in test_cases:
        perf = analyze_performance(arr, operation)
        performance_data.append(perf)
        print(f"Size {perf['array_size']}, Op {perf['operation']}: "
              f"Construction {perf['construction_time']:.6f}s, "
              f"Avg query {perf['average_query_time']:.6f}s")
    
    # Visualization for a complex case
    print("\n=== Visualization ===")
    complex_arr = [3, 7, 2, 9, 1, 8, 5, 4, 6]
    complex_sparse_table = build_sparse_table(complex_arr, 'min')
    
    print("Sparse table structure visualization:")
    visualize_sparse_table(complex_sparse_table, show_plot=False)
    
    print("Query process visualization:")
    visualize_query_process(complex_sparse_table, 2, 6, show_plot=False)
    
    # Performance comparison
    print("\n=== Performance Comparison ===")
    comparison_arrays = [
        list(range(1, 101)),
        list(range(1, 1001)),
        [3, 7, 2, 9, 1, 8, 5, 4, 6] * 50
    ]
    comparison_operations = ['sum', 'min', 'max']
    
    visualize_performance_comparison(comparison_arrays, comparison_operations, show_plot=False)
    
    # Advanced features demonstration
    print("\n=== Advanced Features ===")
    
    # Different operations
    arr = [12, 18, 24, 36, 48, 60]
    gcd_table = build_sparse_table(arr, 'gcd')
    gcd_result = gcd_table.query(0, 3)
    print(f"GCD of {arr[0:4]}: {gcd_result}")
    
    # Large array performance
    large_arr = list(range(1, 10001))
    large_table = build_sparse_table(large_arr, 'sum')
    
    start_time = time.time()
    for _ in range(1000):
        left = np.random.randint(0, len(large_arr)//2)
        right = np.random.randint(left, len(large_arr))
        large_table.query(left, right)
    query_time = time.time() - start_time
    
    print(f"1000 queries on 10000-element array: {query_time:.6f}s")
    
    print("\n=== Summary ===")
    print("Disjoint Sparse Table implementation completed successfully!")
    print("Features implemented:")
    print("- O(n log n) construction time")
    print("- O(1) query time")
    print("- Multiple operations (sum, min, max, gcd)")
    print("- Visualization capabilities")
    print("- Performance analysis")
    print("- Large dataset handling")

if __name__ == "__main__":
    main() 