import time
import random
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
import matplotlib.pyplot as plt
import networkx as nx

class SkipListNode:
    """Node in a Skip List."""
    
    def __init__(self, key: int, value: Any = None, height: int = 1):
        self.key = key
        self.value = value
        self.height = height
        self.next = [None] * height  # Pointers to next nodes at each level

class SkipList:
    """Skip List implementation."""
    
    def __init__(self, max_height: int = 16):
        self.max_height = max_height
        self.head = SkipListNode(float('-inf'), None, max_height)  # Sentinel node
        self.tail = SkipListNode(float('inf'), None, max_height)   # Sentinel node
        
        # Connect head to tail at all levels
        for i in range(max_height):
            self.head.next[i] = self.tail
        
        self.size = 0
        self.current_height = 1
    
    def _random_height(self) -> int:
        """Generate a random height for a new node."""
        height = 1
        while height < self.max_height and random.random() < 0.5:
            height += 1
        return height
    
    def _find_path(self, key: int) -> List[SkipListNode]:
        """Find the path to insert/search a key."""
        path = [None] * self.max_height
        current = self.head
        
        # Start from the top level and work down
        for level in range(self.current_height - 1, -1, -1):
            while current.next[level] and current.next[level].key < key:
                current = current.next[level]
            path[level] = current
        
        return path
    
    def insert(self, key: int, value: Any = None) -> None:
        """Insert a key-value pair into the skip list."""
        path = self._find_path(key)
        
        # Check if key already exists
        if path[0].next[0] and path[0].next[0].key == key:
            path[0].next[0].value = value
            return
        
        # Create new node with random height
        height = self._random_height()
        new_node = SkipListNode(key, value, height)
        
        # Update the current height if necessary
        if height > self.current_height:
            self.current_height = height
        
        # Insert the node at all levels up to its height
        for level in range(height):
            if path[level] is not None:
                new_node.next[level] = path[level].next[level]
                path[level].next[level] = new_node
        
        self.size += 1
    
    def search(self, key: int) -> Optional[Any]:
        """Search for a key in the skip list."""
        current = self.head
        
        # Start from the top level and work down
        for level in range(self.current_height - 1, -1, -1):
            while current.next[level] and current.next[level].key < key:
                current = current.next[level]
        
        # Check if we found the key
        if current.next[0] and current.next[0].key == key:
            return current.next[0].value
        
        return None
    
    def delete(self, key: int) -> bool:
        """Delete a key from the skip list."""
        path = self._find_path(key)
        
        # Check if key exists
        if not path[0].next[0] or path[0].next[0].key != key:
            return False
        
        # Remove the node from all levels
        node_to_delete = path[0].next[0]
        for level in range(node_to_delete.height):
            path[level].next[level] = node_to_delete.next[level]
        
        # Update current height if necessary
        while (self.current_height > 1 and 
               self.head.next[self.current_height - 1] == self.tail):
            self.current_height -= 1
        
        self.size -= 1
        return True
    
    def get_range(self, start: int, end: int) -> List[Tuple[int, Any]]:
        """Get all key-value pairs in the specified range."""
        result = []
        current = self.head
        
        # Find the starting position
        for level in range(self.current_height - 1, -1, -1):
            while current.next[level] and current.next[level].key < start:
                current = current.next[level]
        
        # Traverse from start to end
        current = current.next[0]
        while current and current.key <= end:
            if current.key >= start:
                result.append((current.key, current.value))
            current = current.next[0]
        
        return result
    
    def get_min(self) -> Optional[Tuple[int, Any]]:
        """Get the minimum key-value pair."""
        if self.size == 0:
            return None
        
        min_node = self.head.next[0]
        if min_node and min_node.key != float('inf'):
            return (min_node.key, min_node.value)
        return None
    
    def get_max(self) -> Optional[Tuple[int, Any]]:
        """Get the maximum key-value pair."""
        if self.size == 0:
            return None
        
        # Find the rightmost node
        current = self.head
        for level in range(self.current_height - 1, -1, -1):
            while current.next[level] and current.next[level].key != float('inf'):
                current = current.next[level]
        
        if current and current.key != float('-inf'):
            return (current.key, current.value)
        return None
    
    def get_all_elements(self) -> List[Tuple[int, Any]]:
        """Get all key-value pairs in sorted order."""
        result = []
        current = self.head.next[0]
        
        while current and current.key != float('inf'):
            result.append((current.key, current.value))
            current = current.next[0]
        
        return result
    
    def get_height_distribution(self) -> Dict[int, int]:
        """Get the distribution of node heights."""
        distribution = {}
        current = self.head.next[0]
        
        while current and current.key != float('inf'):
            height = current.height
            distribution[height] = distribution.get(height, 0) + 1
            current = current.next[0]
        
        return distribution

def analyze_performance(operations: List[Tuple[str, int]]) -> Dict:
    """Analyze performance of skip list operations."""
    skip_list = SkipList()
    operation_times = []
    height_distribution = {}
    
    for op_type, key in operations:
        start_time = time.time()
        
        if op_type == 'insert':
            skip_list.insert(key, f"value_{key}")
        elif op_type == 'search':
            skip_list.search(key)
        elif op_type == 'delete':
            skip_list.delete(key)
        
        operation_times.append(time.time() - start_time)
    
    # Get height distribution
    height_distribution = skip_list.get_height_distribution()
    
    return {
        "total_operations": len(operations),
        "skip_list_size": skip_list.size,
        "current_height": skip_list.current_height,
        "operation_times": operation_times,
        "average_time": np.mean(operation_times),
        "max_time": np.max(operation_times),
        "min_time": np.min(operation_times),
        "height_distribution": height_distribution
    }

def generate_test_cases() -> List[List[Tuple[str, int]]]:
    """Generate test cases for skip list."""
    return [
        # Sequential insertion
        [('insert', i) for i in range(10)] + [('search', i) for i in range(10)],
        
        # Random insertion
        [('insert', i) for i in [5, 2, 8, 1, 9, 3, 7, 4, 6, 0]] + 
        [('search', i) for i in [3, 7, 1, 9, 5]],
        
        # Mixed operations
        [('insert', i) for i in [10, 5, 15, 3, 7, 12, 18]] +
        [('search', 5), ('delete', 5), ('insert', 5), ('search', 5)],
        
        # Large dataset
        [('insert', i) for i in range(100)] + 
        [('search', i % 20) for i in range(50)],
        
        # Range queries
        [('insert', i) for i in range(20)] +
        [('range', 5, 15)] * 5
    ]

def visualize_skip_list(skip_list: SkipList, show_plot: bool = True) -> None:
    """Visualize the skip list structure."""
    if skip_list.size == 0:
        print("Empty skip list")
        return
    
    # Create a multi-level visualization
    fig, axes = plt.subplots(skip_list.current_height, 1, 
                            figsize=(12, 3 * skip_list.current_height))
    if skip_list.current_height == 1:
        axes = [axes]
    
    for level in range(skip_list.current_height):
        ax = axes[level]
        
        # Get nodes at this level
        nodes = []
        current = skip_list.head
        while current:
            if current.key != float('-inf') and current.key != float('inf'):
                nodes.append(current.key)
            current = current.next[level]
        
        # Create visualization
        if nodes:
            x_positions = list(range(len(nodes)))
            ax.bar(x_positions, [1] * len(nodes), 
                  color='lightblue', alpha=0.7)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(nodes, rotation=45)
            ax.set_ylabel(f'Level {level}')
            ax.set_title(f'Skip List Level {level}')
            ax.set_ylim(0, 1.2)
        
        # Add level indicator
        ax.text(0.02, 0.95, f'Level {level}', transform=ax.transAxes, 
               fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig('skip_list_structure.png', dpi=300, bbox_inches='tight')
        plt.close()

def visualize_search_path(skip_list: SkipList, key: int, show_plot: bool = True) -> None:
    """Visualize the search path for a key."""
    if skip_list.size == 0:
        print("Empty skip list")
        return
    
    # Find the search path
    path = skip_list._find_path(key)
    path_keys = [node.key for node in path[:skip_list.current_height] if node]
    
    # Create visualization
    fig, axes = plt.subplots(skip_list.current_height, 1, 
                            figsize=(12, 3 * skip_list.current_height))
    if skip_list.current_height == 1:
        axes = [axes]
    
    for level in range(skip_list.current_height):
        ax = axes[level]
        
        # Get all nodes at this level
        nodes = []
        current = skip_list.head
        while current:
            if current.key != float('-inf') and current.key != float('inf'):
                nodes.append(current.key)
            current = current.next[level]
        
        if nodes:
            x_positions = list(range(len(nodes)))
            colors = ['red' if key in path_keys else 'lightblue' for key in nodes]
            
            ax.bar(x_positions, [1] * len(nodes), 
                  color=colors, alpha=0.7)
            ax.set_xticks(x_positions)
            ax.set_xticklabels(nodes, rotation=45)
            ax.set_ylabel(f'Level {level}')
            ax.set_title(f'Search Path at Level {level}')
            ax.set_ylim(0, 1.2)
    
    plt.tight_layout()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig(f'search_path_{key}.png', dpi=300, bbox_inches='tight')
        plt.close()

def visualize_height_distribution(skip_list: SkipList, show_plot: bool = True) -> None:
    """Visualize the height distribution of nodes."""
    distribution = skip_list.get_height_distribution()
    
    if not distribution:
        print("No nodes to visualize")
        return
    
    heights = list(distribution.keys())
    counts = list(distribution.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(heights, counts, color='lightgreen', alpha=0.7)
    plt.xlabel('Node Height')
    plt.ylabel('Number of Nodes')
    plt.title('Height Distribution in Skip List')
    plt.grid(True, alpha=0.3)
    
    # Add theoretical distribution (geometric with p=0.5)
    theoretical_heights = list(range(1, max(heights) + 1))
    theoretical_counts = [len(skip_list.get_all_elements()) * (0.5 ** (h-1)) * 0.5 
                         for h in theoretical_heights]
    plt.plot(theoretical_heights, theoretical_counts, 'r--', 
             label='Theoretical Distribution', linewidth=2)
    
    plt.legend()
    
    if show_plot:
        plt.show()
    else:
        plt.savefig('height_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to demonstrate Skip List."""
    print("=== Skip List Implementation ===\n")
    
    # Test cases
    test_cases = generate_test_cases()
    
    for i, operations in enumerate(test_cases):
        print(f"Test Case {i+1}: {len(operations)} operations")
        print("-" * 50)
        
        # Create skip list and perform operations
        skip_list = SkipList()
        
        for operation in operations:
            if len(operation) == 2:
                op_type, key = operation
                if op_type == 'insert':
                    skip_list.insert(key, f"value_{key}")
                elif op_type == 'search':
                    result = skip_list.search(key)
                elif op_type == 'delete':
                    skip_list.delete(key)
            elif len(operation) == 3:
                op_type, start, end = operation
                if op_type == 'range':
                    # Simulate range query
                    elements = skip_list.get_range(start, end)
        
        print(f"Final size: {skip_list.size}")
        print(f"Current height: {skip_list.current_height}")
        print(f"All elements: {skip_list.get_all_elements()[:10]}...")
        print()
    
    # Performance analysis
    print("=== Performance Analysis ===")
    performance_data = []
    
    for operations in test_cases:
        perf = analyze_performance(operations)
        performance_data.append(perf)
        print(f"Operations: {perf['total_operations']}, "
              f"Size: {perf['skip_list_size']}, "
              f"Height: {perf['current_height']}, "
              f"Avg time: {perf['average_time']:.6f}s")
    
    # Visualization for a complex case
    print("\n=== Visualization ===")
    complex_skip_list = SkipList()
    
    # Insert some elements
    for key in [10, 5, 15, 3, 7, 12, 18, 2, 8, 13, 20, 1, 6, 11, 16]:
        complex_skip_list.insert(key, f"value_{key}")
    
    print("Skip list structure visualization:")
    visualize_skip_list(complex_skip_list, show_plot=False)
    
    print("Search path visualization:")
    visualize_search_path(complex_skip_list, 7, show_plot=False)
    
    print("Height distribution visualization:")
    visualize_height_distribution(complex_skip_list, show_plot=False)
    
    # Advanced features demonstration
    print("\n=== Advanced Features ===")
    
    # Range query
    range_elements = complex_skip_list.get_range(5, 15)
    print(f"Range query [5, 15]: {range_elements}")
    
    # Min/Max
    min_element = complex_skip_list.get_min()
    max_element = complex_skip_list.get_max()
    print(f"Min element: {min_element}")
    print(f"Max element: {max_element}")
    
    # Height distribution
    height_dist = complex_skip_list.get_height_distribution()
    print(f"Height distribution: {height_dist}")
    
    # Performance comparison
    print("\n=== Performance Comparison ===")
    elements = list(range(1000))
    random.shuffle(elements)
    
    # Test insertion performance
    start_time = time.time()
    for key in elements[:100]:
        complex_skip_list.insert(key, f"value_{key}")
    insert_time = time.time() - start_time
    
    # Test search performance
    start_time = time.time()
    for key in elements[:100]:
        complex_skip_list.search(key)
    search_time = time.time() - start_time
    
    print(f"Insertion time for 100 elements: {insert_time:.6f}s")
    print(f"Search time for 100 elements: {search_time:.6f}s")
    
    print("\n=== Summary ===")
    print("Skip List implementation completed successfully!")
    print("Features implemented:")
    print("- Probabilistic structure")
    print("- Insert, search, delete operations")
    print("- Range queries")
    print("- Height distribution analysis")
    print("- Visualization capabilities")
    print("- Performance analysis")

if __name__ == "__main__":
    main() 