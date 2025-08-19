import time
import numpy as np
from typing import Optional, List, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
from collections import defaultdict

class FibonacciNode:
    """Node for Fibonacci Heap"""
    
    def __init__(self, key: int, value: Any = None):
        self.key = key
        self.value = value
        self.degree = 0
        self.marked = False
        self.parent = None
        self.child = None
        self.left = None
        self.right = None

class FibonacciHeap:
    """Fibonacci Heap data structure with amortized O(1) operations"""
    
    def __init__(self):
        self.min_node = None
        self.n = 0  # Number of nodes
        self.operation_log = []
    
    def insert(self, key: int, value: Any = None) -> FibonacciNode:
        """Insert new node with key and optional value"""
        node = FibonacciNode(key, value)
        
        # Add to root list
        if self.min_node is None:
            node.left = node
            node.right = node
            self.min_node = node
        else:
            # Insert to the left of min_node
            node.left = self.min_node.left
            node.right = self.min_node
            self.min_node.left.right = node
            self.min_node.left = node
            
            # Update min if necessary
            if key < self.min_node.key:
                self.min_node = node
        
        self.n += 1
        
        # Log operation
        self.operation_log.append({
            'operation': 'insert',
            'key': key,
            'timestamp': time.time()
        })
        
        return node
    
    def get_min(self) -> Optional[FibonacciNode]:
        """Get minimum node without removing it"""
        return self.min_node
    
    def extract_min(self) -> Optional[FibonacciNode]:
        """Extract and return minimum node"""
        if self.min_node is None:
            return None
        
        min_node = self.min_node
        
        # Add children to root list
        if min_node.child is not None:
            child = min_node.child
            while True:
                next_child = child.right
                child.parent = None
                child.left = self.min_node.left
                child.right = self.min_node
                self.min_node.left.right = child
                self.min_node.left = child
                child = next_child
                if child == min_node.child:
                    break
        
        # Remove min_node from root list
        if min_node.right == min_node:
            self.min_node = None
        else:
            min_node.left.right = min_node.right
            min_node.right.left = min_node.left
            self.min_node = min_node.right
            self._consolidate()
        
        self.n -= 1
        
        # Log operation
        self.operation_log.append({
            'operation': 'extract_min',
            'key': min_node.key,
            'timestamp': time.time()
        })
        
        return min_node
    
    def _consolidate(self) -> None:
        """Consolidate trees of same degree"""
        if self.min_node is None:
            return
        
        # Array to track trees by degree
        degree_array = {}
        current = self.min_node
        
        while True:
            next_node = current.right
            degree = current.degree
            
            while degree in degree_array:
                other = degree_array[degree]
                if current.key > other.key:
                    current, other = other, current
                
                self._link(other, current)
                degree_array.pop(degree)
                degree += 1
            
            degree_array[degree] = current
            current = next_node
            if current == self.min_node:
                break
        
        # Reconstruct root list and find new min
        self.min_node = None
        for node in degree_array.values():
            if self.min_node is None:
                self.min_node = node
                node.left = node
                node.right = node
            else:
                node.left = self.min_node.left
                node.right = self.min_node
                self.min_node.left.right = node
                self.min_node.left = node
                if node.key < self.min_node.key:
                    self.min_node = node
    
    def _link(self, child: FibonacciNode, parent: FibonacciNode) -> None:
        """Link child to parent"""
        # Remove child from root list
        child.left.right = child.right
        child.right.left = child.left
        
        # Make child a child of parent
        child.parent = parent
        if parent.child is None:
            parent.child = child
            child.left = child
            child.right = child
        else:
            child.left = parent.child.left
            child.right = parent.child
            parent.child.left.right = child
            parent.child.left = child
        
        parent.degree += 1
        child.marked = False
    
    def decrease_key(self, node: FibonacciNode, new_key: int) -> None:
        """Decrease key of given node"""
        if new_key > node.key:
            raise ValueError("New key is greater than current key")
        
        node.key = new_key
        parent = node.parent
        
        if parent is not None and node.key < parent.key:
            self._cut(node, parent)
            self._cascading_cut(parent)
        
        if node.key < self.min_node.key:
            self.min_node = node
        
        # Log operation
        self.operation_log.append({
            'operation': 'decrease_key',
            'old_key': node.key,
            'new_key': new_key,
            'timestamp': time.time()
        })
    
    def _cut(self, child: FibonacciNode, parent: FibonacciNode) -> None:
        """Cut child from parent and add to root list"""
        # Remove child from parent's child list
        if child.right == child:
            parent.child = None
        else:
            child.left.right = child.right
            child.right.left = child.left
            if parent.child == child:
                parent.child = child.right
        
        parent.degree -= 1
        
        # Add child to root list
        child.left = self.min_node.left
        child.right = self.min_node
        self.min_node.left.right = child
        self.min_node.left = child
        
        child.parent = None
        child.marked = False
    
    def _cascading_cut(self, node: FibonacciNode) -> None:
        """Cascading cut operation"""
        parent = node.parent
        if parent is not None:
            if not node.marked:
                node.marked = True
            else:
                self._cut(node, parent)
                self._cascading_cut(parent)
    
    def delete(self, node: FibonacciNode) -> None:
        """Delete given node"""
        self.decrease_key(node, float('-inf'))
        self.extract_min()
        
        # Log operation
        self.operation_log.append({
            'operation': 'delete',
            'key': node.key,
            'timestamp': time.time()
        })
    
    def union(self, other: 'FibonacciHeap') -> None:
        """Union this heap with another heap"""
        if other.min_node is None:
            return
        
        if self.min_node is None:
            self.min_node = other.min_node
            self.n = other.n
            return
        
        # Concatenate root lists
        self.min_node.left.right = other.min_node
        other.min_node.left.right = self.min_node
        temp = self.min_node.left
        self.min_node.left = other.min_node.left
        other.min_node.left = temp
        
        # Update min if necessary
        if other.min_node.key < self.min_node.key:
            self.min_node = other.min_node
        
        self.n += other.n
        
        # Log operation
        self.operation_log.append({
            'operation': 'union',
            'other_size': other.n,
            'timestamp': time.time()
        })
    
    def get_size(self) -> int:
        """Get number of nodes in heap"""
        return self.n
    
    def is_empty(self) -> bool:
        """Check if heap is empty"""
        return self.min_node is None
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the heap"""
        return {
            'size': self.n,
            'min_key': self.min_node.key if self.min_node else None,
            'total_operations': len(self.operation_log),
            'is_empty': self.is_empty()
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
            'heap_statistics': self.get_statistics()
        }

def build_fibonacci_heap(keys: List[int]) -> FibonacciHeap:
    """Build Fibonacci heap from list of keys"""
    heap = FibonacciHeap()
    for key in keys:
        heap.insert(key)
    return heap

def generate_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for Fibonacci Heap"""
    test_cases = []
    
    # Test case 1: Basic operations
    fh1 = FibonacciHeap()
    fh1.insert(5)
    fh1.insert(3)
    fh1.insert(7)
    
    test_cases.append({
        'name': 'Basic Operations',
        'heap': fh1,
        'expected_size': 3,
        'expected_min': 3
    })
    
    # Test case 2: Extract min
    fh2 = FibonacciHeap()
    fh2.insert(10)
    fh2.insert(5)
    fh2.insert(15)
    
    min_node = fh2.extract_min()
    
    test_cases.append({
        'name': 'Extract Min',
        'heap': fh2,
        'extracted_min': min_node.key if min_node else None,
        'expected_size': 2
    })
    
    # Test case 3: Decrease key
    fh3 = FibonacciHeap()
    node = fh3.insert(20)
    fh3.insert(10)
    fh3.decrease_key(node, 5)
    
    test_cases.append({
        'name': 'Decrease Key',
        'heap': fh3,
        'expected_min': 5
    })
    
    return test_cases

def visualize_fibonacci_heap(heap: FibonacciHeap, show_plot: bool = True) -> None:
    """Visualize the Fibonacci heap structure"""
    if heap.is_empty():
        print("Empty heap")
        return
    
    # Create networkx graph
    G = nx.DiGraph()
    pos = {}
    
    def add_nodes(node: Optional[FibonacciNode], x: float = 0, y: float = 0, 
                 level: int = 0, width: float = 2.0):
        if node is None:
            return
        
        G.add_node(id(node), key=node.key, degree=node.degree, marked=node.marked)
        pos[id(node)] = (x, -y)
        
        # Add edges to children
        if node.child:
            child = node.child
            while True:
                G.add_edge(id(node), id(child))
                add_nodes(child, x - width/2, y + 1, level + 1, width/2)
                child = child.right
                if child == node.child:
                    break
        
        # Add edges to siblings in root list
        if node.right != node:
            G.add_edge(id(node), id(node.right), edge_type='sibling')
    
    # Add all root nodes
    current = heap.min_node
    while True:
        add_nodes(current)
        current = current.right
        if current == heap.min_node:
            break
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Tree structure
    nx.draw(G, pos, ax=ax1, with_labels=True, 
           node_color='lightblue', node_size=1000,
           arrows=True, arrowstyle='->', arrowsize=20)
    
    # Add node labels
    labels = {node: f"{G.nodes[node]['key']}\n{G.nodes[node]['degree']}" 
              for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1)
    
    ax1.set_title('Fibonacci Heap Structure (Key/Degree)')
    
    # Plot 2: Statistics
    stats = heap.get_statistics()
    metrics = ['Size', 'Min Key', 'Operations', 'Empty']
    values = [stats['size'], stats['min_key'] or 0, 
              stats['total_operations'], 1 if stats['is_empty'] else 0]
    
    ax2.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'red'])
    ax2.set_title('Heap Statistics')
    ax2.set_ylabel('Value')
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def visualize_performance_metrics(heap: FibonacciHeap, show_plot: bool = True) -> None:
    """Visualize performance metrics"""
    if not heap.operation_log:
        print("No operations to analyze")
        return
    
    # Extract data
    operations = [op['operation'] for op in heap.operation_log]
    timestamps = [op['timestamp'] for op in heap.operation_log]
    
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
    
    # Plot 3: Heap size growth
    size_growth = []
    current_size = 0
    for op in operations:
        if op == 'insert':
            current_size += 1
        elif op == 'extract_min':
            current_size -= 1
        size_growth.append(current_size)
    
    ax3.plot(range(len(size_growth)), size_growth, 'g-', marker='s')
    ax3.set_title('Heap Size Growth')
    ax3.set_xlabel('Operation Index')
    ax3.set_ylabel('Size')
    
    # Plot 4: Heap statistics
    stats = heap.get_statistics()
    metrics = ['Size', 'Min Key', 'Operations', 'Empty']
    values = [stats['size'], stats['min_key'] or 0, 
              stats['total_operations'], 1 if stats['is_empty'] else 0]
    
    ax4.bar(metrics, values, color=['orange', 'red', 'purple', 'green'])
    ax4.set_title('Heap Statistics')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def main():
    """Main function to demonstrate Fibonacci Heap"""
    print("=== Fibonacci Heap Demo ===\n")
    
    # Create Fibonacci heap
    fh = FibonacciHeap()
    
    print("1. Basic Operations:")
    print("   Inserting elements: 5, 3, 7, 1, 9")
    fh.insert(5)
    fh.insert(3)
    fh.insert(7)
    fh.insert(1)
    fh.insert(9)
    
    print(f"   Heap size: {fh.get_size()}")
    print(f"   Is empty: {fh.is_empty()}")
    
    print("\n2. Get Minimum:")
    min_node = fh.get_min()
    print(f"   Minimum key: {min_node.key if min_node else 'None'}")
    
    print("\n3. Extract Minimum:")
    extracted = fh.extract_min()
    print(f"   Extracted: {extracted.key if extracted else 'None'}")
    print(f"   New size: {fh.get_size()}")
    
    print("\n4. Decrease Key:")
    node = fh.insert(20)
    fh.insert(15)
    print(f"   Before decrease: {node.key}")
    fh.decrease_key(node, 2)
    print(f"   After decrease: {node.key}")
    
    print("\n5. Union Operation:")
    fh2 = FibonacciHeap()
    fh2.insert(8)
    fh2.insert(12)
    fh.union(fh2)
    print(f"   After union size: {fh.get_size()}")
    
    print("\n6. Performance Analysis:")
    perf = fh.analyze_performance()
    print(f"   Total operations: {perf.get('total_operations', 0)}")
    print(f"   Operation counts: {perf.get('operation_counts', {})}")
    
    print("\n7. Heap Statistics:")
    stats = fh.get_statistics()
    print(f"   Size: {stats['size']}")
    print(f"   Min key: {stats['min_key']}")
    print(f"   Is empty: {stats['is_empty']}")
    print(f"   Total operations: {stats['total_operations']}")
    
    print("\n8. Visualization:")
    print("   Generating visualizations...")
    
    # Generate visualizations
    visualize_fibonacci_heap(fh, show_plot=False)
    visualize_performance_metrics(fh, show_plot=False)
    
    print("   Visualizations created successfully!")
    
    print("\n9. Test Cases:")
    test_cases = generate_test_cases()
    print(f"   Generated {len(test_cases)} test cases")
    
    for i, test_case in enumerate(test_cases):
        print(f"   Test case {i+1}: {test_case['name']}")
    
    print("\n=== Demo Complete ===")
    
    # Show one visualization
    visualize_fibonacci_heap(fh)

if __name__ == "__main__":
    main() 