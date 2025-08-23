import time
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque

class BTreeNode:
    """Node for B-Tree"""
    
    def __init__(self, leaf: bool = True):
        self.leaf = leaf
        self.keys = []
        self.children = []
        self.n = 0  # Number of keys
    
    def is_full(self, t: int) -> bool:
        """Check if node is full (has 2t-1 keys)"""
        return len(self.keys) == 2 * t - 1
    
    def is_minimal(self, t: int) -> bool:
        """Check if node has minimum keys (t-1 keys)"""
        return len(self.keys) == t - 1

class BTree:
    """B-Tree data structure for efficient disk-based operations"""
    
    def __init__(self, t: int):
        self.root = BTreeNode(True)
        self.t = t  # Minimum degree
        self.size = 0
        self.operation_log = []
    
    def search(self, key: int) -> Optional[BTreeNode]:
        """Search for key in B-tree"""
        return self._search_recursive(self.root, key)
    
    def _search_recursive(self, node: BTreeNode, key: int) -> Optional[BTreeNode]:
        """Recursively search for key"""
        i = 0
        while i < node.n and key > node.keys[i]:
            i += 1
        
        if i < node.n and key == node.keys[i]:
            return node
        
        if node.leaf:
            return None
        
        return self._search_recursive(node.children[i], key)
    
    def insert(self, key: int) -> None:
        """Insert key into B-tree"""
        root = self.root
        
        # If root is full, split it
        if root.is_full(self.t):
            new_root = BTreeNode(False)
            new_root.children.append(root)
            self._split_child(new_root, 0)
            self.root = new_root
        
        self._insert_non_full(self.root, key)
        self.size += 1
        
        # Log operation
        self.operation_log.append({
            'operation': 'insert',
            'key': key,
            'timestamp': time.time()
        })
    
    def _insert_non_full(self, node: BTreeNode, key: int) -> None:
        """Insert key into non-full node"""
        i = node.n - 1
        
        if node.leaf:
            # Insert into leaf node
            if node.n == 0:
                node.keys.append(key)
            else:
                while i >= 0 and key < node.keys[i]:
                    node.keys.insert(i + 1, node.keys[i])
                    i -= 1
                node.keys.insert(i + 1, key)
            node.n += 1
        else:
            # Find child to insert into
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1
            
            # If child is full, split it
            if node.children[i].is_full(self.t):
                self._split_child(node, i)
                # Recalculate the correct child index after splitting
                i = 0
                while i < node.n and key > node.keys[i]:
                    i += 1
            
            self._insert_non_full(node.children[i], key)
    
    def _split_child(self, parent: BTreeNode, child_index: int) -> None:
        """Split a full child node"""
        t = self.t
        child = parent.children[child_index]
        new_node = BTreeNode(child.leaf)
        
        # Move keys to new node
        new_node.keys = child.keys[t:]
        child.keys = child.keys[:t-1]
        
        # Move children if not leaf
        if not child.leaf:
            new_node.children = child.children[t:]
            child.children = child.children[:t]
        
        # Update key counts
        new_node.n = len(new_node.keys)
        child.n = len(child.keys)
        
        # Insert new key into parent
        parent.keys.insert(child_index, child.keys.pop())
        parent.children.insert(child_index + 1, new_node)
        parent.n += 1
    
    def delete(self, key: int) -> bool:
        """Delete key from B-tree"""
        if not self.search(key):
            return False
        
        self._delete_recursive(self.root, key)
        self.size -= 1
        
        # Log operation
        self.operation_log.append({
            'operation': 'delete',
            'key': key,
            'timestamp': time.time()
        })
        
        return True
    
    def _delete_recursive(self, node: BTreeNode, key: int) -> None:
        """Recursively delete key from B-tree"""
        i = 0
        while i < node.n and key > node.keys[i]:
            i += 1
        
        if node.leaf:
            # Key is in leaf node
            if i < node.n and key == node.keys[i]:
                node.keys.pop(i)
                node.n -= 1
        else:
            # Key is in internal node
            if i < node.n and key == node.keys[i]:
                # Key is in this node
                self._delete_internal_node(node, key, i)
            else:
                # Key is in subtree
                self._delete_from_subtree(node, key, i)
    
    def _delete_internal_node(self, node: BTreeNode, key: int, index: int) -> None:
        """Delete key from internal node"""
        if node.children[index].n >= self.t:
            # Predecessor has enough keys
            pred = self._get_predecessor(node.children[index])
            node.keys[index] = pred
            self._delete_recursive(node.children[index], pred)
        elif node.children[index + 1].n >= self.t:
            # Successor has enough keys
            succ = self._get_successor(node.children[index + 1])
            node.keys[index] = succ
            self._delete_recursive(node.children[index + 1], succ)
        else:
            # Both children have minimum keys
            self._merge_children(node, index)
            self._delete_recursive(node.children[index], key)
    
    def _delete_from_subtree(self, node: BTreeNode, key: int, index: int) -> None:
        """Delete key from subtree"""
        if node.children[index].n == self.t - 1:
            # Child has minimum keys, need to borrow or merge
            self._ensure_minimum_keys(node, index)
        
        # Determine which child to recurse into
        if index > 0 and node.children[index - 1].n > self.t - 1:
            self._delete_recursive(node.children[index - 1], key)
        else:
            self._delete_recursive(node.children[index], key)
    
    def _get_predecessor(self, node: BTreeNode) -> int:
        """Get predecessor of node"""
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1]
    
    def _get_successor(self, node: BTreeNode) -> int:
        """Get successor of node"""
        while not node.leaf:
            node = node.children[0]
        return node.keys[0]
    
    def _merge_children(self, parent: BTreeNode, index: int) -> None:
        """Merge two children of parent"""
        left_child = parent.children[index]
        right_child = parent.children[index + 1]
        
        # Move key from parent to left child
        left_child.keys.append(parent.keys[index])
        left_child.keys.extend(right_child.keys)
        
        if not left_child.leaf:
            left_child.children.extend(right_child.children)
        
        left_child.n = len(left_child.keys)
        
        # Remove key and right child from parent
        parent.keys.pop(index)
        parent.children.pop(index + 1)
        parent.n -= 1
    
    def _ensure_minimum_keys(self, parent: BTreeNode, index: int) -> None:
        """Ensure child has minimum keys by borrowing or merging"""
        child = parent.children[index]
        
        # Try to borrow from left sibling
        if index > 0 and parent.children[index - 1].n > self.t - 1:
            self._borrow_from_left(parent, index)
        # Try to borrow from right sibling
        elif index < parent.n and parent.children[index + 1].n > self.t - 1:
            self._borrow_from_right(parent, index)
        # Merge with sibling
        else:
            if index > 0:
                self._merge_children(parent, index - 1)
            else:
                self._merge_children(parent, index)
    
    def _borrow_from_left(self, parent: BTreeNode, index: int) -> None:
        """Borrow key from left sibling"""
        child = parent.children[index]
        left_sibling = parent.children[index - 1]
        
        # Move key from parent to child
        child.keys.insert(0, parent.keys[index - 1])
        child.n += 1
        
        # Move key from left sibling to parent
        parent.keys[index - 1] = left_sibling.keys.pop()
        left_sibling.n -= 1
        
        # Move child if not leaf
        if not child.leaf:
            child.children.insert(0, left_sibling.children.pop())
    
    def _borrow_from_right(self, parent: BTreeNode, index: int) -> None:
        """Borrow key from right sibling"""
        child = parent.children[index]
        right_sibling = parent.children[index + 1]
        
        # Move key from parent to child
        child.keys.append(parent.keys[index])
        child.n += 1
        
        # Move key from right sibling to parent
        parent.keys[index] = right_sibling.keys.pop(0)
        right_sibling.n -= 1
        
        # Move child if not leaf
        if not child.leaf:
            child.children.append(right_sibling.children.pop(0))
    
    def traverse(self) -> List[int]:
        """Inorder traversal of B-tree"""
        result = []
        self._traverse_recursive(self.root, result)
        return result
    
    def _traverse_recursive(self, node: BTreeNode, result: List[int]) -> None:
        """Recursively traverse B-tree"""
        if node is None:
            return
        
        for i in range(node.n):
            if not node.leaf:
                self._traverse_recursive(node.children[i], result)
            result.append(node.keys[i])
        
        if not node.leaf:
            self._traverse_recursive(node.children[node.n], result)
    
    def get_height(self) -> int:
        """Get height of B-tree"""
        return self._get_height_recursive(self.root)
    
    def _get_height_recursive(self, node: BTreeNode) -> int:
        """Recursively get height of B-tree"""
        if node.leaf:
            return 1
        return 1 + self._get_height_recursive(node.children[0])
    
    def get_size(self) -> int:
        """Get number of keys in B-tree"""
        return self.size
    
    def range_query(self, left: int, right: int) -> List[int]:
        """Get all keys in range [left, right]"""
        result = []
        self._range_query_recursive(self.root, left, right, result)
        return result
    
    def _range_query_recursive(self, node: BTreeNode, left: int, right: int, result: List[int]) -> None:
        """Recursively find keys in range"""
        if node is None:
            return
        
        i = 0
        while i < node.n:
            if not node.leaf:
                self._range_query_recursive(node.children[i], left, right, result)
            
            if left <= node.keys[i] <= right:
                result.append(node.keys[i])
            
            i += 1
        
        if not node.leaf:
            self._range_query_recursive(node.children[i], left, right, result)
    
    def bulk_insert(self, keys: List[int]) -> None:
        """Insert multiple keys efficiently"""
        for key in keys:
            self.insert(key)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the B-tree"""
        return {
            'size': self.size,
            'height': self.get_height(),
            'min_degree': self.t,
            'total_operations': len(self.operation_log),
            'is_balanced': self._is_balanced()
        }
    
    def _is_balanced(self) -> bool:
        """Check if B-tree is balanced"""
        return self._check_balance_recursive(self.root, 0, -1)
    
    def _check_balance_recursive(self, node: BTreeNode, depth: int, leaf_depth: int) -> bool:
        """Recursively check if B-tree is balanced"""
        if node.leaf:
            if leaf_depth == -1:
                leaf_depth = depth
            return depth == leaf_depth
        
        for child in node.children:
            if not self._check_balance_recursive(child, depth + 1, leaf_depth):
                return False
        
        return True
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze performance metrics"""
        if not self.operation_log:
            return {}
        
        operations = [op['operation'] for op in self.operation_log]
        timestamps = [op['timestamp'] for op in self.operation_log]
        
        # Calculate operation frequencies
        op_counts = {}
        for op in operations:
            op_counts[op] = op_counts.get(op, 0) + 1
        
        # Calculate time intervals
        intervals = []
        for i in range(1, len(timestamps)):
            intervals.append(timestamps[i] - timestamps[i-1])
        
        return {
            'total_operations': len(self.operation_log),
            'operation_counts': op_counts,
            'average_interval': np.mean(intervals) if intervals else 0,
            'total_time': timestamps[-1] - timestamps[0] if len(timestamps) > 1 else 0,
            'tree_statistics': self.get_statistics()
        }

def build_b_tree(keys: List[int], t: int = 3) -> BTree:
    """Build B-tree from list of keys"""
    btree = BTree(t)
    for key in keys:
        btree.insert(key)
    return btree

def generate_test_cases() -> List[Dict[str, Any]]:
    """Generate test cases for B-Tree"""
    test_cases = []
    
    # Test case 1: Basic operations
    btree1 = BTree(3)
    btree1.insert(10)
    btree1.insert(20)
    btree1.insert(5)
    btree1.insert(6)
    
    test_cases.append({
        'name': 'Basic Operations',
        'tree': btree1,
        'expected_size': 4,
        'expected_traversal': [5, 6, 10, 20]
    })
    
    # Test case 2: Edge cases
    btree2 = BTree(2)
    for i in range(10):
        btree2.insert(i)
    
    test_cases.append({
        'name': 'Minimum Degree',
        'tree': btree2,
        'expected_size': 10
    })
    
    # Test case 3: Range queries
    btree3 = BTree(4)
    keys = [1, 3, 5, 7, 9, 11, 13, 15]
    for key in keys:
        btree3.insert(key)
    
    test_cases.append({
        'name': 'Range Queries',
        'tree': btree3,
        'expected_range': (5, 11),
        'expected_result': [5, 7, 9, 11]
    })
    
    return test_cases

def visualize_b_tree(tree: BTree, show_plot: bool = True) -> None:
    """Visualize the B-tree structure"""
    if not tree.root:
        print("Empty tree")
        return
    
    # Create networkx graph
    G = nx.DiGraph()
    pos = {}
    
    def add_nodes(node: BTreeNode, x: float = 0, y: float = 0, 
                 level: int = 0, width: float = 2.0):
        if node is None:
            return
        
        # Create node label
        node_id = id(node)
        G.add_node(node_id, keys=node.keys, leaf=node.leaf)
        pos[node_id] = (x, -y)
        
        # Add edges to children
        if not node.leaf:
            child_width = width / len(node.children)
            for i, child in enumerate(node.children):
                child_x = x - width/2 + (i + 0.5) * child_width
                G.add_edge(node_id, id(child))
                add_nodes(child, child_x, y + 1, level + 1, child_width)
    
    add_nodes(tree.root)
    
    # Create visualization
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Tree structure
    nx.draw(G, pos, ax=ax1, with_labels=True, 
           node_color='lightblue', node_size=1000,
           arrows=True, arrowstyle='->', arrowsize=20)
    
    # Add node labels
    labels = {node: f"Keys: {G.nodes[node]['keys']}\nLeaf: {G.nodes[node]['leaf']}" 
              for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels, ax=ax1, font_size=8)
    
    ax1.set_title('B-Tree Structure')
    
    # Plot 2: Statistics
    stats = tree.get_statistics()
    metrics = ['Size', 'Height', 'Min Degree', 'Operations']
    values = [stats['size'], stats['height'], stats['min_degree'], stats['total_operations']]
    
    ax2.bar(metrics, values, color=['skyblue', 'lightgreen', 'orange', 'red'])
    ax2.set_title('B-Tree Statistics')
    ax2.set_ylabel('Value')
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def visualize_insertion_process(tree: BTree, key: int, show_plot: bool = True) -> None:
    """Visualize the insertion process"""
    # Create a copy for visualization
    temp_tree = BTree(tree.t)
    for k in tree.traverse():
        if k != key:
            temp_tree.insert(k)
    
    # Show before insertion
    print(f"Before inserting key={key}")
    visualize_b_tree(temp_tree, show_plot=False)
    
    # Insert and show after
    temp_tree.insert(key)
    print(f"After inserting key={key}")
    visualize_b_tree(temp_tree, show_plot=show_plot)

def visualize_performance_metrics(tree: BTree, show_plot: bool = True) -> None:
    """Visualize performance metrics"""
    if not tree.operation_log:
        print("No operations to analyze")
        return
    
    # Extract data
    operations = [op['operation'] for op in tree.operation_log]
    timestamps = [op['timestamp'] for op in tree.operation_log]
    
    # Create subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Operation frequency
    op_counts = {}
    for op in operations:
        op_counts[op] = op_counts.get(op, 0) + 1
    
    ax1.bar(op_counts.keys(), op_counts.values(), color='skyblue')
    ax1.set_title('Operation Frequency')
    ax1.set_ylabel('Count')
    
    # Plot 2: Operations over time
    ax2.plot(range(len(timestamps)), timestamps, 'b-', marker='o')
    ax2.set_title('Operations Timeline')
    ax2.set_xlabel('Operation Index')
    ax2.set_ylabel('Timestamp')
    
    # Plot 3: Tree size growth
    size_growth = []
    current_size = 0
    for op in operations:
        if op == 'insert':
            current_size += 1
        elif op == 'delete':
            current_size -= 1
        size_growth.append(current_size)
    
    ax3.plot(range(len(size_growth)), size_growth, 'g-', marker='s')
    ax3.set_title('Tree Size Growth')
    ax3.set_xlabel('Operation Index')
    ax3.set_ylabel('Size')
    
    # Plot 4: Tree statistics
    stats = tree.get_statistics()
    metrics = ['Size', 'Height', 'Operations', 'Balanced']
    values = [stats['size'], stats['height'], stats['total_operations'], 
              1 if stats['is_balanced'] else 0]
    
    ax4.bar(metrics, values, color=['orange', 'red', 'purple', 'green'])
    ax4.set_title('Tree Statistics')
    ax4.set_ylabel('Value')
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def main():
    """Main function to demonstrate B-Tree"""
    print("=== B-Tree Demo ===\n")
    
    # Create B-tree
    btree = BTree(3)
    
    print("1. Basic Operations:")
    print("   Inserting elements: 10, 20, 5, 6, 12, 30, 7, 17")
    keys = [10, 20, 5, 6, 12, 30, 7, 17]
    for key in keys:
        btree.insert(key)
    
    print(f"   B-tree size: {btree.get_size()}")
    print(f"   B-tree height: {btree.get_height()}")
    print(f"   Minimum degree: {btree.t}")
    
    print("\n2. Search Operations:")
    search_keys = [6, 15, 30]
    for key in search_keys:
        result = btree.search(key)
        print(f"   Search {key}: {'Found' if result else 'Not found'}")
    
    print("\n3. Traversal:")
    traversal = btree.traverse()
    print(f"   Inorder traversal: {traversal}")
    
    print("\n4. Range Queries:")
    ranges = [(5, 15), (10, 25), (0, 50)]
    for left, right in ranges:
        result = btree.range_query(left, right)
        print(f"   Range [{left}, {right}]: {result}")
    
    print("\n5. Deletion:")
    delete_keys = [6, 20]
    for key in delete_keys:
        success = btree.delete(key)
        print(f"   Delete {key}: {'Success' if success else 'Failed'}")
        print(f"   New size: {btree.get_size()}")
    
    print("\n6. Performance Analysis:")
    perf = btree.analyze_performance()
    print(f"   Total operations: {perf.get('total_operations', 0)}")
    print(f"   Operation counts: {perf.get('operation_counts', {})}")
    
    print("\n7. Tree Statistics:")
    stats = btree.get_statistics()
    print(f"   Size: {stats['size']}")
    print(f"   Height: {stats['height']}")
    print(f"   Minimum degree: {stats['min_degree']}")
    print(f"   Is balanced: {stats['is_balanced']}")
    
    print("\n8. Visualization:")
    print("   Generating visualizations...")
    
    # Generate visualizations
    visualize_b_tree(btree, show_plot=False)
    visualize_performance_metrics(btree, show_plot=False)
    
    print("   Visualizations created successfully!")
    
    print("\n9. Test Cases:")
    test_cases = generate_test_cases()
    print(f"   Generated {len(test_cases)} test cases")
    
    for i, test_case in enumerate(test_cases):
        print(f"   Test case {i+1}: {test_case['name']}")
    
    print("\n=== Demo Complete ===")
    
    # Show one visualization
    visualize_b_tree(btree)

if __name__ == "__main__":
    main() 