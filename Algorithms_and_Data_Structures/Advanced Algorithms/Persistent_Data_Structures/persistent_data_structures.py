import time
import numpy as np
from typing import Dict, List, Set, Tuple, Optional, Any
import matplotlib.pyplot as plt
import networkx as nx
from copy import deepcopy
from collections import defaultdict
import random

class PersistentNode:
    """Node for persistent binary search tree"""
    def __init__(self, value: int, left: Optional['PersistentNode'] = None, 
                 right: Optional['PersistentNode'] = None, version: int = 0):
        self.value = value
        self.left = left
        self.right = right
        self.version = version
        self.size = 1
        if left:
            self.size += left.size
        if right:
            self.size += right.size

class PersistentTree:
    """Persistent Binary Search Tree with version management"""
    def __init__(self):
        self.versions = {}  # version_id -> root
        self.version_history = {}  # version_id -> parent_version
        self.current_version = "v0"
        self.next_version_id = 1
        self.versions["v0"] = None
        self.version_history["v0"] = None
        self.operation_log = []  # Track operations for analysis

    def _create_node(self, value: int, left: Optional[PersistentNode] = None, 
                    right: Optional[PersistentNode] = None, version: int = 0) -> PersistentNode:
        """Create a new node with proper size calculation"""
        node = PersistentNode(value, left, right, version)
        node.size = 1
        if left:
            node.size += left.size
        if right:
            node.size += right.size
        return node

    def _copy_path(self, node: Optional[PersistentNode], target_version: str) -> Optional[PersistentNode]:
        """Copy the path from root to target node for new version"""
        if node is None:
            return None
        
        # Create new node with same value but new version
        new_node = self._create_node(node.value, node.left, node.right, 
                                   self.next_version_id)
        return new_node

    def _insert_recursive(self, node: Optional[PersistentNode], value: int, 
                         version: str) -> PersistentNode:
        """Recursively insert value into tree, creating new nodes as needed"""
        if node is None:
            return self._create_node(value, version=self.next_version_id)
        
        # Create new node for this path
        new_node = self._create_node(node.value, node.left, node.right, 
                                   self.next_version_id)
        
        if value < node.value:
            new_node.left = self._insert_recursive(node.left, value, version)
        elif value > node.value:
            new_node.right = self._insert_recursive(node.right, value, version)
        else:
            # Value already exists, return current node
            return node
        
        # Update size
        new_node.size = 1
        if new_node.left:
            new_node.size += new_node.left.size
        if new_node.right:
            new_node.size += new_node.right.size
        
        return new_node

    def insert(self, value: int, version: Optional[str] = None) -> str:
        """Insert value into specified version, return new version id"""
        if version is None:
            version = self.current_version
        
        if version not in self.versions:
            raise ValueError(f"Version {version} does not exist")
        
        # Create new version
        new_version = f"v{self.next_version_id}"
        self.next_version_id += 1
        
        # Copy the tree and insert
        root = self.versions[version]
        new_root = self._insert_recursive(root, value, version)
        
        # Store new version
        self.versions[new_version] = new_root
        self.version_history[new_version] = version
        self.current_version = new_version
        
        # Log operation
        self.operation_log.append({
            'operation': 'insert',
            'value': value,
            'from_version': version,
            'to_version': new_version,
            'timestamp': time.time()
        })
        
        return new_version

    def _search_recursive(self, node: Optional[PersistentNode], value: int) -> Optional[PersistentNode]:
        """Recursively search for value in tree"""
        if node is None:
            return None
        
        if value == node.value:
            return node
        elif value < node.value:
            return self._search_recursive(node.left, value)
        else:
            return self._search_recursive(node.right, value)

    def search(self, value: int, version: Optional[str] = None) -> Optional[PersistentNode]:
        """Search for value in specified version"""
        if version is None:
            version = self.current_version
        
        if version not in self.versions:
            raise ValueError(f"Version {version} does not exist")
        
        root = self.versions[version]
        return self._search_recursive(root, value)

    def _find_min(self, node: PersistentNode) -> PersistentNode:
        """Find minimum value in subtree"""
        while node.left:
            node = node.left
        return node

    def _delete_recursive(self, node: Optional[PersistentNode], value: int) -> Optional[PersistentNode]:
        """Recursively delete value from tree, creating new nodes as needed"""
        if node is None:
            return None
        
        # Create new node for this path
        new_node = self._create_node(node.value, node.left, node.right, 
                                   self.next_version_id)
        
        if value < node.value:
            new_node.left = self._delete_recursive(node.left, value)
        elif value > node.value:
            new_node.right = self._delete_recursive(node.right, value)
        else:
            # Node to delete found
            if node.left is None:
                return node.right
            elif node.right is None:
                return node.left
            else:
                # Node has two children
                min_node = self._find_min(node.right)
                new_node.value = min_node.value
                new_node.right = self._delete_recursive(node.right, min_node.value)
        
        # Update size
        new_node.size = 1
        if new_node.left:
            new_node.size += new_node.left.size
        if new_node.right:
            new_node.size += new_node.right.size
        
        return new_node

    def delete(self, value: int, version: Optional[str] = None) -> str:
        """Delete value from specified version, return new version id"""
        if version is None:
            version = self.current_version
        
        if version not in self.versions:
            raise ValueError(f"Version {version} does not exist")
        
        # Create new version
        new_version = f"v{self.next_version_id}"
        self.next_version_id += 1
        
        # Copy the tree and delete
        root = self.versions[version]
        new_root = self._delete_recursive(root, value)
        
        # Store new version
        self.versions[new_version] = new_root
        self.version_history[new_version] = version
        self.current_version = new_version
        
        # Log operation
        self.operation_log.append({
            'operation': 'delete',
            'value': value,
            'from_version': version,
            'to_version': new_version,
            'timestamp': time.time()
        })
        
        return new_version

    def create_version(self, version_id: str, base_version: Optional[str] = None) -> str:
        """Create a new version based on existing version"""
        if base_version is None:
            base_version = self.current_version

        if base_version not in self.versions:
            raise ValueError(f"Base version {base_version} does not exist")
        if version_id in self.versions:
            raise ValueError(f"Version {version_id} already exists")

        # Copy the root from base version
        root = self.versions[base_version]
        new_root = root  # no-op copy; full structural sharing

        # Store new version
        self.versions[version_id] = new_root
        self.version_history[version_id] = base_version
        self.current_version = version_id

        # Log operation
        self.operation_log.append({
            'operation': 'create_version',
            'version_id': version_id,
            'base_version': base_version,
            'timestamp': time.time()
        })

        return version_id
    def get_all_versions(self) -> List[str]:
        """Get all available versions"""
        return list(self.versions.keys())

    def get_version_tree(self) -> Dict[str, List[str]]:
        """Get version hierarchy as a tree"""
        tree = defaultdict(list)
        for version, parent in self.version_history.items():
            if parent:
                tree[parent].append(version)
        return dict(tree)

    def compare_versions(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two versions and return differences"""
        if version1 not in self.versions or version2 not in self.versions:
            raise ValueError("One or both versions do not exist")
        
        def get_all_values(node: Optional[PersistentNode]) -> Set[int]:
            """Get all values in tree"""
            if node is None:
                return set()
            return {node.value} | get_all_values(node.left) | get_all_values(node.right)
        
        values1 = get_all_values(self.versions[version1])
        values2 = get_all_values(self.versions[version2])
        
        return {
            'only_in_v1': values1 - values2,
            'only_in_v2': values2 - values1,
            'common': values1 & values2,
            'total_v1': len(values1),
            'total_v2': len(values2)
        }

    def get_version_info(self, version: str) -> Dict[str, Any]:
        """Get detailed information about a version"""
        if version not in self.versions:
            raise ValueError(f"Version {version} does not exist")
        
        root = self.versions[version]
        
        def count_nodes(node: Optional[PersistentNode]) -> int:
            if node is None:
                return 0
            return 1 + count_nodes(node.left) + count_nodes(node.right)
        
        def get_height(node: Optional[PersistentNode]) -> int:
            if node is None:
                return 0
            return 1 + max(get_height(node.left), get_height(node.right))
        
        return {
            'node_count': count_nodes(root),
            'height': get_height(root),
            'parent_version': self.version_history[version],
            'has_root': root is not None
        }

    def get_memory_usage(self) -> Dict[str, int]:
        """Estimate memory usage for all versions"""
        total_nodes = 0
        shared_nodes = 0
        
        # Count total nodes across all versions
        all_nodes = set()
        for version, root in self.versions.items():
            if root:
                nodes = self._get_all_nodes(root)
                total_nodes += len(nodes)
                all_nodes.update(nodes)
        
        shared_nodes = len(all_nodes)
        
        return {
            'total_nodes': total_nodes,
            'unique_nodes': shared_nodes,
            'sharing_efficiency': (total_nodes - shared_nodes) / max(total_nodes, 1),
            'versions_count': len(self.versions)
        }

    def _get_all_nodes(self, node: Optional[PersistentNode]) -> Set[int]:
        """Get all node IDs in tree (for memory analysis)"""
        if node is None:
            return set()
        return {id(node)} | self._get_all_nodes(node.left) | self._get_all_nodes(node.right)

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
            'memory_usage': self.get_memory_usage(),
            'version_count': len(self.versions)
        }

    def generate_test_cases(self) -> List[Dict[str, Any]]:
        """Generate test cases for the persistent tree"""
        test_cases = []
        
        # Test case 1: Basic operations
        tree = PersistentTree()
        tree.insert(5)
        tree.insert(3)
        tree.insert(7)
-        tree.create_version("v1")
        tree.create_version("branch1")
        tree.insert(2, version="branch1")
        
        test_cases.append({
            'name': 'Basic Operations',
            'tree': tree,
            'expected_versions': ['v0', 'v1', 'v2', 'v3', 'v4'],
            'expected_search': {
-                'v0': [5, 3, 7],
                'v3': [5, 3, 7],     # after three inserts
                'v4': [5, 3, 7, 2]   # version created from 'branch1' insert
            }
        })
        
        # Test case 2: Deletion
        tree2 = PersistentTree()
        tree2.insert(1)
        tree2.insert(2)
        tree2.insert(3)
        tree2.delete(2)
        
        test_cases.append({
            'name': 'Deletion Test',
            'tree': tree2,
            'expected_versions': ['v0', 'v1', 'v2', 'v3'],
            'expected_search': {
                'v2': [1, 2, 3],
                'v3': [1, 3]
            }
        })
        
        return test_cases
def visualize_persistent_tree(tree: PersistentTree, show_plot: bool = True) -> None:
    """Visualize the persistent tree with all versions"""
    if not tree.versions:
        print("No versions to visualize")
        return
    
    # Create subplots for each version
    n_versions = len(tree.versions)
    cols = min(3, n_versions)
    rows = (n_versions + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    for i, (version, root) in enumerate(tree.versions.items()):
        if i >= len(axes):
            break
        
        ax = axes[i]
        ax.set_title(f'Version: {version}')
        
        if root is None:
            ax.text(0.5, 0.5, 'Empty Tree', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=12)
            ax.set_xlim(-1, 1)
            ax.set_ylim(-1, 1)
        else:
            # Create networkx graph for visualization
            G = nx.DiGraph()
            pos = {}
            
            def add_nodes(node: PersistentNode, x: float = 0, y: float = 0, 
                         level: int = 0, width: float = 2.0):
                if node is None:
                    return
                
                G.add_node(id(node), value=node.value, version=node.version)
                pos[id(node)] = (x, -y)
                
                if node.left:
                    G.add_edge(id(node), id(node.left))
                    add_nodes(node.left, x - width/2, y + 1, level + 1, width/2)
                
                if node.right:
                    G.add_edge(id(node), id(node.right))
                    add_nodes(node.right, x + width/2, y + 1, level + 1, width/2)
            
            add_nodes(root)
            
            # Draw the tree
            nx.draw(G, pos, ax=ax, with_labels=True, 
                   node_color='lightblue', node_size=1000, 
                   arrows=True, arrowstyle='->', arrowsize=20)
            
            # Add node labels
            labels = {node: G.nodes[node]['value'] for node in G.nodes()}
            nx.draw_networkx_labels(G, pos, labels, ax=ax)
    
    # Hide unused subplots
    for i in range(n_versions, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def visualize_version_graph(tree: PersistentTree, show_plot: bool = True) -> None:
    """Visualize the version hierarchy as a directed graph"""
    G = nx.DiGraph()
    
    # Add nodes and edges
    for version, parent in tree.version_history.items():
        G.add_node(version)
        if parent:
            G.add_edge(parent, version)
    
    # Create layout
    pos = nx.spring_layout(G, k=3, iterations=50)
    
    # Draw the graph
    plt.figure(figsize=(12, 8))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen', 
           node_size=2000, arrows=True, arrowstyle='->', arrowsize=20,
           font_size=10, font_weight='bold')
    
    plt.title('Version Hierarchy Graph')
    plt.tight_layout()
    
    if show_plot:
        plt.show()

def visualize_performance_metrics(tree: PersistentTree, show_plot: bool = True) -> None:
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
    
    # Plot 3: Version growth
    version_counts = []
    for i in range(len(timestamps)):
        # Count versions up to this point
        count = len([op for op in tree.operation_log[:i+1] 
                    if op['operation'] in ['insert', 'delete', 'create_version']])
        version_counts.append(count)
    
    ax3.plot(range(len(version_counts)), version_counts, 'g-', marker='s')
    ax3.set_title('Version Growth Over Time')
    ax3.set_xlabel('Operation Index')
    ax3.set_ylabel('Version Count')
    
    # Plot 4: Memory usage
    memory_info = tree.get_memory_usage()
    metrics = ['Total Nodes', 'Unique Nodes', 'Sharing Efficiency']
    values = [memory_info['total_nodes'], memory_info['unique_nodes'], 
              memory_info['sharing_efficiency']]
    
    ax4.bar(metrics, values, color=['orange', 'red', 'purple'])
    ax4.set_title('Memory Usage Metrics')
    ax4.set_ylabel('Count/Ratio')
    
    plt.tight_layout()
    if show_plot:
        plt.show()

def main():
    """Main function to demonstrate persistent data structures"""
    print("=== Persistent Data Structures Demo ===\n")
    
    # Create persistent tree
    tree = PersistentTree()
    
    print("1. Basic Operations:")
    print("   Inserting elements: 5, 3, 7")
    tree.insert(5)
    tree.insert(3)
    tree.insert(7)
    
    print(f"   Current version: {tree.current_version}")
    print(f"   Available versions: {tree.get_all_versions()}")
    
    print("\n2. Version Creation:")
    print("   Creating version 'v1'")
    tree.create_version("v1")
    
    print("   Inserting 2 in version 'v1'")
    tree.insert(2, version="v1")
    
    print("   Searching in different versions:")
    result1 = tree.search(2)  # Should not find in current version
    result2 = tree.search(2, version="v1")  # Should find in v1
    
    print(f"   Search 2 in current version: {'Found' if result1 else 'Not found'}")
    print(f"   Search 2 in v1: {'Found' if result2 else 'Not found'}")
    
    print("\n3. Deletion:")
    print("   Deleting 3 from current version")
    tree.delete(3)
    
    result3 = tree.search(3)
    print(f"   Search 3 after deletion: {'Found' if result3 else 'Not found'}")
    
    print("\n4. Version Comparison:")
    diff = tree.compare_versions("v0", "v1")
    print(f"   Differences between v0 and v1: {diff}")
    
    print("\n5. Performance Analysis:")
    perf = tree.analyze_performance()
    print(f"   Total operations: {perf.get('total_operations', 0)}")
    print(f"   Operation counts: {perf.get('operation_counts', {})}")
    
    print("\n6. Memory Usage:")
    memory = tree.get_memory_usage()
    print(f"   Total nodes: {memory['total_nodes']}")
    print(f"   Unique nodes: {memory['unique_nodes']}")
    print(f"   Sharing efficiency: {memory['sharing_efficiency']:.2f}")
    
    print("\n7. Visualization:")
    print("   Generating visualizations...")
    
    # Generate visualizations
    visualize_persistent_tree(tree, show_plot=False)
    visualize_version_graph(tree, show_plot=False)
    visualize_performance_metrics(tree, show_plot=False)
    
    print("   Visualizations created successfully!")
    
    print("\n8. Test Cases:")
    test_cases = tree.generate_test_cases()
    print(f"   Generated {len(test_cases)} test cases")
    
    for i, test_case in enumerate(test_cases):
        print(f"   Test case {i+1}: {test_case['name']}")
    
    print("\n=== Demo Complete ===")
    
    # Show one visualization
    visualize_persistent_tree(tree)

if __name__ == "__main__":
    main() 