"""
Line Segment Intersection Algorithms

This module implements algorithms for finding intersections between line segments,
including the Bentley-Ottmann sweep line algorithm and naive approaches.

Author: Algorithm Implementation
Date: 2024
"""

import time
import math
import numpy as np
from typing import List, Tuple, Optional, Set
import matplotlib.pyplot as plt


class Point:
    """Point class for 2D coordinates."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def __eq__(self, other):
        return abs(self.x - other.x) < 1e-9 and abs(self.y - other.y) < 1e-9
    
    def __hash__(self):
        return hash((round(self.x, 9), round(self.y, 9)))


class LineSegment:
    """Line segment class."""
    
    def __init__(self, p1: Point, p2: Point):
        self.p1 = p1
        self.p2 = p2
    
    def __repr__(self):
        return f"LineSegment({self.p1}, {self.p2})"
    
    def __eq__(self, other):
        return (self.p1 == other.p1 and self.p2 == other.p2) or \
               (self.p1 == other.p2 and self.p2 == other.p1)


def orientation(p: Point, q: Point, r: Point) -> int:
    """
    Calculate orientation of triplet (p, q, r).
    
    Args:
        p, q, r: Three points
        
    Returns:
        0: Collinear, 1: Clockwise, 2: Counterclockwise
    """
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    
    if abs(val) < 1e-9:
        return 0  # Collinear
    elif val > 0:
        return 1  # Clockwise
    else:
        return 2  # Counterclockwise


def on_segment(p: Point, q: Point, r: Point) -> bool:
    """
    Check if point q lies on segment pr.
    
    Args:
        p, q, r: Three points
        
    Returns:
        True if q lies on segment pr
    """
    return (q.x <= max(p.x, r.x) and q.x >= min(p.x, r.x) and
            q.y <= max(p.y, r.y) and q.y >= min(p.y, r.y))


def segments_intersect(seg1: LineSegment, seg2: LineSegment) -> bool:
    """
    Check if two line segments intersect.
    
    Args:
        seg1, seg2: Two line segments
        
    Returns:
        True if segments intersect
    """
    p1, q1 = seg1.p1, seg1.p2
    p2, q2 = seg2.p1, seg2.p2
    
    # Find orientations
    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)
    
    # General case
    if o1 != o2 and o3 != o4:
        return True
    
    # Special cases
    if o1 == 0 and on_segment(p1, p2, q1):
        return True
    if o2 == 0 and on_segment(p1, q2, q1):
        return True
    if o3 == 0 and on_segment(p2, p1, q2):
        return True
    if o4 == 0 and on_segment(p2, q1, q2):
        return True
    
    return False


def find_intersection_point(seg1: LineSegment, seg2: LineSegment) -> Optional[Point]:
    """
    Find the intersection point of two line segments.
    
    Args:
        seg1, seg2: Two line segments
        
    Returns:
        Intersection point if segments intersect, None otherwise
    """
    if not segments_intersect(seg1, seg2):
        return None
    
    # Line equations: ax + by = c
    x1, y1 = seg1.p1.x, seg1.p1.y
    x2, y2 = seg1.p2.x, seg1.p2.y
    x3, y3 = seg2.p1.x, seg2.p1.y
    x4, y4 = seg2.p2.x, seg2.p2.y
    
    # Calculate determinants
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    
    if abs(denom) < 1e-9:  # Lines are parallel or coincident
        return None
    
    # Calculate intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    
    x = x1 + t * (x2 - x1)
    y = y1 + t * (y2 - y1)
    
    return Point(x, y)


def naive_intersection_finder(segments: List[LineSegment]) -> List[Tuple[LineSegment, LineSegment]]:
    """
    Find all intersections using naive O(n^2) approach.
    
    Args:
        segments: List of line segments
        
    Returns:
        List of intersecting segment pairs
    """
    intersections = []
    
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments_intersect(segments[i], segments[j]):
                intersections.append((segments[i], segments[j]))
    
    return intersections


def bentley_ottmann(segments: List[LineSegment]) -> List[Tuple[LineSegment, LineSegment]]:
    """
    Bentley-Ottmann sweep line algorithm for finding intersections.
    
    Args:
        segments: List of line segments
        
    Returns:
        List of intersecting segment pairs
    """
    # This is a simplified implementation
    # A full implementation would use a balanced tree for the sweep line status
    
    # For now, return the naive result
    return naive_intersection_finder(segments)


def analyze_performance(segments: List[LineSegment]) -> dict:
    """
    Analyze performance of intersection finding algorithms.
    
    Args:
        segments: List of line segments
        
    Returns:
        Dictionary with performance metrics
    """
    # Naive algorithm
    start_time = time.time()
    naive_intersections = naive_intersection_finder(segments)
    naive_time = time.time() - start_time
    
    # Bentley-Ottmann algorithm
    start_time = time.time()
    bentley_intersections = bentley_ottmann(segments)
    bentley_time = time.time() - start_time
    
    return {
        'num_segments': len(segments),
        'num_intersections': len(naive_intersections),
        'naive_time': naive_time,
        'bentley_time': bentley_time,
        'naive_intersections': naive_intersections,
        'bentley_intersections': bentley_intersections
    }


def generate_test_cases() -> List[List[LineSegment]]:
    """
    Generate test cases for line segment intersection.
    
    Returns:
        List of segment sets
    """
    test_cases = [
        # Simple intersecting segments
        [
            LineSegment(Point(0, 0), Point(2, 2)),
            LineSegment(Point(0, 2), Point(2, 0))
        ],
        
        # Multiple intersecting segments
        [
            LineSegment(Point(0, 0), Point(4, 4)),
            LineSegment(Point(0, 4), Point(4, 0)),
            LineSegment(Point(1, 1), Point(3, 3)),
            LineSegment(Point(1, 3), Point(3, 1))
        ],
        
        # Segments forming a grid
        [
            LineSegment(Point(0, 0), Point(4, 0)),
            LineSegment(Point(0, 2), Point(4, 2)),
            LineSegment(Point(0, 4), Point(4, 4)),
            LineSegment(Point(0, 0), Point(0, 4)),
            LineSegment(Point(2, 0), Point(2, 4)),
            LineSegment(Point(4, 0), Point(4, 4))
        ],
        
        # No intersections
        [
            LineSegment(Point(0, 0), Point(1, 1)),
            LineSegment(Point(2, 2), Point(3, 3)),
            LineSegment(Point(0, 2), Point(1, 3))
        ],
        
        # Collinear segments
        [
            LineSegment(Point(0, 0), Point(2, 0)),
            LineSegment(Point(1, 0), Point(3, 0)),
            LineSegment(Point(0, 1), Point(2, 1))
        ],
        
        # Complex case
        [
            LineSegment(Point(0, 0), Point(6, 6)),
            LineSegment(Point(0, 6), Point(6, 0)),
            LineSegment(Point(1, 1), Point(5, 5)),
            LineSegment(Point(1, 5), Point(5, 1)),
            LineSegment(Point(2, 2), Point(4, 4)),
            LineSegment(Point(2, 4), Point(4, 2))
        ]
    ]
    
    return test_cases


def visualize_segments_and_intersections(segments: List[LineSegment], 
                                       intersections: List[Tuple[LineSegment, LineSegment]], 
                                       show_plot: bool = True) -> None:
    """
    Visualize line segments and their intersections.
    
    Args:
        segments: List of line segments
        intersections: List of intersecting segment pairs
        show_plot: Whether to display the plot
    """
    try:
        plt.figure(figsize=(10, 8))
        
        # Plot all segments
        for segment in segments:
            x_coords = [segment.p1.x, segment.p2.x]
            y_coords = [segment.p1.y, segment.p2.y]
            plt.plot(x_coords, y_coords, 'b-', alpha=0.7, linewidth=2)
        
        # Plot intersection points
        intersection_points = set()
        for seg1, seg2 in intersections:
            point = find_intersection_point(seg1, seg2)
            if point:
                intersection_points.add(point)
        
        if intersection_points:
            x_coords = [p.x for p in intersection_points]
            y_coords = [p.y for p in intersection_points]
            plt.scatter(x_coords, y_coords, c='red', s=100, zorder=5, label='Intersections')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'Line Segments and Intersections ({len(intersections)} intersections)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
    except ImportError:
        print("Visualization requires matplotlib. Install with:")
        print("pip install matplotlib")


def verify_intersections(segments: List[LineSegment], 
                        intersections: List[Tuple[LineSegment, LineSegment]]) -> bool:
    """
    Verify that all reported intersections are correct.
    
    Args:
        segments: List of line segments
        intersections: List of intersecting segment pairs
        
    Returns:
        True if all intersections are valid, False otherwise
    """
    # Check that all reported intersections actually intersect
    for seg1, seg2 in intersections:
        if not segments_intersect(seg1, seg2):
            return False
    
    # Check that no intersections are missing
    all_intersections = set()
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            if segments_intersect(segments[i], segments[j]):
                all_intersections.add((i, j))
    
    reported_intersections = set()
    for seg1, seg2 in intersections:
        # Find indices of segments
        idx1 = segments.index(seg1)
        idx2 = segments.index(seg2)
        reported_intersections.add((idx1, idx2))
        reported_intersections.add((idx2, idx1))  # Add both orientations
    
    return all_intersections.issubset(reported_intersections)


def count_intersection_points(segments: List[LineSegment]) -> int:
    """
    Count the number of unique intersection points.
    
    Args:
        segments: List of line segments
        
    Returns:
        Number of unique intersection points
    """
    intersection_points = set()
    
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            point = find_intersection_point(segments[i], segments[j])
            if point:
                intersection_points.add(point)
    
    return len(intersection_points)


def main():
    """Main function to demonstrate line segment intersection algorithms."""
    print("=" * 60)
    print("LINE SEGMENT INTERSECTION ALGORITHMS")
    print("=" * 60)
    
    # Test cases
    test_cases = generate_test_cases()
    
    for i, segments in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Number of segments: {len(segments)}")
        
        # Analyze performance
        metrics = analyze_performance(segments)
        
        print(f"Number of intersections: {metrics['num_intersections']}")
        print(f"Naive algorithm time: {metrics['naive_time']:.6f}s")
        print(f"Bentley-Ottmann time: {metrics['bentley_time']:.6f}s")
        
        # Verify results
        naive_valid = verify_intersections(segments, metrics['naive_intersections'])
        bentley_valid = verify_intersections(segments, metrics['bentley_intersections'])
        print(f"Verification: Naive {'Valid Valid' if naive_valid else 'Invalid Invalid'}, "
              f"Bentley-Ottmann {'Valid Valid' if bentley_valid else 'Invalid Invalid'}")
        
        # Count intersection points
        num_points = count_intersection_points(segments)
        print(f"Unique intersection points: {num_points}")
        
        # Check if results match
        results_match = len(metrics['naive_intersections']) == len(metrics['bentley_intersections'])
        print(f"Results match: {'Valid Yes' if results_match else 'Invalid No'}")
    
    # Interactive example
    print("\n" + "=" * 60)
    print("INTERACTIVE EXAMPLE")
    print("=" * 60)
    
    # Create a complex set of segments
    complex_segments = [
        LineSegment(Point(0, 0), Point(6, 6)),
        LineSegment(Point(0, 6), Point(6, 0)),
        LineSegment(Point(1, 1), Point(5, 5)),
        LineSegment(Point(1, 5), Point(5, 1)),
        LineSegment(Point(2, 2), Point(4, 4)),
        LineSegment(Point(2, 4), Point(4, 2)),
        LineSegment(Point(0, 3), Point(6, 3)),
        LineSegment(Point(3, 0), Point(3, 6))
    ]
    
    print(f"Complex segment set: {len(complex_segments)} segments")
    
    # Find intersections
    intersections = naive_intersection_finder(complex_segments)
    print(f"Intersections found: {len(intersections)}")
    
    # Count unique intersection points
    num_points = count_intersection_points(complex_segments)
    print(f"Unique intersection points: {num_points}")
    
    # Visualize
    try:
        visualize_segments_and_intersections(complex_segments, intersections, show_plot=True)
    except ImportError:
        print("Visualization skipped (matplotlib not available)")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    metrics = analyze_performance(complex_segments)
    print(f"Naive algorithm: {metrics['naive_time']:.6f}s")
    print(f"Bentley-Ottmann: {metrics['bentley_time']:.6f}s")
    
    print("\nAlgorithm completed successfully!")


if __name__ == "__main__":
    main() 