"""
Jarvis March (Gift Wrapping) Algorithm

This module implements the Jarvis March algorithm for finding the convex hull
of a set of points in 2D space.

Author: Algorithm Implementation
Date: 2024
"""

import time
import math
import numpy as np
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt


class Point:
    """Point class for 2D coordinates."""
    
    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y
    
    def __repr__(self):
        return f"Point({self.x}, {self.y})"
    
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __hash__(self):
        return hash((self.x, self.y))


def orientation(p: Point, q: Point, r: Point) -> int:
    """
    Calculate orientation of triplet (p, q, r).
    
    Args:
        p, q, r: Three points
        
    Returns:
        0: Collinear, 1: Clockwise, 2: Counterclockwise
    """
    val = (q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)
    
    if val == 0:
        return 0  # Collinear
    elif val > 0:
        return 1  # Clockwise
    else:
        return 2  # Counterclockwise


def distance(p1: Point, p2: Point) -> float:
    """
    Calculate Euclidean distance between two points.
    
    Args:
        p1, p2: Two points
        
    Returns:
        Distance between points
    """
    return math.sqrt((p2.x - p1.x)**2 + (p2.y - p1.y)**2)


def jarvis_march(points: List[Point]) -> List[Point]:
    """
    Find convex hull using Jarvis March (Gift Wrapping) algorithm.
    
    Args:
        points: List of points
        
    Returns:
        List of points forming the convex hull in counterclockwise order
    """
    if len(points) < 3:
        return points
    
    # Find the leftmost point
    leftmost = min(points, key=lambda p: (p.x, p.y))
    
    hull = []
    current = leftmost
    
    while True:
        hull.append(current)
        
        # Find the next point with minimum polar angle
        next_point = None
        for point in points:
            if point == current:
                continue
            
            if next_point is None:
                next_point = point
            else:
                orient = orientation(current, next_point, point)
                if orient == 2:  # Counterclockwise
                    next_point = point
                elif orient == 0:  # Collinear
                    # Choose the farthest point
                    if distance(current, point) > distance(current, next_point):
                        next_point = point
        
        # If we're back to the starting point, we're done
        if next_point == leftmost:
            break
        
        current = next_point
    
    return hull


def jarvis_march_optimized(points: List[Point]) -> List[Point]:
    """
    Optimized version of Jarvis March with better handling of collinear points.
    
    Args:
        points: List of points
        
    Returns:
        List of points forming the convex hull
    """
    if len(points) < 3:
        return points
    
    # Find the leftmost point
    leftmost = min(points, key=lambda p: (p.x, p.y))
    
    hull = []
    current = leftmost
    
    while True:
        hull.append(current)
        
        # Find the next point with minimum polar angle
        next_point = None
        min_angle = float('inf')
        
        for point in points:
            if point == current:
                continue
            
            # Calculate polar angle
            dx = point.x - current.x
            dy = point.y - current.y
            
            if dx == 0 and dy == 0:
                continue
            
            angle = math.atan2(dy, dx)
            if angle < 0:
                angle += 2 * math.pi
            
            if angle < min_angle:
                min_angle = angle
                next_point = point
            elif angle == min_angle:
                # If angles are equal, choose the farthest point
                if distance(current, point) > distance(current, next_point):
                    next_point = point
        
        # If we're back to the starting point, we're done
        if next_point == leftmost:
            break
        
        current = next_point
    
    return hull


def graham_scan(points: List[Point]) -> List[Point]:
    """
    Graham Scan algorithm for comparison.
    
    Args:
        points: List of points
        
    Returns:
        List of points forming the convex hull
    """
    if len(points) < 3:
        return points
    
    # Find the lowest point (and leftmost if tied)
    lowest = min(points, key=lambda p: (p.y, p.x))
    
    # Sort points by polar angle with respect to lowest point
    def polar_angle(p):
        if p == lowest:
            return -1
        return math.atan2(p.y - lowest.y, p.x - lowest.x)
    
    sorted_points = sorted(points, key=polar_angle)
    
    # Remove duplicates
    unique_points = []
    for i, point in enumerate(sorted_points):
        if i == 0 or point != sorted_points[i-1]:
            unique_points.append(point)
    
    if len(unique_points) < 3:
        return unique_points
    
    # Graham scan
    hull = [unique_points[0], unique_points[1]]
    
    for i in range(2, len(unique_points)):
        while len(hull) > 1 and orientation(hull[-2], hull[-1], unique_points[i]) != 2:
            hull.pop()
        hull.append(unique_points[i])
    
    return hull


def analyze_performance(points: List[Point]) -> dict:
    """
    Analyze performance of convex hull algorithms.
    
    Args:
        points: List of points
        
    Returns:
        Dictionary with performance metrics
    """
    # Jarvis March
    start_time = time.time()
    jarvis_hull = jarvis_march(points)
    jarvis_time = time.time() - start_time
    
    # Optimized Jarvis March
    start_time = time.time()
    jarvis_opt_hull = jarvis_march_optimized(points)
    jarvis_opt_time = time.time() - start_time
    
    # Graham Scan
    start_time = time.time()
    graham_hull = graham_scan(points)
    graham_time = time.time() - start_time
    
    return {
        'num_points': len(points),
        'hull_size': len(jarvis_hull),
        'jarvis_time': jarvis_time,
        'jarvis_opt_time': jarvis_opt_time,
        'graham_time': graham_time,
        'jarvis_hull': jarvis_hull,
        'jarvis_opt_hull': jarvis_opt_hull,
        'graham_hull': graham_hull
    }


def generate_test_cases() -> List[List[Point]]:
    """
    Generate test cases for convex hull algorithms.
    
    Returns:
        List of point sets
    """
    test_cases = [
        # Simple convex hull
        [Point(0, 0), Point(1, 0), Point(0, 1), Point(1, 1)],
        
        # Points forming a square
        [Point(0, 0), Point(2, 0), Point(2, 2), Point(0, 2), Point(1, 1)],
        
        # Points with collinear edges
        [Point(0, 0), Point(1, 0), Point(2, 0), Point(0, 1), Point(1, 1), Point(2, 1)],
        
        # Random points
        [Point(1, 1), Point(2, 3), Point(4, 2), Point(5, 5), Point(3, 4), Point(6, 1)],
        
        # Points forming a triangle
        [Point(0, 0), Point(2, 0), Point(1, 2), Point(1, 1)],
        
        # Large set of points
        [Point(x, y) for x, y in [
            (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
            (0, 1), (1, 1), (2, 1), (3, 1), (4, 1),
            (0, 2), (1, 2), (2, 2), (3, 2), (4, 2),
            (0, 3), (1, 3), (2, 3), (3, 3), (4, 3),
            (0, 4), (1, 4), (2, 4), (3, 4), (4, 4)
        ]]
    ]
    
    return test_cases


def visualize_convex_hull(points: List[Point], hull: List[Point], 
                         algorithm: str = "Jarvis March", 
                         show_plot: bool = True) -> None:
    """
    Visualize the points and convex hull.
    
    Args:
        points: List of all points
        hull: List of hull points
        algorithm: Name of the algorithm used
        show_plot: Whether to display the plot
    """
    try:
        plt.figure(figsize=(10, 8))
        
        # Plot all points
        x_coords = [p.x for p in points]
        y_coords = [p.y for p in points]
        plt.scatter(x_coords, y_coords, c='blue', s=50, alpha=0.6, label='Points')
        
        # Plot hull points
        hull_x = [p.x for p in hull]
        hull_y = [p.y for p in hull]
        plt.scatter(hull_x, hull_y, c='red', s=100, label='Hull Points')
        
        # Connect hull points
        if len(hull) > 1:
            hull_x.append(hull[0].x)  # Close the polygon
            hull_y.append(hull[0].y)
            plt.plot(hull_x, hull_y, 'r-', linewidth=2, label='Convex Hull')
        
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f'{algorithm} - Convex Hull ({len(hull)} points)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        
        if show_plot:
            plt.show()
        
    except ImportError:
        print("Visualization requires matplotlib. Install with:")
        print("pip install matplotlib")


def verify_convex_hull(points: List[Point], hull: List[Point]) -> bool:
    """
    Verify that the convex hull is correct.
    
    Args:
        points: List of all points
        hull: List of hull points
        
    Returns:
        True if hull is valid, False otherwise
    """
    if len(hull) < 3:
        return len(points) == len(hull)
    
    # Check that all hull points are in the original set
    hull_set = set(hull)
    points_set = set(points)
    if not hull_set.issubset(points_set):
        return False
    
    # Check that all other points are inside or on the hull
    for point in points:
        if point in hull:
            continue
        
        # Check if point is inside the convex hull
        inside = False
        n = len(hull)
        
        for i in range(n):
            p1 = hull[i]
            p2 = hull[(i + 1) % n]
            
            # Check if point is on the same side of all edges
            orient = orientation(p1, p2, point)
            if orient == 1:  # Clockwise - point is outside
                return False
    
    return True


def calculate_hull_area(hull: List[Point]) -> float:
    """
    Calculate the area of the convex hull.
    
    Args:
        hull: List of hull points
        
    Returns:
        Area of the convex hull
    """
    if len(hull) < 3:
        return 0.0
    
    area = 0.0
    n = len(hull)
    
    for i in range(n):
        j = (i + 1) % n
        area += hull[i].x * hull[j].y
        area -= hull[j].x * hull[i].y
    
    return abs(area) / 2.0


def main():
    """Main function to demonstrate Jarvis March algorithm."""
    print("=" * 60)
    print("JARVIS MARCH (GIFT WRAPPING) ALGORITHM")
    print("=" * 60)
    
    # Test cases
    test_cases = generate_test_cases()
    
    for i, points in enumerate(test_cases, 1):
        print(f"\nTest Case {i}:")
        print(f"Number of points: {len(points)}")
        
        # Analyze performance
        metrics = analyze_performance(points)
        
        print(f"Hull size: {metrics['hull_size']}")
        print(f"Jarvis March time: {metrics['jarvis_time']:.6f}s")
        print(f"Optimized Jarvis time: {metrics['jarvis_opt_time']:.6f}s")
        print(f"Graham Scan time: {metrics['graham_time']:.6f}s")
        
        # Verify results
        jarvis_valid = verify_convex_hull(points, metrics['jarvis_hull'])
        graham_valid = verify_convex_hull(points, metrics['graham_hull'])
        print(f"Verification: Jarvis {'Valid Valid' if jarvis_valid else 'Invalid Invalid'}, "
              f"Graham {'Valid Valid' if graham_valid else 'Invalid Invalid'}")
        
        # Calculate area
        area = calculate_hull_area(metrics['jarvis_hull'])
        print(f"Hull area: {area:.2f}")
        
        # Check if results match
        results_match = len(metrics['jarvis_hull']) == len(metrics['graham_hull'])
        print(f"Results match: {'Valid Yes' if results_match else 'Invalid No'}")
    
    # Interactive example
    print("\n" + "=" * 60)
    print("INTERACTIVE EXAMPLE")
    print("=" * 60)
    
    # Create a complex set of points
    complex_points = [
        Point(0, 0), Point(2, 0), Point(4, 0), Point(6, 0),
        Point(0, 2), Point(2, 2), Point(4, 2), Point(6, 2),
        Point(0, 4), Point(2, 4), Point(4, 4), Point(6, 4),
        Point(1, 1), Point(3, 1), Point(5, 1),
        Point(1, 3), Point(3, 3), Point(5, 3)
    ]
    
    print(f"Complex point set: {len(complex_points)} points")
    
    # Find convex hull
    hull = jarvis_march(complex_points)
    print(f"Convex hull: {len(hull)} points")
    print(f"Hull points: {hull}")
    
    # Calculate area
    area = calculate_hull_area(hull)
    print(f"Hull area: {area:.2f}")
    
    # Visualize
    try:
        visualize_convex_hull(complex_points, hull, "Jarvis March", show_plot=True)
    except ImportError:
        print("Visualization skipped (matplotlib not available)")
    
    # Performance comparison
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON")
    print("=" * 60)
    
    metrics = analyze_performance(complex_points)
    print(f"Jarvis March: {metrics['jarvis_time']:.6f}s")
    print(f"Optimized Jarvis: {metrics['jarvis_opt_time']:.6f}s")
    print(f"Graham Scan: {metrics['graham_time']:.6f}s")
    
    print("\nAlgorithm completed successfully!")


if __name__ == "__main__":
    main() 