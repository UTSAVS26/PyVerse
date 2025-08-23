# Line Segment Intersection

Algorithms for detecting and computing intersections between line segments. Implements the Bentley-Ottmann sweep line algorithm.

## Overview

The line segment intersection problem involves finding all intersection points between a set of line segments. This is a fundamental problem in computational geometry with applications in computer graphics, GIS, and CAD systems.

## Algorithm

### Bentley-Ottmann Sweep Line Algorithm

1. **Sort events** by x-coordinate (segment endpoints and intersections)
2. **Maintain sweep line status** (active segments sorted by y-coordinate)
3. **Process events**:
   - **Segment start**: Insert segment into status, check for intersections with adjacent segments
   - **Segment end**: Remove segment from status, check for intersections between adjacent segments
   - **Intersection**: Swap segments in status, check for new intersections

### Naive Algorithm

For comparison, also implements the O(n²) naive algorithm:
- Check all pairs of segments
- Use orientation tests to determine intersection
- Calculate intersection point using parametric equations

## Time Complexity

- **Bentley-Ottmann**: O((n+k) log n) where n is number of segments, k is number of intersections
- **Naive**: O(n²)
- **Space**: O(n + k)

## Implementations

### 1. Naive Algorithm
- Checks all pairs of segments
- Simple but inefficient
- Good for small datasets or comparison

### 2. Bentley-Ottmann Sweep Line
- Efficient sweep line algorithm
- Handles all intersection cases
- Optimal for large datasets

### 3. Visualization Support
- Plots segments and intersection points
- Color-coded segments
- Interactive visualization

## Usage

```python
from line_segment_intersection import find_intersections

# Basic usage
segments = [((0, 0), (2, 2)), ((0, 2), (2, 0))]
intersections = find_intersections(segments)
print(intersections)  # [(1.0, 1.0)]

# With visualization
from line_segment_intersection import visualize_intersections
visualize_intersections(segments, intersections, show_plot=True)
```

## Mathematical Background

### Orientation Test
For three points p, q, r:
- **0**: Collinear
- **1**: Clockwise
- **2**: Counterclockwise

### Segment Intersection
Two segments intersect if and only if:
1. The orientations of (p1, p2, q1) and (p1, p2, q2) are different
2. The orientations of (q1, q2, p1) and (q1, q2, p2) are different

### Intersection Point Calculation
Using parametric equations:
- Segment 1: p1 + t1(p2 - p1)
- Segment 2: q1 + t2(q2 - q1)
- Solve for t1, t2 to find intersection

## Applications

### 1. Computer Graphics
- Ray tracing
- Polygon clipping
- Hidden surface removal
- Collision detection

### 2. Geographic Information Systems (GIS)
- Map overlay operations
- Spatial analysis
- Route planning
- Territory boundaries

### 3. Computer-Aided Design (CAD)
- Geometric modeling
- Constraint solving
- Path planning
- Design validation

### 4. Robotics
- Path planning
- Obstacle avoidance
- Workspace analysis
- Sensor data processing

## Performance Analysis

The implementation includes performance comparison:

```python
from line_segment_intersection import analyze_performance

metrics = analyze_performance(segments)
print(f"Number of segments: {metrics['num_segments']}")
print(f"Number of intersections: {metrics['num_intersections']}")
print(f"Naive time: {metrics['naive_time']:.6f}s")
print(f"Sweep line time: {metrics['sweep_time']:.6f}s")
print(f"Speedup: {metrics['speedup']:.2f}x")
```

## Visualization

The algorithm includes comprehensive visualization:

```python
from line_segment_intersection import visualize_intersections

# Visualize segments and intersections
visualize_intersections(segments, intersections, show_plot=True)
```

## Requirements

- Python 3.7+
- NumPy (for numerical operations)
- Matplotlib (for visualization, optional)

## Installation

```bash
pip install numpy matplotlib
```

## Algorithm Details

### Sweep Line Events
1. **Segment Start**: Insert segment into status
2. **Segment End**: Remove segment from status
3. **Intersection**: Swap segments in status

### Status Structure
- Balanced binary search tree
- Segments ordered by y-coordinate at current x
- Efficient insertion/deletion operations

### Intersection Detection
- Check adjacent segments in status
- Use orientation tests
- Calculate intersection points

## Edge Cases

### Collinear Segments
- Handles overlapping segments
- Detects collinear intersections
- Proper endpoint handling

### Degenerate Cases
- Single point segments
- Zero-length segments
- Identical segments

### Numerical Precision
- Uses tolerance for floating-point comparisons
- Handles near-collinear segments
- Robust intersection calculations

## Comparison with Other Algorithms

| Algorithm | Time Complexity | Space Complexity | Advantages |
|-----------|----------------|------------------|------------|
| Naive | O(n²) | O(1) | Simple, easy to implement |
| Bentley-Ottmann | O((n+k) log n) | O(n+k) | Optimal for many intersections |
| Plane Sweep | O(n log n) | O(n) | Good for few intersections |
| Randomized | O(n log n) | O(n) | Expected case optimal |

## Historical Context

The Bentley-Ottmann algorithm was developed by Jon Bentley and Thomas Ottmann in 1979. It was one of the first efficient algorithms for the line segment intersection problem and remains a fundamental algorithm in computational geometry.

## Applications in Computer Science

- **Computational Geometry**: Fundamental algorithm
- **Computer Vision**: Feature detection and matching
- **Game Development**: Physics and collision detection
- **Data Visualization**: Graph layout algorithms
- **Machine Learning**: Geometric feature extraction 