# Jarvis March (Gift Wrapping)

A simple algorithm for computing the convex hull of a set of points. Also known as the gift wrapping algorithm.

## Overview

The Jarvis March algorithm finds the convex hull of a set of points by "wrapping" around the points like a gift wrapper. It starts with the leftmost point and repeatedly finds the next point that makes the smallest angle with the current edge.

## Algorithm

1. **Find the leftmost point** (lowest x-coordinate, then lowest y-coordinate if tied)
2. **Start with this point** as the first hull vertex
3. **For each current point**:
   - Find the next point that makes the smallest polar angle with the current edge
   - If multiple points have the same angle, choose the farthest one
   - Add this point to the hull
4. **Continue until** we return to the starting point

## Time Complexity

- **Time**: O(nh) where n is the number of points and h is the number of hull vertices
- **Space**: O(h)

## Implementations

### 1. Basic Implementation
- Standard Jarvis March algorithm
- Uses orientation test for angle comparison
- Handles collinear points correctly

### 2. Optimized Implementation
- Uses angle calculation with `atan2`
- More efficient for large datasets
- Better handling of edge cases

### 3. Visualization Support
- Optional matplotlib integration
- Plots points and convex hull
- Highlights hull vertices

## Usage

```python
from jarvis_march import jarvis_march

# Basic usage
points = [(0, 0), (1, 1), (2, 0), (1, -1), (0.5, 0.5)]
hull = jarvis_march(points)
print(hull)  # [(0, 0), (2, 0), (1, -1), (0, 0)]

# With visualization
from jarvis_march import jarvis_march_with_visualization
hull = jarvis_march_with_visualization(points, show_plot=True)
```

## Mathematical Background

### Orientation Test
For three points p, q, r, the orientation is:
- **0**: Collinear
- **1**: Clockwise
- **2**: Counterclockwise

Calculated using the cross product: `(q.y - p.y) * (r.x - q.x) - (q.x - p.x) * (r.y - q.y)`

### Polar Angle
The angle from the positive x-axis to the line segment from current point to candidate point.

## Applications

### 1. Computer Graphics
- Rendering convex shapes
- Collision detection
- Bounding box computation

### 2. Geographic Information Systems (GIS)
- Territory boundary computation
- Spatial analysis
- Map visualization

### 3. Pattern Recognition
- Shape analysis
- Object detection
- Feature extraction

### 4. Robotics
- Path planning
- Obstacle avoidance
- Workspace analysis

## Performance Analysis

The implementation includes performance analysis:

```python
from jarvis_march import analyze_performance

metrics = analyze_performance(points)
print(f"Number of points: {metrics['num_points']}")
print(f"Hull vertices: {metrics['hull_vertices']}")
print(f"Execution time: {metrics['execution_time']:.6f}s")
print(f"Complexity: {metrics['theoretical_complexity']}")
```

## Visualization

The algorithm includes visualization capabilities:

```python
from jarvis_march import jarvis_march_with_visualization

# Visualize the convex hull
hull = jarvis_march_with_visualization(points, show_plot=True)
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

### Step-by-Step Example
Given points: [(0, 0), (1, 1), (2, 0), (1, -1)]

1. **Start**: Leftmost point (0, 0)
2. **Find next**: Point with smallest angle from horizontal
   - (1, 1): angle = 45°
   - (2, 0): angle = 0°
   - (1, -1): angle = -45°
   - Choose (2, 0)
3. **Continue**: Find next point from (2, 0)
   - Choose (1, -1)
4. **Complete**: Return to (0, 0)

Result: [(0, 0), (2, 0), (1, -1), (0, 0)]

## Edge Cases

### Collinear Points
- Algorithm handles collinear points correctly
- Chooses the farthest point when angles are equal

### Duplicate Points
- Removes duplicate points automatically
- Ensures unique hull vertices

### Degenerate Cases
- Single point: Returns the point itself
- Two points: Returns both points
- Collinear points: Returns the two endpoints

## Comparison with Other Algorithms

| Algorithm | Time Complexity | Space Complexity | Advantages |
|-----------|----------------|------------------|------------|
| Jarvis March | O(nh) | O(h) | Simple, easy to implement |
| Graham Scan | O(n log n) | O(n) | Better for large datasets |
| Quick Hull | O(n log n) | O(n) | Divide and conquer approach |
| Chan's Algorithm | O(n log h) | O(n) | Optimal for small hulls |

## Historical Context

The algorithm is named after R. A. Jarvis, who published it in 1973. It's one of the earliest convex hull algorithms and remains popular due to its simplicity and intuitive geometric interpretation.

## Applications in Computer Science

- **Computational Geometry**: Fundamental algorithm
- **Computer Vision**: Shape analysis and object detection
- **Game Development**: Collision detection and physics
- **Data Visualization**: Boundary computation
- **Machine Learning**: Feature extraction and preprocessing 