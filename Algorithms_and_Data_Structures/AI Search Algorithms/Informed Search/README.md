# Informed (Heuristic) Search Algorithms

Informed search algorithms use **heuristics** to improve search efficiency. A heuristic function `h(n)` estimates the cost from node `n` to the goal. These algorithms often find optimal solutions faster than uninformed ones.

---

## 1. A* Search

- Uses: `f(n) = g(n) + h(n)`
- `g(n)`: Cost from the start node to `n`
- `h(n)`: Estimated cost from `n` to goal
- Guarantees optimality if `h(n)` is admissible (never overestimates).

**Pros**: Optimal and complete  
**Cons**: Memory-intensive

---

## 2. Best First Search

- Greedy approach using only `h(n)` to decide the next node.
- Prioritizes the most promising node based on the heuristic.

**Pros**: Faster in some cases  
**Cons**: Not optimal or complete

---

## 3. Hill Climbing

- Iteratively moves to the neighboring state with the lowest `h(n)`.
- Stops when no better neighbors exist (local maximum).

**Pros**: Simple, uses less memory  
**Cons**: May get stuck in local maxima or plateaus

---

## 4. AO* Search

- Used for **AND-OR** graph structures.
- Selects the best partial solution path based on cost and heuristics.
- Efficient for decision trees with **non-deterministic outcomes**.

**Pros**: Optimal for AND-OR graphs  
**Cons**: More complex to implement

---

## 5. Minimax Algorithm

- Used in **two-player adversarial games** like chess.
- Each node has a minimax value:
  - `Max` player tries to maximize score.
  - `Min` player tries to minimize it.

**Pros**: Decision-making under uncertainty  
**Cons**: High branching factor

---

## 6. Alpha-Beta Pruning

- Optimization of Minimax.
- Eliminates branches that don't influence the final decision.

**Pros**: Reduces time complexity  
**Cons**: Requires good node ordering for max benefit

---

## Time & Space Complexity Overview

| Algorithm          | Time Complexity      | Space Complexity     | Optimal | Complete |
|--------------------|----------------------|-----------------------|---------|----------|
| A* Search          | O(b^d)               | O(b^d)                | ✅ Yes (if h is admissible) | ✅ Yes |
| Best First Search  | O(b^m)               | O(b^m)                | ❌ No   | ❌ No     |
| Hill Climbing      | O(b^m)               | O(b)                  | ❌ No   | ❌ No     |
| AO* Search         | Varies (depends on graph) | Varies            | ✅ Yes (if heuristic is admissible) | ✅ Yes |
| Minimax            | O(b^d)               | O(bd)                 | ✅ Yes | ✅ Yes    |
| Alpha-Beta Pruning | O(b^(d/2)) [Best Case]| O(bd)                | ✅ Yes | ✅ Yes    |

**Legend:**
- `b` = branching factor
- `d` = depth of the goal node
- `m` = maximum depth of the search space
