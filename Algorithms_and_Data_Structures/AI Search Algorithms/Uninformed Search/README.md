# Uninformed (Blind) Search Algorithms

Uninformed search algorithms explore the search space **without using problem-specific knowledge**. They are guaranteed to find a solution (if one exists), but are not always efficient.

---

## 1. Breadth-First Search (BFS)

- Explores all nodes at the current depth before moving deeper.
- Uses a **queue (FIFO)**.
- Guarantees the **shortest path** in an unweighted graph.

**Pros**: Complete and optimal  
**Cons**: High memory usage (stores all nodes at a level)

---

## 2. Depth-First Search (DFS)

- Explores as far as possible down each branch before backtracking.
- Uses a **stack (LIFO)** (or recursion).
- May go deep into infinite branches.

**Pros**: Memory-efficient  
**Cons**: Not complete or optimal in infinite/deep graphs

---

## Key Differences

| Feature       | BFS             | DFS             |
|---------------|------------------|------------------|
| Data Structure| Queue (FIFO)     | Stack (LIFO)     |
| Completeness  | ✅ Yes            | ❌ No (not always)|
| Optimality    | ✅ Yes            | ❌ No             |
| Memory Usage  | ❌ High           | ✅ Low            |
| Time Complexity  | O(b^d)           | O(b^m)            |
| Space Complexity  | O(b^d)      | O(m)            |

**Legend:**
- `b` = branching factor
- `d` = depth of the goal node
- `m` = maximum depth of the search space

These are foundational algorithms that are essential to understanding more advanced AI techniques.
