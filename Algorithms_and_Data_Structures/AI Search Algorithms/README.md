# AI Search Algorithms – Overview

This section of the DSA repository introduces **AI Search Algorithms**, a fundamental part of Artificial Intelligence used for problem-solving and decision-making.

Search algorithms are categorized into:

## 1. Uninformed Search (Blind Search)

These algorithms explore the search space **without any domain-specific knowledge**. They do not know whether a state is good or bad until they reach it.

- Breadth-First Search (BFS)
- Depth-First Search (DFS)

## 2. Informed Search (Heuristic-Based Search)

These algorithms use a **heuristic function** to estimate the cost from the current node to the goal, making them more efficient for complex problems.

- Best First Search
- A* Search
- AO* Search
- Hill Climbing
- Minimax
- Alpha-Beta Pruning

---

## Search Problem Definition

A search problem is defined by:

- **Initial state**: Starting point.
- **Goal state**: Desired end state.
- **Successor function**: Defines possible moves from a given state.
- **Cost function**: (Optional) Evaluates the cost of moving from one state to another.
- **Heuristic function**: (Only in informed search) Provides an estimate of the distance to the goal.

---

## Directory Structure

```bash
AI Search Algorithms/
│
├── Informed Search/
│   ├── 8_puzzle_a_star.py
│   ├── 8_puzzle_best_first.py
│   ├── a_star_search.py
│   ├── alpha_beta_pruning.py
│   ├── ao_star_search.py
│   ├── best_first_search.py
│   ├── hill_climb.py
│   ├── minmax.py
│   └── README.md
│
├── Uninformed Search/
│   ├── 8_puzzle_bfs.py
│   ├── 8_puzzle_dfs.py
│   ├── bfs.py
│   ├── dfs.py
│   └── README.md
│
└── README.md
