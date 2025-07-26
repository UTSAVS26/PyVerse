
def ao_star(node, adj, heuristic, solved, solution):
    if node in solved:
        return heuristic[node]
    
    if not adj[node]: # leaf node
        solved[node] = True
        return heuristic[node]
    
    min_cost = float('inf')
    best_children = None
    
    for children in adj[node]: # children is a list: OR=[child], AND=[child1, child2]
        cost = 0 
        for child in children:
            cost += ao_star(child, adj, heuristic, solved, solution)
        if cost < min_cost:
            min_cost = cost
            best_children = children
            
    heuristic[node] = min_cost
    solution[node] = best_children
    
    all_solved = True
    for child in best_children:
        if child not in solved:
            all_solved = False
            break

    
    if all_solved:
        solved[node] = True
    
    return heuristic[node] # min cost

def main():
    adj = {                        # consider edge cost as 1
        'A': [['B', 'C'], ['D']],  # A is an OR node with two options: AND(B, C) or D
        'B': [['E']],              # B -> E
        'C': [['F']],              # C -> F
        'D': [],                   # D is a leaf
        'E': [],                   # E is a leaf
        'F': []                    # F is a leaf
    }

    heuristic = {
        'A': 999, 'B': 999, 'C': 999, 'D': 3, 'E': 1, 'F': 2
    }

    solved = {}
    solution = {}

    ao_star('A', adj, heuristic, solved, solution)

    print("Final Heuristic Values:", heuristic)
    print("Solution Graph:")
    for k, v in solution.items():
        print(f"{k} -> {v}")

main()