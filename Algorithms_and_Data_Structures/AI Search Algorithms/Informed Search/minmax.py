def minimax(node, depth, maximizing_player, evaluate, get_children):
    if depth == 0 or not get_children(node):
        return evaluate(node)
    
    if maximizing_player:
        max_eval = float('-inf')
        for child in get_children(node):
            eval = minimax(child, depth - 1, False, evaluate, get_children)
            max_eval = max(max_eval, eval)
        return max_eval
    else:
        min_eval = float('inf')
        for child in get_children(node):
            eval = minimax(child, depth - 1, True, evaluate, get_children)
            min_eval = min(min_eval, eval)
        return min_eval


def main():
    tree = {
        'A': ['B', 'C'],
        'B': ['D', 'E'],
        'C': ['F', 'G'],
        'D': [], 'E': [], 'F': [], 'G': []
    }
    values = {
        'D': 3,
        'E': 5,
        'F': 2,
        'G': 9
    }
    
    def evaluate(node):
        return values.get(node, 0)

    def get_children(node):
        return tree.get(node, [])
    
    best_value = minimax('A', 3, True, evaluate, get_children)
    print("Best value for maximizing player:", best_value)

main()