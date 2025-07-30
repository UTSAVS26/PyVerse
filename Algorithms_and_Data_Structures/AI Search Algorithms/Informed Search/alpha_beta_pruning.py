def alphabeta(node, depth, alpha, beta, maximizing_player, evaluate, get_children):
    if depth == 0 or not get_children(node):
        return evaluate(node)
    
    if maximizing_player:
        value = float('-inf')
        for child in get_children(node):
            value = max(value, alphabeta(child, depth - 1, alpha, beta, False, evaluate, get_children))
            alpha = max(alpha, value)
            if alpha >= beta:
                break  # β cut-off
        return value
    else:
        value = float('inf')
        for child in get_children(node):
            value = min(value, alphabeta(child, depth - 1, alpha, beta, True, evaluate, get_children))
            beta = min(beta, value)
            if beta <= alpha:
                break  # α cut-off
        return value

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

    best = alphabeta('A', 3, float('-inf'), float('inf'), True, evaluate, get_children)
    print("Best value with Alpha-Beta:", best)

main()