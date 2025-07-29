import random

def hill_climb(start, evaluate, neighbors, max_iterations=1000):
    current = start
    for _ in range(max_iterations):
        neighbor_list = neighbors(current)
        next_node = max(neighbor_list, key=evaluate, default=current)
        if evaluate(next_node) <= evaluate(current):
            break  # No better neighbor found
        current = next_node
    return current

# Example: Maximize f(x) = -x^2 + 10x
def objective(x):
    return -x**2 + 10*x

def get_neighbors(x):
    step_size = 0.1
    return [x + step_size, x - step_size]

def main():
    start = random.uniform(0, 10)
    result = hill_climb(start, objective, get_neighbors)
    print(f"Start at x = {start:.2f}")
    print(f"Best x = {result:.2f}, f(x) = {objective(result):.2f}")

main()