def asteroid_collision(asteroids):
    """
    LeetCode #735 - Asteroid Collision
    ----------------------------------
    Simulate asteroid collisions moving along a line.
    Positive -> right, Negative -> left.
    """
    stack = []
    for a in asteroids:
        while stack and stack[-1] > 0 and a < 0 and stack[-1] < abs(a):
            stack.pop()
        if not stack or a > 0 or stack[-1] < 0:
            stack.append(a)
        elif stack[-1] == -a:
            stack.pop()
    return stack


if __name__ == "__main__":
    arr = list(map(int, input("Enter asteroid values separated by space: ").split()))
    print("Result after collisions:", asteroid_collision(arr))
