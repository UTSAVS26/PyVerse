def insert_bottom(stack, item):
    if not stack:
        stack.append(item)
        return
    temp = stack.pop()
    insert_bottom(stack, item)
    stack.append(temp)

def reverse_stack(stack):
    if len(stack) <= 1:
        return
    temp = stack.pop()
    reverse_stack(stack)
    insert_bottom(stack, temp)

# Example
stack = [1, 2, 3, 4]
reverse_stack(stack)
print("Reversed stack:", stack)
