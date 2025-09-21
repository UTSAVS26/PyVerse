def insert_sorted(stack, element):
    if not stack or element > stack[-1]:
        stack.append(element)
        return
    temp = stack.pop()
    insert_sorted(stack, element)
    stack.append(temp)

def sort_stack(stack):
    if len(stack) <= 1:
        return
    temp = stack.pop()
    sort_stack(stack)
    insert_sorted(stack, temp)

# Example
stack = [3, 1, 4, 2]
sort_stack(stack)
print("Sorted stack:", stack)
