def is_stack_permutation(input, output):
    stack = []  # Use list as a stack
    j = 0  # Pointer for the output sequence
    for i in range(len(input)):
        # Push the current element of the input array onto the stack
        stack.append(input[i])
        # Check if the top of the stack matches the output array
        while stack and stack[-1] == output[j]:
            stack.pop()
            j += 1  # Move to the next element in output
    # If j has reached the length of output, then output is a valid permutation
    return j == len(output)

# Example usage
input = [1, 2, 3]
output = [2, 1, 3]

if is_stack_permutation(input, output):
    print("Yes, it is a stack permutation")
else:
    print("No, it is not a stack permutation")
