class Node:
    def __init__(self, val):
        self.value = val
        self.left = None
        self.right = None

def is_operator(c):
    return c in ['+', '-', '*', '/']

def construct_expression_tree(postfix):
    stack = []
    for ch in postfix:
        if is_operator(ch):
            node = Node(ch)
            # Pop two operands for the operator
            node.right = stack.pop()
            node.left = stack.pop()
            stack.append(node)
        else:
            # Operand, push to stack
            stack.append(Node(ch))

    # The root of the expression tree
    return stack[-1]
  
def inorder_traversal(node):
    if not node:
        return

    # Print left operand with parentheses for correct infix form
    if is_operator(node.value):
        print("(", end="")
    inorder_traversal(node.left)
    print(node.value, end="")
    inorder_traversal(node.right)
    if is_operator(node.value):
        print(")", end="")

if __name__ == "__main__":
    postfix = input("Enter postfix expression: ")
    root = construct_expression_tree(postfix)
    print("Infix expression: ", end="")
    inorder_traversal(root)
    print()
