import ast

class TimeComplexityAnalyzer(ast.NodeVisitor):
    def __init__(self):
        self.complexity = 0
        self.loop_depth = 0

    def visit_For(self, node):
        self.loop_depth += 1
        self.complexity += 2 ** self.loop_depth
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_While(self, node):
        self.loop_depth += 1
        self.complexity += 2 ** self.loop_depth
        self.generic_visit(node)
        self.loop_depth -= 1

    def visit_FunctionDef(self, node):
        self.generic_visit(node)

    def visit_Call(self, node):
        self.generic_visit(node)

    def visit_If(self, node):
        self.generic_visit(node)

    def get_complexity(self):
        if self.complexity == 0:
            return "O(1) - Constant Time"
        elif self.complexity == 1:
            return "O(log n) - Logarithmic Time"
        elif self.complexity == 2:
            return "O(n) - Linear Time"
        elif self.complexity == 3:
            return "O(n log n) - Linearithmic Time"
        elif self.complexity == 4:
            return "O(n^2) - Quadratic Time"
        elif self.complexity == 5:
            return "O(n^3) - Cubic Time"
        elif self.complexity >= 6:
            return f"O(n^{self.complexity}) - Polynomial Time"
        return "O(2^n) - Exponential Time"

def analyze_code(code):
    try:
        tree = ast.parse(code)
        analyzer = TimeComplexityAnalyzer()
        analyzer.visit(tree)
        return analyzer.get_complexity()
    except SyntaxError as e:
        return f"Syntax Error: {e.msg} at line {e.lineno}, column {e.offset}"
    except Exception as e:
        return f"An unexpected error occurred: {str(e)}"

if __name__ == "__main__":
    print("Welcome to the Time Complexity Analyzer!")
    user_code = input("Please enter a piece of Python code:\n")
    complexity = analyze_code(user_code)
    print(f"Estimated time complexity: {complexity}")
