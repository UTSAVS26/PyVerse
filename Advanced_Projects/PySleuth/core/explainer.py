import ast
from core.ast_parser import ASTParser

def explain_branch(frame, line):
    try:
        # If the line is a control statement, append 'pass' to make it a valid block
        if line.strip().endswith(':'):
            line = line + ' pass'
        tree = ast.parse(line)
        node = tree.body[0]
        if isinstance(node, ast.If):
            # Safe evaluation for simple comparison operations only
            if isinstance(node.test, (ast.Compare, ast.BoolOp, ast.UnaryOp)):
                try:
                    cond = compile(ast.Expression(node.test), '<ast>', 'eval')
                    # Create restricted globals/locals for evaluation
                    safe_globals = {"__builtins__": {}}
                    safe_locals = {
                        k: v for k, v in frame.f_locals.items()
                        if isinstance(v, (int, float, str, bool, type(None)))
                    }
                    result = eval(cond, safe_globals, safe_locals)
                except (NameError, AttributeError, TypeError):
                    result = "Unable to evaluate safely"
            else:
                result = "Complex expression"
            return f"Entered 'if' because {ast.unparse(node.test) if hasattr(ast, 'unparse') else ast.dump(node.test)} → {result} (locals: {frame.f_locals})"
        elif isinstance(node, ast.While):
            cond = compile(ast.Expression(node.test), '<ast>', 'eval')
            result = eval(cond, frame.f_globals, frame.f_locals)
            return f"Entered 'while' because {ast.unparse(node.test) if hasattr(ast, 'unparse') else ast.dump(node.test)} → {result} (locals: {frame.f_locals})"
        elif isinstance(node, ast.FunctionDef):
            return f"Entered function '{node.name}' with args {frame.f_locals}"
    except Exception as e:
        return f"Could not explain: {e}"
    return ""