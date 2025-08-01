"""
Pattern Rules for Behavior-Based Script Detector

This module defines the patterns and rules used to detect suspicious
behavior in Python scripts using AST analysis.
"""

import ast
import asttokens
from typing import List, Dict, Any, Tuple
import re


class PatternRules:
    """Defines patterns and rules for detecting suspicious behavior in Python code."""
    
    def __init__(self):
        """Initialize pattern rules with predefined suspicious patterns."""
        self.suspicious_patterns = {
            # Dangerous function calls
            'exec_usage': {
                'pattern': ['exec', 'eval'],
                'score': 25,
                'description': 'Dynamic code execution using exec() or eval()',
                'severity': 'HIGH'
            },
            
            'subprocess_usage': {
                'pattern': ['subprocess', 'os.system', 'os.popen'],
                'score': 20,
                'description': 'Shell command execution',
                'severity': 'HIGH'
            },
            
            'pickle_usage': {
                'pattern': ['pickle.load', 'pickle.loads', 'marshal.load', 'marshal.loads'],
                'score': 15,
                'description': 'Unsafe deserialization using pickle or marshal',
                'severity': 'MEDIUM'
            },
            
            # File system operations
            'sensitive_file_access': {
                'pattern': [
                    r'/etc/', r'~/.ssh', r'/root', r'/var/log',
                    r'C:\\Windows', r'C:\\System32'
                ],
                'score': 18,
                'description': 'Access to sensitive system directories',
                'severity': 'HIGH'
            },
            
            'file_deletion': {
                'pattern': ['os.remove', 'os.unlink', 'shutil.rmtree'],
                'score': 12,
                'description': 'File deletion operations',
                'severity': 'MEDIUM'
            },
            
            # Network operations
            'network_download': {
                'pattern': ['urllib.request', 'requests.get', 'wget', 'curl'],
                'score': 15,
                'description': 'Network download operations',
                'severity': 'MEDIUM'
            },
            
            'socket_usage': {
                'pattern': ['socket.socket', 'socket.connect'],
                'score': 10,
                'description': 'Raw socket operations',
                'severity': 'MEDIUM'
            },
            
            # Encoding/Decoding
            'encoding_operations': {
                'pattern': ['base64', 'marshal', 'zlib', 'bz2'],
                'score': 8,
                'description': 'Encoding/decoding operations',
                'severity': 'LOW'
            },
            
            # Process manipulation
            'process_creation': {
                'pattern': ['os.fork', 'multiprocessing.Process', 'threading.Thread'],
                'score': 12,
                'description': 'Process or thread creation',
                'severity': 'MEDIUM'
            },
            
            # Import suspicious modules
            'suspicious_imports': {
                'pattern': ['ctypes', 'win32api', 'win32com', 'pywin32'],
                'score': 10,
                'description': 'Suspicious module imports',
                'severity': 'MEDIUM'
            },
            
            # Obfuscation techniques
            'obfuscated_code': {
                'pattern': [r'\\x[0-9a-fA-F]{2}', r'\\u[0-9a-fA-F]{4}'],
                'score': 15,
                'description': 'Potentially obfuscated code',
                'severity': 'MEDIUM'
            },
            
            # Environment manipulation
            'env_manipulation': {
                'pattern': ['os.environ', 'os.putenv', 'os.setenv'],
                'score': 8,
                'description': 'Environment variable manipulation',
                'severity': 'LOW'
            },
            
            # Registry access (Windows)
            'registry_access': {
                'pattern': ['winreg', 'regedit'],
                'score': 15,
                'description': 'Windows registry access',
                'severity': 'MEDIUM'
            }
        }
    
    def analyze_ast(self, tree: ast.AST, atok: asttokens.ASTTokens) -> List[Dict[str, Any]]:
        """
        Analyze AST tree for suspicious patterns.
        
        Args:
            tree: AST tree of the Python code
            atok: AST tokens for line number information
            
        Returns:
            List of detected suspicious patterns with details
        """
        findings = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                findings.extend(self._check_function_calls(node, atok))
            elif isinstance(node, ast.Import):
                findings.extend(self._check_imports(node, atok))
            elif isinstance(node, ast.ImportFrom):
                findings.extend(self._check_imports(node, atok))
            elif isinstance(node, ast.Str):
                findings.extend(self._check_string_literals(node, atok))
            elif isinstance(node, ast.Constant):
                findings.extend(self._check_string_literals(node, atok))
        
        return findings
    
    def _check_function_calls(self, node: ast.Call, atok: asttokens.ASTTokens) -> List[Dict[str, Any]]:
        """Check function calls for suspicious patterns."""
        findings = []
        
        # Get function name
        func_name = self._get_function_name(node.func)
        if not func_name:
            return findings
        
        # Check against patterns
        for pattern_name, pattern_info in self.suspicious_patterns.items():
            if pattern_name in ['sensitive_file_access', 'obfuscated_code']:
                continue  # These are checked in string literals
                
            for pattern in pattern_info['pattern']:
                if func_name.startswith(pattern) or func_name == pattern:
                    try:
                        line = atok.get_text_positions(node, padded=True)[0][0]
                    except (TypeError, IndexError):
                        line = getattr(node, 'lineno', 0)
                    
                    findings.append({
                        'line': line,
                        'pattern': pattern_name,
                        'description': pattern_info['description'],
                        'score': pattern_info['score'],
                        'severity': pattern_info['severity'],
                        'code': atok.get_text(node)
                    })
        
        return findings
    
-from typing import List, Dict, Any
+from typing import List, Dict, Any, Union

     def _check_imports(self, node: Union[ast.Import, ast.ImportFrom], atok: asttokens.ASTTokens) -> List[Dict[str, Any]]:
        """Check import statements for suspicious modules."""
        findings = []
        
        if isinstance(node, ast.Import):
            for alias in node.names:
                module_name = alias.name
                findings.extend(self._check_module_name(module_name, node, atok))
        elif isinstance(node, ast.ImportFrom):
            module_name = node.module or ""
            findings.extend(self._check_module_name(module_name, node, atok))
        
        return findings
    
    def _check_module_name(self, module_name: str, node: ast.AST, atok: asttokens.ASTTokens) -> List[Dict[str, Any]]:
        """Check if a module name matches suspicious patterns."""
        findings = []
        
        for pattern_name, pattern_info in self.suspicious_patterns.items():
            if pattern_name == 'suspicious_imports':
                for pattern in pattern_info['pattern']:
                    if pattern in module_name:
                        try:
                            line = atok.get_text_positions(node, padded=True)[0][0]
                        except (TypeError, IndexError):
                            line = getattr(node, 'lineno', 0)
                        
                        findings.append({
                            'line': line,
                            'pattern': pattern_name,
                            'description': pattern_info['description'],
                            'score': pattern_info['score'],
                            'severity': pattern_info['severity'],
                            'code': atok.get_text(node)
                        })
        
        return findings
    
    def _check_string_literals(self, node: ast.AST, atok: asttokens.ASTTokens) -> List[Dict[str, Any]]:
        """Check string literals for suspicious patterns."""
        findings = []
        
        # Get string value
        if isinstance(node, ast.Str):
            string_value = node.s
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            string_value = node.value
        else:
            return findings
        
        # Check for sensitive file paths
        for pattern_name, pattern_info in self.suspicious_patterns.items():
            if pattern_name == 'sensitive_file_access':
                for pattern in pattern_info['pattern']:
                    if re.search(pattern, string_value, re.IGNORECASE):
                        try:
                            line = atok.get_text_positions(node, padded=True)[0][0]
                        except (TypeError, IndexError):
                            line = getattr(node, 'lineno', 0)
                        
                        findings.append({
                            'line': line,
                            'pattern': pattern_name,
                            'description': pattern_info['description'],
                            'score': pattern_info['score'],
                            'severity': pattern_info['severity'],
                            'code': atok.get_text(node)
                        })
            
            elif pattern_name == 'obfuscated_code':
                for pattern in pattern_info['pattern']:
                    if re.search(pattern, string_value):
                        try:
                            line = atok.get_text_positions(node, padded=True)[0][0]
                        except (TypeError, IndexError):
                            line = getattr(node, 'lineno', 0)
                        
                        findings.append({
                            'line': line,
                            'pattern': pattern_name,
                            'description': pattern_info['description'],
                            'score': pattern_info['score'],
                            'severity': pattern_info['severity'],
                            'code': atok.get_text(node)
                        })
        
        return findings
    
    def _get_function_name(self, func_node: ast.AST) -> str:
        """Extract function name from AST node."""
        if isinstance(func_node, ast.Name):
            return func_node.id
        elif isinstance(func_node, ast.Attribute):
            return f"{self._get_function_name(func_node.value)}.{func_node.attr}"
        elif isinstance(func_node, ast.Call):
            return self._get_function_name(func_node.func)
        else:
            return "" 