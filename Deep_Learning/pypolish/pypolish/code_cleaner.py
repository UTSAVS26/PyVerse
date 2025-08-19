"""
Main Code Cleaner for PyPolish

Orchestrates the entire code cleaning process using AST analysis, formatting, and improvements.
"""

import ast
from typing import Dict, Any, Optional, Tuple
from .ast_analyzer import ASTAnalyzer
from .formatter import CodeFormatter
from .diff_viewer import DiffViewer


class CodeCleaner:
    """Main class that orchestrates the code cleaning process."""
    
    def __init__(self, line_length: int = 88):
        self.analyzer = ASTAnalyzer()
        self.formatter = CodeFormatter(line_length=line_length)
        self.diff_viewer = DiffViewer()
    
    def clean_code(self, code: str, show_analysis: bool = True, show_diff: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Clean the given Python code and return the cleaned version along with analysis results.
        
        Args:
            code: The original Python code as a string
            show_analysis: Whether to display analysis results
            show_diff: Whether to display diff between original and cleaned code
            
        Returns:
            Tuple of (cleaned_code, analysis_results)
        """
        # Step 1: Analyze the code
        analysis_results = self.analyzer.analyze(code)
        
        if 'error' in analysis_results:
            print(f"Error analyzing code: {analysis_results['error']}")
            return code, analysis_results
        
        # Step 2: Apply improvements
        cleaned_code = self._apply_improvements(code, analysis_results)
        
        # Step 3: Format the code
        cleaned_code = self.formatter.format_code(cleaned_code)
        
        # Step 4: Display results if requested
        if show_analysis:
            self.diff_viewer.show_analysis_summary(analysis_results)
        
        if show_diff and cleaned_code != code:
            self.diff_viewer.show_diff(code, cleaned_code)
            self.diff_viewer.show_statistics(code, cleaned_code)
            self.diff_viewer.show_improvements_list(analysis_results)
        
        return cleaned_code, analysis_results
    
    def _apply_improvements(self, code: str, analysis_results: Dict[str, Any]) -> str:
        """Apply various improvements to the code based on analysis results."""
        improved_code = code
        
        # Remove unused imports
        unused_imports = analysis_results.get('used_names', set()) - analysis_results.get('imported_names', set())
        if unused_imports:
            improved_code = self.formatter.remove_unused_imports(improved_code, list(unused_imports))
        
        # Add type hints
        improved_code = self.formatter.add_type_hints(improved_code)
        
        # Add docstrings
        improved_code = self.formatter.add_docstrings(improved_code)
        
        # Convert simple if/else to ternary expressions
        improved_code = self.formatter.convert_to_ternary(improved_code)
        
        # Apply specific improvements based on analysis
        improved_code = self._apply_specific_improvements(improved_code, analysis_results)
        
        return improved_code
    
    def _apply_specific_improvements(self, code: str, analysis_results: Dict[str, Any]) -> str:
        """Apply specific improvements based on analysis results."""
        # This is a simplified version - in a real implementation, you'd have more sophisticated
        # AST transformations to apply specific improvements
        
        # For now, we'll just return the code as-is
        # In a full implementation, you'd parse the AST, make specific transformations,
        # and then unparse back to code
        return code
    
    def clean_file(self, file_path: str, output_path: Optional[str] = None, 
                   show_analysis: bool = True, show_diff: bool = True) -> Tuple[str, Dict[str, Any]]:
        """
        Clean a Python file and optionally save the result.
        
        Args:
            file_path: Path to the input Python file
            output_path: Optional path to save the cleaned code
            show_analysis: Whether to display analysis results
            show_diff: Whether to display diff between original and cleaned code
            
        Returns:
            Tuple of (cleaned_code, analysis_results)
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                original_code = f.read()
            
            cleaned_code, analysis_results = self.clean_code(original_code, show_analysis, show_diff)
            
            if output_path:
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(cleaned_code)
                print(f"Cleaned code saved to: {output_path}")
            
            return cleaned_code, analysis_results
            
        except FileNotFoundError:
            print(f"Error: File '{file_path}' not found.")
            return "", {'error': f"File '{file_path}' not found."}
        except Exception as e:
            print(f"Error processing file: {e}")
            return "", {'error': str(e)}
    
    def get_cleaning_report(self, original_code: str, cleaned_code: str, 
                          analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive cleaning report."""
        original_lines = len(original_code.split('\n'))
        cleaned_lines = len(cleaned_code.split('\n'))
        original_chars = len(original_code)
        cleaned_chars = len(cleaned_code)
        
        issues_count = len(analysis_results.get('issues', []))
        suggestions_count = len(analysis_results.get('suggestions', []))
        
        return {
            'original_lines': original_lines,
            'cleaned_lines': cleaned_lines,
            'original_chars': original_chars,
            'cleaned_chars': cleaned_chars,
            'line_difference': cleaned_lines - original_lines,
            'char_difference': cleaned_chars - original_chars,
            'issues_found': issues_count,
            'suggestions_made': suggestions_count,
            'improvement_percentage': ((cleaned_chars - original_chars) / original_chars * 100) if original_chars > 0 else 0,
            'has_changes': original_code != cleaned_code
        }
    
    def validate_code(self, code: str) -> bool:
        """Validate that the code is syntactically correct Python."""
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def get_code_metrics(self, code: str) -> Dict[str, Any]:
        """Get various metrics about the code."""
        analysis_results = self.analyzer.analyze(code)
        
        if 'error' in analysis_results:
            return {'error': analysis_results['error']}
        
        return {
            'function_count': analysis_results.get('function_count', 0),
            'class_count': analysis_results.get('class_count', 0),
            'issues_count': len(analysis_results.get('issues', [])),
            'suggestions_count': len(analysis_results.get('suggestions', [])),
            'unused_imports': len(self.analyzer.get_unused_imports()),
            'undefined_names': len(self.analyzer.get_undefined_names()),
            'total_lines': len(code.split('\n')),
            'total_chars': len(code)
        }
