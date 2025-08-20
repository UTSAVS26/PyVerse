"""
Tests for Diff Viewer module.
"""

import pytest
from pypolish.diff_viewer import DiffViewer


class TestDiffViewer:
    """Test cases for DiffViewer class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.diff_viewer = DiffViewer()
    
    def test_show_diff_identical_codes(self):
        """Test showing diff for identical codes."""
        code = """def test():
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_diff(code, code)
    
    def test_show_diff_different_codes(self):
        """Test showing diff for different codes."""
        original_code = """def test():
    return True
"""
        cleaned_code = """def test() -> bool:
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_diff(original_code, cleaned_code)
    
    def test_show_side_by_side(self):
        """Test showing side by side comparison."""
        original_code = """def test():
    return True
"""
        cleaned_code = """def test() -> bool:
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_side_by_side(original_code, cleaned_code)
    
    def test_show_side_by_side_different_lengths(self):
        """Test showing side by side with different line counts."""
        original_code = """def test():
    return True
"""
        cleaned_code = """def test() -> bool:
    \"\"\"Test function.\"\"\"
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_side_by_side(original_code, cleaned_code)
    
    def test_show_analysis_summary_no_issues(self):
        """Test showing analysis summary with no issues."""
        analysis_results = {
            'issues': [],
            'suggestions': []
        }
        
        # Should not raise any exceptions
        self.diff_viewer.show_analysis_summary(analysis_results)
    
    def test_show_analysis_summary_with_issues(self):
        """Test showing analysis summary with issues."""
        analysis_results = {
            'issues': [
                {
                    'type': 'long_function',
                    'message': 'Function is too long',
                    'line': 5,
                    'severity': 'warning'
                }
            ],
            'suggestions': [
                {
                    'type': 'missing_docstring',
                    'message': 'Add docstring',
                    'line': 3,
                    'severity': 'info'
                }
            ]
        }
        
        # Should not raise any exceptions
        self.diff_viewer.show_analysis_summary(analysis_results)
    
    def test_show_syntax_highlighted(self):
        """Test showing syntax highlighted code."""
        code = """def test():
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_syntax_highlighted(code)
    
    def test_show_syntax_highlighted_with_title(self):
        """Test showing syntax highlighted code with title."""
        code = """def test():
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_syntax_highlighted(code, title="Test Code")
    
    def test_show_before_after_highlighted(self):
        """Test showing before and after code with highlighting."""
        original_code = """def test():
    return True
"""
        cleaned_code = """def test() -> bool:
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_before_after_highlighted(original_code, cleaned_code)
    
    def test_generate_diff(self):
        """Test generating diff between codes."""
        original_code = """def test():
    return True
"""
        cleaned_code = """def test() -> bool:
    return True
"""
        
        diff_lines = self.diff_viewer._generate_diff(original_code, cleaned_code)
        
        assert isinstance(diff_lines, list)
        assert len(diff_lines) > 0
        # Should contain diff markers
        assert any(line.startswith('---') for line in diff_lines)
        assert any(line.startswith('+++') for line in diff_lines)
    
    def test_generate_diff_identical(self):
        """Test generating diff for identical codes."""
        code = """def test():
    return True
"""
        
        diff_lines = self.diff_viewer._generate_diff(code, code)
        
        assert isinstance(diff_lines, list)
        # Should be empty for identical codes
        assert len(diff_lines) == 0
    
    def test_show_statistics(self):
        """Test showing statistics."""
        original_code = """def test():
    return True
"""
        cleaned_code = """def test() -> bool:
    \"\"\"Test function.\"\"\"
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_statistics(original_code, cleaned_code)
    
    def test_show_statistics_identical(self):
        """Test showing statistics for identical codes."""
        code = """def test():
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_statistics(code, code)
    
    def test_show_improvements_list_no_improvements(self):
        """Test showing improvements list with no improvements."""
        analysis_results = {
            'issues': [],
            'suggestions': []
        }
        
        # Should not raise any exceptions
        self.diff_viewer.show_improvements_list(analysis_results)
    
    def test_show_improvements_list_with_improvements(self):
        """Test showing improvements list with improvements."""
        analysis_results = {
            'issues': [
                {
                    'type': 'long_function',
                    'message': 'Function is too long',
                    'line': 5,
                    'severity': 'warning'
                }
            ],
            'suggestions': [
                {
                    'type': 'missing_type_hint',
                    'message': 'Add type hint',
                    'line': 3,
                    'severity': 'info'
                },
                {
                    'type': 'missing_docstring',
                    'message': 'Add docstring',
                    'line': 3,
                    'severity': 'info'
                }
            ]
        }
        
        # Should not raise any exceptions
        self.diff_viewer.show_improvements_list(analysis_results)
    
    def test_show_diff_empty_codes(self):
        """Test showing diff for empty codes."""
        # Should not raise any exceptions
        self.diff_viewer.show_diff("", "")
    
    def test_show_diff_one_empty(self):
        """Test showing diff when one code is empty."""
        code = """def test():
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_diff(code, "")
        self.diff_viewer.show_diff("", code)
    
    def test_show_side_by_side_empty_codes(self):
        """Test showing side by side for empty codes."""
        # Should not raise any exceptions
        self.diff_viewer.show_side_by_side("", "")
    
    def test_show_side_by_side_one_empty(self):
        """Test showing side by side when one code is empty."""
        code = """def test():
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_side_by_side(code, "")
        self.diff_viewer.show_side_by_side("", code)
    
    def test_show_syntax_highlighted_empty(self):
        """Test showing syntax highlighted empty code."""
        # Should not raise any exceptions
        self.diff_viewer.show_syntax_highlighted("")
    
    def test_show_before_after_highlighted_empty(self):
        """Test showing before and after with empty codes."""
        # Should not raise any exceptions
        self.diff_viewer.show_before_after_highlighted("", "")
    
    def test_show_statistics_empty_codes(self):
        """Test showing statistics for empty codes."""
        # Should not raise any exceptions
        self.diff_viewer.show_statistics("", "")
    
    def test_show_analysis_summary_empty(self):
        """Test showing analysis summary with empty results."""
        analysis_results = {}
        
        # Should not raise any exceptions
        self.diff_viewer.show_analysis_summary(analysis_results)
    
    def test_show_improvements_list_empty(self):
        """Test showing improvements list with empty results."""
        analysis_results = {}
        
        # Should not raise any exceptions
        self.diff_viewer.show_improvements_list(analysis_results)
    
    def test_show_diff_with_title(self):
        """Test showing diff with custom title."""
        original_code = """def test():
    return True
"""
        cleaned_code = """def test() -> bool:
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_diff(original_code, cleaned_code, title="Custom Title")
    
    def test_show_side_by_side_with_title(self):
        """Test showing side by side with custom title."""
        original_code = """def test():
    return True
"""
        cleaned_code = """def test() -> bool:
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_side_by_side(original_code, cleaned_code, title="Custom Title")
    
    def test_show_syntax_highlighted_different_language(self):
        """Test showing syntax highlighted code with different language."""
        code = """def test():
    return True
"""
        
        # Should not raise any exceptions
        self.diff_viewer.show_syntax_highlighted(code, language="python")
    
    def test_show_analysis_summary_multiple_severities(self):
        """Test showing analysis summary with multiple severity levels."""
        analysis_results = {
            'issues': [
                {
                    'type': 'long_function',
                    'message': 'Function is too long',
                    'line': 5,
                    'severity': 'warning'
                },
                {
                    'type': 'infinite_loop',
                    'message': 'Potential infinite loop',
                    'line': 10,
                    'severity': 'error'
                }
            ],
            'suggestions': [
                {
                    'type': 'missing_docstring',
                    'message': 'Add docstring',
                    'line': 3,
                    'severity': 'info'
                }
            ]
        }
        
        # Should not raise any exceptions
        self.diff_viewer.show_analysis_summary(analysis_results)
