"""
Test cases for Behavior-Based Script Detector

This module contains comprehensive tests for all components
of the behavior-based script detector.
"""

import pytest
import ast
import asttokens
import tempfile
import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from analyzer.pattern_rules import PatternRules
from analyzer.score_calculator import ScoreCalculator, RiskLevel
from analyzer.report_generator import ReportGenerator
from utils.file_loader import FileLoader


class TestPatternRules:
    """Test cases for PatternRules class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pattern_rules = PatternRules()
    
    def test_init(self):
        """Test PatternRules initialization."""
        assert self.pattern_rules.suspicious_patterns is not None
        assert len(self.pattern_rules.suspicious_patterns) > 0
    
    def test_exec_pattern_detection(self):
        """Test detection of exec usage."""
        code = "exec('print(\"Hello\")')"
        tree = ast.parse(code)
        atok = asttokens.ASTTokens(code, tree=tree)
        
        findings = self.pattern_rules.analyze_ast(tree, atok)
        
        assert len(findings) > 0
        exec_findings = [f for f in findings if f['pattern'] == 'exec_usage']
        assert len(exec_findings) > 0
    
    def test_eval_pattern_detection(self):
        """Test detection of eval usage."""
        code = "result = eval('2 + 2')"
        tree = ast.parse(code)
        atok = asttokens.ASTTokens(code, tree=tree)
        
        findings = self.pattern_rules.analyze_ast(tree, atok)
        
        assert len(findings) > 0
        eval_findings = [f for f in findings if f['pattern'] == 'exec_usage']
        assert len(eval_findings) > 0
    
    def test_subprocess_pattern_detection(self):
        """Test detection of subprocess usage."""
        code = "subprocess.run(['ls', '-la'])"
        tree = ast.parse(code)
        atok = asttokens.ASTTokens(code, tree=tree)
        
        findings = self.pattern_rules.analyze_ast(tree, atok)
        
        assert len(findings) > 0
        subprocess_findings = [f for f in findings if f['pattern'] == 'subprocess_usage']
        assert len(subprocess_findings) > 0
    
    def test_os_system_pattern_detection(self):
        """Test detection of os.system usage."""
        code = "os.system('echo hello')"
        tree = ast.parse(code)
        atok = asttokens.ASTTokens(code, tree=tree)
        
        findings = self.pattern_rules.analyze_ast(tree, atok)
        
        assert len(findings) > 0
        system_findings = [f for f in findings if f['pattern'] == 'subprocess_usage']
        assert len(system_findings) > 0
    
    def test_pickle_pattern_detection(self):
        """Test detection of pickle usage."""
        code = "pickle.loads(data)"
        tree = ast.parse(code)
        atok = asttokens.ASTTokens(code, tree=tree)
        
        findings = self.pattern_rules.analyze_ast(tree, atok)
        
        assert len(findings) > 0
        pickle_findings = [f for f in findings if f['pattern'] == 'pickle_usage']
        assert len(pickle_findings) > 0
    
    def test_sensitive_file_access_detection(self):
        """Test detection of sensitive file access."""
        code = 'open("/etc/passwd", "r")'
        tree = ast.parse(code)
        atok = asttokens.ASTTokens(code, tree=tree)
        
        findings = self.pattern_rules.analyze_ast(tree, atok)
        
        assert len(findings) > 0
        file_findings = [f for f in findings if f['pattern'] == 'sensitive_file_access']
        assert len(file_findings) > 0
    
    def test_network_download_detection(self):
        """Test detection of network download operations."""
        code = "urllib.request.urlretrieve('http://example.com/file.txt', 'file.txt')"
        tree = ast.parse(code)
        atok = asttokens.ASTTokens(code, tree=tree)
        
        findings = self.pattern_rules.analyze_ast(tree, atok)
        
        assert len(findings) > 0
        network_findings = [f for f in findings if f['pattern'] == 'network_download']
        assert len(network_findings) > 0
    
    def test_suspicious_imports_detection(self):
        """Test detection of suspicious imports."""
        code = "import ctypes"
        tree = ast.parse(code)
        atok = asttokens.ASTTokens(code, tree=tree)
        
        findings = self.pattern_rules.analyze_ast(tree, atok)
        
        assert len(findings) > 0
        import_findings = [f for f in findings if f['pattern'] == 'suspicious_imports']
        assert len(import_findings) > 0
    
    def test_obfuscated_code_detection(self):
        """Test detection of obfuscated code."""
        code = 'obfuscated = "\\x48\\x65\\x6c\\x6c\\x6f"'
        tree = ast.parse(code)
        atok = asttokens.ASTTokens(code, tree=tree)
        
        findings = self.pattern_rules.analyze_ast(tree, atok)
        
        assert len(findings) > 0
        obfuscated_findings = [f for f in findings if f['pattern'] == 'obfuscated_code']
        assert len(obfuscated_findings) > 0
    
    def test_safe_code_no_findings(self):
        """Test that safe code produces no findings."""
        code = """
def safe_function():
    x = 1 + 2
    print("Hello")
    return x
"""
        tree = ast.parse(code)
        atok = asttokens.ASTTokens(code, tree=tree)
        
        findings = self.pattern_rules.analyze_ast(tree, atok)
        
        assert len(findings) == 0
    
    def test_multiple_patterns_detection(self):
        """Test detection of multiple patterns in same code."""
        code = """
import subprocess
import os
exec('print("Hello")')
os.system('echo test')
"""
        tree = ast.parse(code)
        atok = asttokens.ASTTokens(code, tree=tree)
        
        findings = self.pattern_rules.analyze_ast(tree, atok)
        
        assert len(findings) >= 3  # At least 3 patterns should be detected
        patterns = [f['pattern'] for f in findings]
        assert 'suspicious_imports' in patterns
        assert 'exec_usage' in patterns
        assert 'subprocess_usage' in patterns


class TestScoreCalculator:
    """Test cases for ScoreCalculator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.calculator = ScoreCalculator()
    
    def test_init(self):
        """Test ScoreCalculator initialization."""
        assert self.calculator.risk_thresholds is not None
        assert self.calculator.severity_multipliers is not None
    
    def test_empty_findings(self):
        """Test score calculation with no findings."""
        findings = []
        result = self.calculator.calculate_risk_score(findings)
        
        assert result['risk_score'] == 0
        assert result['verdict'] == RiskLevel.LOW.value
        assert result['total_findings'] == 0
    
    def test_single_finding(self):
        """Test score calculation with single finding."""
        findings = [{
            'pattern': 'exec_usage',
            'score': 25,
            'severity': 'HIGH',
            'description': 'Test finding'
        }]
        
        result = self.calculator.calculate_risk_score(findings)
        
        assert result['risk_score'] > 0
        assert result['total_findings'] == 1
    
    def test_multiple_findings(self):
        """Test score calculation with multiple findings."""
        findings = [
            {
                'pattern': 'exec_usage',
                'score': 25,
                'severity': 'HIGH',
                'description': 'Test finding 1'
            },
            {
                'pattern': 'subprocess_usage',
                'score': 20,
                'severity': 'HIGH',
                'description': 'Test finding 2'
            }
        ]
        
        result = self.calculator.calculate_risk_score(findings)
        
        assert result['risk_score'] > 0
        assert result['total_findings'] == 2
    
    def test_risk_level_determination(self):
        """Test risk level determination."""
        # Test low risk
        findings = [{'score': 10, 'severity': 'LOW'}]
        result = self.calculator.calculate_risk_score(findings)
        assert result['verdict'] == RiskLevel.LOW.value
        
        # Test medium risk
        findings = [{'score': 50, 'severity': 'MEDIUM'}]
        result = self.calculator.calculate_risk_score(findings)
        assert result['verdict'] == RiskLevel.MEDIUM.value
        
        # Test high risk
        findings = [{'score': 85, 'severity': 'HIGH'}]
        result = self.calculator.calculate_risk_score(findings)
        assert result['verdict'] == RiskLevel.HIGH.value
    
    def test_severity_multipliers(self):
        """Test severity multiplier application."""
        findings = [
            {'score': 10, 'severity': 'LOW'},
            {'score': 10, 'severity': 'HIGH'}
        ]
        
        result = self.calculator.calculate_risk_score(findings)
        
        # HIGH severity should contribute more to the score
        assert result['risk_score'] > 20
    
    def test_score_capping(self):
        """Test that score is capped at 100."""
        findings = [
            {'score': 50, 'severity': 'HIGH'},
            {'score': 50, 'severity': 'HIGH'},
            {'score': 50, 'severity': 'HIGH'}
        ]
        
        result = self.calculator.calculate_risk_score(findings)
        
        assert result['risk_score'] <= 100
    
    def test_risk_assessment(self):
        """Test comprehensive risk assessment."""
        findings = [
            {
                'pattern': 'exec_usage',
                'score': 25,
                'severity': 'HIGH',
                'description': 'Test finding'
            }
        ]
        
        score = 30
        assessment = self.calculator.get_risk_assessment(score, findings)
        
        assert 'score' in assessment
        assert 'level' in assessment
        assert 'confidence' in assessment
        assert 'recommendations' in assessment
        assert 'summary' in assessment


class TestReportGenerator:
    """Test cases for ReportGenerator class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.generator = ReportGenerator()
    
    def test_init(self):
        """Test ReportGenerator initialization."""
        assert self.generator.console is not None
    
    def test_json_report_generation(self):
        """Test JSON report generation."""
        filename = "test.py"
        analysis_result = {
            'risk_score': 75,
            'verdict': 'High Risk',
            'total_findings': 3,
            'severity_breakdown': {'HIGH': 2, 'MEDIUM': 1},
            'findings': [
                {
                    'line': 10,
                    'pattern': 'exec_usage',
                    'description': 'Test finding',
                    'score': 25,
                    'severity': 'HIGH'
                }
            ],
            'assessment': {
                'summary': 'Test summary',
                'recommendations': ['Test recommendation']
            }
        }
        
        json_report = self.generator.generate_json_report(filename, analysis_result)
        
        assert json_report is not None
        assert 'filename' in json_report
        assert 'risk_score' in json_report
        assert 'verdict' in json_report
    
    def test_markdown_report_generation(self):
        """Test Markdown report generation."""
        filename = "test.py"
        analysis_result = {
            'risk_score': 75,
            'verdict': 'High Risk',
            'findings': [
                {
                    'line': 10,
                    'pattern': 'exec_usage',
                    'description': 'Test finding',
                    'score': 25,
                    'severity': 'HIGH'
                }
            ],
            'assessment': {
                'summary': 'Test summary',
                'recommendations': ['Test recommendation']
            }
        }
        
        md_report = self.generator.generate_markdown_report(filename, analysis_result)
        
        assert md_report is not None
        assert 'Behavior Analysis Report' in md_report
        assert 'Risk Score' in md_report
        assert 'Test finding' in md_report
    
    def test_batch_report_generation(self):
        """Test batch report generation."""
        results = [
            {
                'filename': 'test1.py',
                'risk_score': 75,
                'verdict': 'High Risk',
                'total_findings': 2
            },
            {
                'filename': 'test2.py',
                'risk_score': 25,
                'verdict': 'Low Risk',
                'total_findings': 1
            }
        ]
        
        batch_report = self.generator.generate_batch_report(results)
        
        assert batch_report is not None
        assert 'total_files' in batch_report
        assert 'files_analyzed' in batch_report
        assert 'summary' in batch_report
    
    def test_report_saving(self):
        """Test report saving functionality."""
        with tempfile.TemporaryDirectory() as temp_dir:
            report_content = '{"test": "data"}'
            filename = "test.py"
            report_type = "json"
            
            report_path = self.generator.save_report(
                report_content, filename, report_type, temp_dir
            )
            
            assert os.path.exists(report_path)
            assert report_path.endswith('.json')
            
            # Check content
            with open(report_path, 'r') as f:
                content = f.read()
                assert content == report_content


class TestFileLoader:
    """Test cases for FileLoader class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.loader = FileLoader()
    
    def test_init(self):
        """Test FileLoader initialization."""
        assert self.loader.supported_extensions == {'.py'}
    
    def test_load_valid_file(self):
        """Test loading a valid Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('print("Hello, World!")')
            file_path = f.name
        
        try:
            content, tree, atok = self.loader.load_file(file_path)
            
            assert content == 'print("Hello, World!")'
            assert isinstance(tree, ast.AST)
            assert isinstance(atok, asttokens.ASTTokens)
        finally:
            os.unlink(file_path)
    
    def test_load_nonexistent_file(self):
        """Test loading a nonexistent file."""
        with pytest.raises(FileNotFoundError):
            self.loader.load_file('nonexistent.py')
    
    def test_load_invalid_file_type(self):
        """Test loading a non-Python file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('This is not Python code')
            file_path = f.name
        
        try:
            with pytest.raises(ValueError):
                self.loader.load_file(file_path)
        finally:
            os.unlink(file_path)
    
    def test_load_syntax_error_file(self):
        """Test loading a file with syntax errors."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('print("Hello" + )')  # Syntax error
            file_path = f.name
        
        try:
            with pytest.raises(SyntaxError):
                self.loader.load_file(file_path)
        finally:
            os.unlink(file_path)
    
    def test_validate_file(self):
        """Test file validation."""
        # Test valid file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('print("Hello")')
            file_path = f.name
        
        try:
            assert self.loader.validate_file(file_path) == True
        finally:
            os.unlink(file_path)
        
        # Test invalid file
        assert self.loader.validate_file('nonexistent.py') == False
    
    def test_get_file_info(self):
        """Test getting file information."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('print("Hello")')
            file_path = f.name
        
        try:
            info = self.loader.get_file_info(file_path)
            
            assert info['path'] == file_path
            assert info['name'] == os.path.basename(file_path)
            assert info['is_file'] == True
            assert info['is_directory'] == False
        finally:
            os.unlink(file_path)


class TestIntegration:
    """Integration tests for the complete system."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.pattern_rules = PatternRules()
        self.score_calculator = ScoreCalculator()
        self.report_generator = ReportGenerator()
        self.file_loader = FileLoader()
    
    def test_complete_analysis_workflow(self):
        """Test the complete analysis workflow."""
        # Create a test file with suspicious patterns
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
import subprocess
import os
exec("print(\'Hello\')")
os.system("echo test")
''')
            file_path = f.name
        
        try:
            # Load and parse file
            content, tree, atok = self.file_loader.load_file(file_path)
            
            # Analyze for patterns
            findings = self.pattern_rules.analyze_ast(tree, atok)
            
            # Calculate risk score
            score_result = self.score_calculator.calculate_risk_score(findings)
            
            # Generate report
            analysis_result = {
                'filename': file_path,
                'findings': findings,
                'risk_score': score_result['risk_score'],
                'verdict': score_result['verdict'],
                'total_findings': score_result['total_findings'],
                'severity_breakdown': score_result['severity_breakdown'],
                'assessment': self.score_calculator.get_risk_assessment(
                    score_result['risk_score'], findings
                )
            }
            
            # Verify results
            assert len(findings) > 0
            assert score_result['risk_score'] > 0
            assert 'High Risk' in score_result['verdict'] or 'Medium Risk' in score_result['verdict']
            
            # Test report generation
            json_report = self.report_generator.generate_json_report(file_path, analysis_result)
            assert json_report is not None
            
        finally:
            os.unlink(file_path)
    
    def test_safe_code_analysis(self):
        """Test analysis of safe code."""
        # Create a test file with safe code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write('''
def safe_function():
    x = 1 + 2
    print("Hello")
    return x
''')
            file_path = f.name
        
        try:
            # Load and parse file
            content, tree, atok = self.file_loader.load_file(file_path)
            
            # Analyze for patterns
            findings = self.pattern_rules.analyze_ast(tree, atok)
            
            # Calculate risk score
            score_result = self.score_calculator.calculate_risk_score(findings)
            
            # Verify results
            assert len(findings) == 0
            assert score_result['risk_score'] == 0
            assert score_result['verdict'] == RiskLevel.LOW.value
            
        finally:
            os.unlink(file_path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 