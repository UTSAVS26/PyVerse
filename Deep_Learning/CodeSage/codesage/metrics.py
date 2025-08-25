"""
Complexity Metrics Module

Provides AI-enhanced code complexity analysis with machine learning-based
threshold detection and intelligent metric calculations.
"""

import ast
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


@dataclass
class FunctionMetrics:
    """Container for function-level metrics."""
    name: str
    cyclomatic_complexity: int
    lines_of_code: int
    nesting_depth: int
    parameters: int
    return_statements: int
    comments_ratio: float
    maintainability_index: float
    ai_risk_score: float


@dataclass
class FileMetrics:
    """Container for file-level metrics."""
    filename: str
    functions: List[FunctionMetrics]
    total_lines: int
    total_functions: int
    average_complexity: float
    maintainability_index: float
    ai_anomaly_score: float


class ComplexityMetrics:
    """AI-enhanced complexity metrics calculator."""
    
    def __init__(self):
        self.complexity_thresholds = {
            'low': 5,
            'medium': 10,
            'high': 15,
            'critical': 25
        }
        self.ml_model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def calculate_cyclomatic_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity using AST analysis."""
        complexity = 1  # Base complexity
        
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor)):
                complexity += 1
            elif isinstance(child, ast.ExceptHandler):
                complexity += 1
            elif isinstance(child, ast.With):
                complexity += 1
            elif isinstance(child, ast.AsyncWith):
                complexity += 1
            elif isinstance(child, ast.Assert):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
                
        return complexity
    
    def calculate_nesting_depth(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth of code blocks."""
        max_depth = 0
        current_depth = 0
        
        def visit_node(n, depth):
            nonlocal max_depth
            max_depth = max(max_depth, depth)
            
            for child in ast.iter_child_nodes(n):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.Try, ast.With)):
                    visit_node(child, depth + 1)
                else:
                    visit_node(child, depth)
        
        visit_node(node, 0)
        return max_depth
    
    def calculate_maintainability_index(self, metrics: FunctionMetrics) -> float:
        """Calculate maintainability index using AI-enhanced formula."""
        # Traditional MI calculation
        halstead_volume = metrics.lines_of_code * math.log2(metrics.parameters + 1)
        cyclomatic_complexity = metrics.cyclomatic_complexity
        
        # AI-enhanced factors
        complexity_penalty = self._calculate_complexity_penalty(metrics)
        nesting_penalty = self._calculate_nesting_penalty(metrics.nesting_depth)
        comment_bonus = self._calculate_comment_bonus(metrics.comments_ratio)
        
        # Base MI calculation (171 - 5.2*ln(HV) - 0.23*CC - 16.2*ln(LOC))
        base_mi = 171 - 5.2 * math.log(halstead_volume + 1) - 0.23 * cyclomatic_complexity - 16.2 * math.log(metrics.lines_of_code + 1)
        
        # Apply AI enhancements
        enhanced_mi = base_mi - complexity_penalty - nesting_penalty + comment_bonus
        
        return max(0, min(100, enhanced_mi))
    
    def _calculate_complexity_penalty(self, metrics: FunctionMetrics) -> float:
        """AI-based complexity penalty calculation."""
        if metrics.cyclomatic_complexity <= self.complexity_thresholds['low']:
            return 0
        elif metrics.cyclomatic_complexity <= self.complexity_thresholds['medium']:
            return 5
        elif metrics.cyclomatic_complexity <= self.complexity_thresholds['high']:
            return 15
        else:
            return 25 + (metrics.cyclomatic_complexity - self.complexity_thresholds['critical']) * 2
    
    def _calculate_nesting_penalty(self, nesting_depth: int) -> float:
        """Calculate penalty for deep nesting."""
        if nesting_depth <= 3:
            return 0
        elif nesting_depth <= 5:
            return 10
        else:
            return 20 + (nesting_depth - 5) * 5
    
    def _calculate_comment_bonus(self, comments_ratio: float) -> float:
        """Calculate bonus for good documentation."""
        if comments_ratio >= 0.3:
            return 10
        elif comments_ratio >= 0.2:
            return 5
        elif comments_ratio >= 0.1:
            return 2
        else:
            return 0
    
    def train_anomaly_detector(self, metrics_data: List[Dict]) -> None:
        """Train ML model to detect anomalous code patterns."""
        if not metrics_data:
            return
            
        # Extract features for ML model
        features = []
        for data in metrics_data:
            feature_vector = [
                data.get('cyclomatic_complexity', 0),
                data.get('lines_of_code', 0),
                data.get('nesting_depth', 0),
                data.get('parameters', 0),
                data.get('comments_ratio', 0)
            ]
            features.append(feature_vector)
        
        if len(features) > 1:
            features_array = np.array(features)
            self.scaler.fit(features_array)
            scaled_features = self.scaler.transform(features_array)
            self.ml_model.fit(scaled_features)
            self.is_trained = True
    
    def calculate_ai_risk_score(self, metrics: FunctionMetrics) -> float:
        """Calculate AI-based risk score for function complexity."""
        if not self.is_trained:
            # Fallback to rule-based scoring
            return self._rule_based_risk_score(metrics)
        
        # Use trained ML model
        feature_vector = np.array([[
            metrics.cyclomatic_complexity,
            metrics.lines_of_code,
            metrics.nesting_depth,
            metrics.parameters,
            metrics.comments_ratio
        ]])
        
        scaled_features = self.scaler.transform(feature_vector)
        anomaly_score = self.ml_model.decision_function(scaled_features)[0]
        
        # Convert anomaly score to risk score (0-100)
        risk_score = max(0, min(100, (1 - anomaly_score) * 100))
        return risk_score
    
    def _rule_based_risk_score(self, metrics: FunctionMetrics) -> float:
        """Rule-based risk scoring when ML model is not trained."""
        risk_score = 0
        
        # Complexity risk
        if metrics.cyclomatic_complexity > self.complexity_thresholds['critical']:
            risk_score += 40
        elif metrics.cyclomatic_complexity > self.complexity_thresholds['high']:
            risk_score += 25
        elif metrics.cyclomatic_complexity > self.complexity_thresholds['medium']:
            risk_score += 15
        
        # Length risk
        if metrics.lines_of_code > 100:
            risk_score += 20
        elif metrics.lines_of_code > 50:
            risk_score += 10
        
        # Nesting risk
        if metrics.nesting_depth > 5:
            risk_score += 20
        elif metrics.nesting_depth > 3:
            risk_score += 10
        
        # Parameter risk
        if metrics.parameters > 7:
            risk_score += 10
        
        return min(100, risk_score)
    
    def analyze_function(self, func_node: ast.FunctionDef, source_lines: List[str]) -> FunctionMetrics:
        """Analyze a single function and return comprehensive metrics."""
        # Count lines of code
        start_line = func_node.lineno - 1
        end_line = func_node.end_lineno if hasattr(func_node, 'end_lineno') else len(source_lines)
        lines_of_code = end_line - start_line
        
        # Count comments in function
        comments = 0
        for i in range(start_line, min(end_line, len(source_lines))):
            line = source_lines[i].strip()
            if line.startswith('#') or line.startswith('"""') or line.startswith("'''"):
                comments += 1
        
        comments_ratio = comments / max(1, lines_of_code)
        
        # Calculate metrics
        cyclomatic_complexity = self.calculate_cyclomatic_complexity(func_node)
        nesting_depth = self.calculate_nesting_depth(func_node)
        # Count all parameter kinds (positional-only, positional/keyword, varargs, kw-only, kwargs)
        args = func_node.args
        parameters = (
            len(getattr(args, "posonlyargs", [])) +
            len(args.args) +
            (1 if args.vararg else 0) +
            len(args.kwonlyargs) +
            (1 if args.kwarg else 0)
        )
        # Count return statements
        return_statements = sum(1 for node in ast.walk(func_node) if isinstance(node, ast.Return))
        
        # Create metrics object
        metrics = FunctionMetrics(
            name=func_node.name,
            cyclomatic_complexity=cyclomatic_complexity,
            lines_of_code=lines_of_code,
            nesting_depth=nesting_depth,
            parameters=parameters,
            return_statements=return_statements,
            comments_ratio=comments_ratio,
            maintainability_index=0.0,  # Will be calculated below
            ai_risk_score=0.0  # Will be calculated below
        )
        
        # Calculate derived metrics
        metrics.maintainability_index = self.calculate_maintainability_index(metrics)
        metrics.ai_risk_score = self.calculate_ai_risk_score(metrics)
        
        return metrics
    
    def analyze_file(self, tree: ast.AST, filename: str, source_lines: List[str]) -> FileMetrics:
        """Analyze an entire file and return file-level metrics."""
        functions = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                func_metrics = self.analyze_function(node, source_lines)
                functions.append(func_metrics)
        
        if not functions:
            return FileMetrics(
                filename=filename,
                functions=[],
                total_lines=len(source_lines),
                total_functions=0,
                average_complexity=0.0,
                maintainability_index=100.0,
                ai_anomaly_score=0.0
            )
        
        # Calculate file-level metrics
        total_lines = len(source_lines)
        total_functions = len(functions)
        average_complexity = sum(f.cyclomatic_complexity for f in functions) / total_functions
        maintainability_index = sum(f.maintainability_index for f in functions) / total_functions
        
        # Calculate AI anomaly score for the file
        file_features = [
            average_complexity,
            total_lines,
            max(f.nesting_depth for f in functions),
            total_functions,
            sum(f.comments_ratio for f in functions) / total_functions
        ]
        
        if self.is_trained:
            file_features_array = np.array([file_features])
            scaled_features = self.scaler.transform(file_features_array)
            anomaly_score = self.ml_model.decision_function(scaled_features)[0]
            ai_anomaly_score = max(0, min(100, (1 - anomaly_score) * 100))
        else:
            ai_anomaly_score = self._calculate_file_anomaly_score(functions, total_lines)
        
        return FileMetrics(
            filename=filename,
            functions=functions,
            total_lines=total_lines,
            total_functions=total_functions,
            average_complexity=average_complexity,
            maintainability_index=maintainability_index,
            ai_anomaly_score=ai_anomaly_score
        )
    
    def _calculate_file_anomaly_score(self, functions: List[FunctionMetrics], total_lines: int) -> float:
        """Calculate file anomaly score using rule-based approach."""
        if not functions:
            return 0.0
        
        # Calculate various risk factors
        high_complexity_functions = sum(1 for f in functions if f.cyclomatic_complexity > 10)
        long_functions = sum(1 for f in functions if f.lines_of_code > 50)
        deeply_nested_functions = sum(1 for f in functions if f.nesting_depth > 3)
        
        # Calculate anomaly score
        anomaly_score = 0
        anomaly_score += (high_complexity_functions / len(functions)) * 30
        anomaly_score += (long_functions / len(functions)) * 25
        anomaly_score += (deeply_nested_functions / len(functions)) * 20
        
        # File size penalty
        if total_lines > 1000:
            anomaly_score += 25
        elif total_lines > 500:
            anomaly_score += 15
        
        return min(100, anomaly_score)
