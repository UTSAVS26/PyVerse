"""
Code Analyzer Module

Main analysis engine that orchestrates AST parsing, metric calculation,
and AI-powered suggestion generation.
"""

import ast
import os
import glob
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import re

from .metrics import ComplexityMetrics, FunctionMetrics, FileMetrics


@dataclass
class AnalysisResult:
    """Container for complete analysis results."""
    files: List[FileMetrics]
    project_metrics: Dict
    suggestions: List[str]
    ai_insights: List[str]
    risk_hotspots: List[Dict]


class CodeAnalyzer:
    """AI-enhanced code analyzer with intelligent pattern recognition."""
    
    def __init__(self):
        self.metrics_calculator = ComplexityMetrics()
        self.suggestion_patterns = self._load_suggestion_patterns()
        self.ai_insights_engine = AIInsightsEngine()
        
    def _load_suggestion_patterns(self) -> Dict:
        """Load AI-trained suggestion patterns."""
        return {
            'high_complexity': {
                'threshold': 15,
                'suggestions': [
                    "Consider breaking this function into smaller, more focused functions",
                    "Extract complex logic into separate helper methods",
                    "Use early returns to reduce nesting and complexity",
                    "Consider using strategy pattern for complex conditional logic"
                ]
            },
            'deep_nesting': {
                'threshold': 4,
                'suggestions': [
                    "Use early returns to reduce nesting depth",
                    "Extract deeply nested logic into separate functions",
                    "Consider using guard clauses to handle edge cases early",
                    "Refactor complex conditions into boolean helper methods"
                ]
            },
            'long_function': {
                'threshold': 50,
                'suggestions': [
                    "This function is doing too much - consider splitting it",
                    "Extract related functionality into separate methods",
                    "Use composition to break down complex operations",
                    "Consider if this function violates the Single Responsibility Principle"
                ]
            },
            'many_parameters': {
                'threshold': 7,
                'suggestions': [
                    "Consider using a data class or configuration object",
                    "Group related parameters into a single object",
                    "Use builder pattern for complex parameter combinations",
                    "Consider if some parameters can be derived from others"
                ]
            }
        }
    
    def analyze_project(self, project_path: str, train_ml: bool = True) -> AnalysisResult:
        """Analyze an entire Python project with AI enhancements."""
        project_path = Path(project_path)
        
        if project_path.is_file():
            files = [project_path]
        else:
            files = self._find_python_files(project_path)
        
        file_metrics = []
        all_function_metrics = []
        
        for file_path in files:
            try:
                file_result = self.analyze_file(str(file_path))
                file_metrics.append(file_result)
                all_function_metrics.extend(file_result.functions)
            except Exception as e:
                print(f"Warning: Could not analyze {file_path}: {e}")
        
        # Train ML model if requested and we have data
        if train_ml and all_function_metrics:
            self._train_ml_models(all_function_metrics)
        
        # Generate project-level insights
        project_metrics = self._calculate_project_metrics(file_metrics)
        suggestions = self._generate_suggestions(file_metrics)
        ai_insights = self.ai_insights_engine.generate_insights(file_metrics, project_metrics)
        risk_hotspots = self._identify_risk_hotspots(file_metrics)
        
        return AnalysisResult(
            files=file_metrics,
            project_metrics=project_metrics,
            suggestions=suggestions,
            ai_insights=ai_insights,
            risk_hotspots=risk_hotspots
        )
    
    def analyze_file(self, file_path: str) -> FileMetrics:
        """Analyze a single Python file."""
        with open(file_path, 'r', encoding='utf-8') as f:
            source_code = f.read()
            source_lines = source_code.splitlines()
        
        try:
            tree = ast.parse(source_code)
        except SyntaxError as e:
            raise ValueError(f"Syntax error in {file_path}: {e}") from e
        return self.metrics_calculator.analyze_file(tree, file_path, source_lines)
    
    def _find_python_files(self, project_path: Path) -> List[Path]:
        """Find all Python files in the project."""
        python_files = []
        
        for pattern in ['**/*.py', '**/*.pyw']:
            python_files.extend(project_path.glob(pattern))
        
        # Filter out common directories to ignore
        ignored_dirs = {'.git', '__pycache__', '.pytest_cache', 'venv', 'env', 'node_modules'}
        filtered_files = []
        
        for file_path in python_files:
            if not any(ignored_dir in file_path.parts for ignored_dir in ignored_dirs):
                filtered_files.append(file_path)
        
        return filtered_files
    
    def _train_ml_models(self, function_metrics: List[FunctionMetrics]) -> None:
        """Train ML models on collected function metrics."""
        # Prepare training data
        training_data = []
        for metrics in function_metrics:
            training_data.append({
                'cyclomatic_complexity': metrics.cyclomatic_complexity,
                'lines_of_code': metrics.lines_of_code,
                'nesting_depth': metrics.nesting_depth,
                'parameters': metrics.parameters,
                'comments_ratio': metrics.comments_ratio
            })
        
        # Train the anomaly detection model
        self.metrics_calculator.train_anomaly_detector(training_data)
        
        # Train the AI insights engine
        self.ai_insights_engine.train_models(function_metrics)
    
    def _calculate_project_metrics(self, file_metrics: List[FileMetrics]) -> Dict:
        """Calculate project-level metrics."""
        if not file_metrics:
            return {}
        
        total_files = len(file_metrics)
        total_lines = sum(f.total_lines for f in file_metrics)
        total_functions = sum(f.total_functions for f in file_metrics)
        
        # Calculate averages
        avg_complexity = np.mean([f.average_complexity for f in file_metrics if f.total_functions > 0])
        avg_maintainability = np.mean([f.maintainability_index for f in file_metrics])
        avg_ai_anomaly = np.mean([f.ai_anomaly_score for f in file_metrics])
        
        # Calculate distributions
        complexity_distribution = self._calculate_complexity_distribution(file_metrics)
        
        return {
            'total_files': total_files,
            'total_lines': total_lines,
            'total_functions': total_functions,
            'average_complexity': avg_complexity,
            'average_maintainability': avg_maintainability,
            'average_ai_anomaly_score': avg_ai_anomaly,
            'complexity_distribution': complexity_distribution,
            'risk_level': self._calculate_project_risk_level(avg_complexity, avg_maintainability, avg_ai_anomaly)
        }
    
    def _calculate_complexity_distribution(self, file_metrics: List[FileMetrics]) -> Dict:
        """Calculate distribution of complexity levels across the project."""
        all_complexities = []
        for file_metric in file_metrics:
            for func_metric in file_metric.functions:
                all_complexities.append(func_metric.cyclomatic_complexity)
        
        if not all_complexities:
            return {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        
        distribution = {
            'low': sum(1 for c in all_complexities if c <= 5),
            'medium': sum(1 for c in all_complexities if 5 < c <= 10),
            'high': sum(1 for c in all_complexities if 10 < c <= 15),
            'critical': sum(1 for c in all_complexities if c > 15)
        }
        
        total = len(all_complexities)
        return {k: (v / total) * 100 for k, v in distribution.items()}
    
    def _calculate_project_risk_level(self, avg_complexity: float, avg_maintainability: float, avg_ai_anomaly: float) -> str:
        """Calculate overall project risk level using AI-enhanced scoring."""
        risk_score = 0
        
        # Complexity risk
        if avg_complexity > 15:
            risk_score += 40
        elif avg_complexity > 10:
            risk_score += 25
        elif avg_complexity > 5:
            risk_score += 10
        
        # Maintainability risk
        if avg_maintainability < 50:
            risk_score += 35
        elif avg_maintainability < 70:
            risk_score += 20
        
        # AI anomaly risk
        if avg_ai_anomaly > 70:
            risk_score += 25
        elif avg_ai_anomaly > 50:
            risk_score += 15
        
        if risk_score >= 80:
            return 'CRITICAL'
        elif risk_score >= 60:
            return 'HIGH'
        elif risk_score >= 40:
            return 'MEDIUM'
        else:
            return 'LOW'
    
    def _generate_suggestions(self, file_metrics: List[FileMetrics]) -> List[str]:
        """Generate AI-powered suggestions for code improvement."""
        suggestions = []
        
        for file_metric in file_metrics:
            for func_metric in file_metric.functions:
                # High complexity suggestions
                if func_metric.cyclomatic_complexity > self.suggestion_patterns['high_complexity']['threshold']:
                    suggestions.append(
                        f"Function '{func_metric.name}' in {file_metric.filename}: "
                        f"{self.suggestion_patterns['high_complexity']['suggestions'][0]}"
                    )
                
                # Deep nesting suggestions
                if func_metric.nesting_depth > self.suggestion_patterns['deep_nesting']['threshold']:
                    suggestions.append(
                        f"Function '{func_metric.name}' in {file_metric.filename}: "
                        f"{self.suggestion_patterns['deep_nesting']['suggestions'][0]}"
                    )
                
                # Long function suggestions
                if func_metric.lines_of_code > self.suggestion_patterns['long_function']['threshold']:
                    suggestions.append(
                        f"Function '{func_metric.name}' in {file_metric.filename}: "
                        f"{self.suggestion_patterns['long_function']['suggestions'][0]}"
                    )
                
                # Many parameters suggestions
                if func_metric.parameters > self.suggestion_patterns['many_parameters']['threshold']:
                    suggestions.append(
                        f"Function '{func_metric.name}' in {file_metric.filename}: "
                        f"{self.suggestion_patterns['many_parameters']['suggestions'][0]}"
                    )
        
        # File-level suggestions
        for file_metric in file_metrics:
            if file_metric.total_lines > 500:
                suggestions.append(
                    f"File {file_metric.filename} is quite large ({file_metric.total_lines} lines). "
                    "Consider breaking it into smaller, more focused modules."
                )
            
            if file_metric.ai_anomaly_score > 70:
                suggestions.append(
                    f"File {file_metric.filename} shows high AI anomaly score. "
                    "This may indicate code quality issues that need attention."
                )
        
        return list(set(suggestions))  # Remove duplicates
    
    def _identify_risk_hotspots(self, file_metrics: List[FileMetrics]) -> List[Dict]:
        """Identify high-risk areas in the codebase using AI analysis."""
        hotspots = []
        
        for file_metric in file_metrics:
            for func_metric in file_metric.functions:
                risk_score = func_metric.ai_risk_score
                
                if risk_score > 70:
                    hotspots.append({
                        'file': file_metric.filename,
                        'function': func_metric.name,
                        'risk_score': risk_score,
                        'complexity': func_metric.cyclomatic_complexity,
                        'lines': func_metric.lines_of_code,
                        'nesting': func_metric.nesting_depth,
                        'risk_type': self._classify_risk_type(func_metric)
                    })
        
        # Sort by risk score (highest first)
        hotspots.sort(key=lambda x: x['risk_score'], reverse=True)
        return hotspots[:10]  # Return top 10 hotspots
    
    def _classify_risk_type(self, func_metric: FunctionMetrics) -> str:
        """Classify the type of risk for a function."""
        if func_metric.cyclomatic_complexity > 15:
            return 'HIGH_COMPLEXITY'
        elif func_metric.lines_of_code > 100:
            return 'LONG_FUNCTION'
        elif func_metric.nesting_depth > 5:
            return 'DEEP_NESTING'
        elif func_metric.parameters > 7:
            return 'MANY_PARAMETERS'
        else:
            return 'MULTIPLE_ISSUES'


class AIInsightsEngine:
    """AI-powered insights generation engine."""
    
    def __init__(self):
        self.complexity_clusters = None
        self.pattern_detector = None
        self.is_trained = False
    
    def train_models(self, function_metrics: List[FunctionMetrics]) -> None:
        """Train AI models for pattern detection and clustering."""
        if len(function_metrics) < 5:
            return
        
        # Prepare features for clustering
        features = []
        for metrics in function_metrics:
            feature_vector = [
                metrics.cyclomatic_complexity,
                metrics.lines_of_code,
                metrics.nesting_depth,
                metrics.parameters,
                metrics.comments_ratio
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        
        # Train complexity clustering model
        n_clusters = min(5, len(features) // 2)
        self.complexity_clusters = KMeans(n_clusters=n_clusters, random_state=42)
        self.complexity_clusters.fit(features_array)
        
        # Train pattern detection model
        self.pattern_detector = TfidfVectorizer(max_features=100, stop_words='english')
        
        # Extract function names and create patterns
        function_names = [metrics.name for metrics in function_metrics]
        if function_names:
            self.pattern_detector.fit(function_names)
        
        self.is_trained = True
    
    def generate_insights(self, file_metrics: List[FileMetrics], project_metrics: Dict) -> List[str]:
        """Generate AI-powered insights about the codebase."""
        insights = []
        
        if not self.is_trained:
            return self._generate_basic_insights(file_metrics, project_metrics)
        
        # Complexity pattern insights
        complexity_insights = self._analyze_complexity_patterns(file_metrics)
        insights.extend(complexity_insights)
        
        # Code organization insights
        organization_insights = self._analyze_code_organization(file_metrics)
        insights.extend(organization_insights)
        
        # Quality trend insights
        quality_insights = self._analyze_quality_trends(file_metrics, project_metrics)
        insights.extend(quality_insights)
        
        return insights
    
    def _generate_basic_insights(self, file_metrics: List[FileMetrics], project_metrics: Dict) -> List[str]:
        """Generate basic insights when ML models are not trained."""
        insights = []
        
        # Overall project health
        if project_metrics.get('average_maintainability', 100) < 70:
            insights.append("⚠️  Overall maintainability is below recommended levels. Consider refactoring complex functions.")
        
        if project_metrics.get('average_complexity', 0) > 10:
            insights.append("🔍 Average cyclomatic complexity is high. This may indicate overly complex logic.")
        
        # File size distribution
        large_files = [f for f in file_metrics if f.total_lines > 500]
        if large_files:
            insights.append(f"📁 Found {len(large_files)} large files (>500 lines). Consider modularization.")
        
        return insights
    
    def _analyze_complexity_patterns(self, file_metrics: List[FileMetrics]) -> List[str]:
        """Analyze complexity patterns using AI clustering."""
        insights = []
        
        if not self.complexity_clusters:
            return insights
        
        # Extract all function features
        all_functions = []
        for file_metric in file_metrics:
            all_functions.extend(file_metric.functions)
        
        if len(all_functions) < 5:
            return insights
        
        features = []
        for func in all_functions:
            feature_vector = [
                func.cyclomatic_complexity,
                func.lines_of_code,
                func.nesting_depth,
                func.parameters,
                func.comments_ratio
            ]
            features.append(feature_vector)
        
        features_array = np.array(features)
        clusters = self.complexity_clusters.predict(features_array)
        
        # Analyze cluster characteristics
        cluster_analysis = {}
        for i, cluster_id in enumerate(clusters):
            if cluster_id not in cluster_analysis:
                cluster_analysis[cluster_id] = []
            cluster_analysis[cluster_id].append(all_functions[i])
        
        # Generate insights based on clusters
        for cluster_id, functions in cluster_analysis.items():
            avg_complexity = np.mean([f.cyclomatic_complexity for f in functions])
            avg_length = np.mean([f.lines_of_code for f in functions])
            
            if avg_complexity > 12 and len(functions) > 3:
                insights.append(
                    f"🎯 AI detected a cluster of {len(functions)} functions with high complexity "
                    f"(avg: {avg_complexity:.1f}). These may benefit from refactoring."
                )
            
            if avg_length > 80 and len(functions) > 2:
                insights.append(
                    f"📏 AI identified {len(functions)} functions with high line count "
                    f"(avg: {avg_length:.1f} lines). Consider extracting helper methods."
                )
        
        return insights
    
    def _analyze_code_organization(self, file_metrics: List[FileMetrics]) -> List[str]:
        """Analyze code organization patterns."""
        insights = []
        
        # Analyze function distribution
        function_counts = [f.total_functions for f in file_metrics if f.total_functions > 0]
        if function_counts:
            avg_functions_per_file = np.mean(function_counts)
            if avg_functions_per_file > 20:
                insights.append("📊 Files contain many functions on average. Consider grouping related functions into classes.")
            elif avg_functions_per_file < 3:
                insights.append("📊 Files contain few functions. This may indicate good separation of concerns.")
        
        # Analyze file size distribution
        file_sizes = [f.total_lines for f in file_metrics]
        if file_sizes:
            size_variance = np.var(file_sizes)
            if size_variance > 10000:  # High variance in file sizes
                insights.append("📈 High variance in file sizes detected. Consider standardizing module sizes.")
        
        return insights
    
    def _analyze_quality_trends(self, file_metrics: List[FileMetrics], project_metrics: Dict) -> List[str]:
        """Analyze quality trends across the project."""
        insights = []
        
        # Analyze maintainability distribution
        maintainability_scores = [f.maintainability_index for f in file_metrics]
        if maintainability_scores:
            low_maintainability = sum(1 for score in maintainability_scores if score < 50)
            if low_maintainability > len(maintainability_scores) * 0.3:
                insights.append("⚠️  More than 30% of files have low maintainability scores. Focus on refactoring these files first.")
        
        # Analyze AI anomaly patterns
        anomaly_scores = [f.ai_anomaly_score for f in file_metrics]
        if anomaly_scores:
            high_anomaly = sum(1 for score in anomaly_scores if score > 70)
            if high_anomaly > 0:
                insights.append(f"🤖 AI detected {high_anomaly} files with unusual complexity patterns. These may need special attention.")
        
        return insights
