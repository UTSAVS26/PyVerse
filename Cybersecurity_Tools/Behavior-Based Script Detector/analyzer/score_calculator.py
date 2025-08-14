"""
Score Calculator for Behavior-Based Script Detector

This module calculates risk scores based on detected suspicious patterns
and provides risk assessment functionality.
"""

from typing import List, Dict, Any, Tuple
from enum import Enum


class RiskLevel(Enum):
    """Enumeration for risk levels."""
    LOW = "Low Risk"
    MEDIUM = "Medium Risk"
    HIGH = "High Risk"
    CRITICAL = "Critical Risk"


class ScoreCalculator:
    """Calculates risk scores based on detected suspicious patterns."""
    
    def __init__(self):
        """Initialize the score calculator with risk thresholds."""
        self.risk_thresholds = {
            RiskLevel.LOW: 30,
            RiskLevel.MEDIUM: 60,
            RiskLevel.HIGH: 80,
            RiskLevel.CRITICAL: 100
        }
        
        # Severity multipliers
        self.severity_multipliers = {
            'LOW': 1.0,
            'MEDIUM': 1.2,
            'HIGH': 1.5,
            'CRITICAL': 2.0
        }
    
    def calculate_risk_score(self, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Calculate overall risk score based on findings.
        
        Args:
            findings: List of detected suspicious patterns
            
        Returns:
            Dictionary containing risk score and assessment
        """
        if not findings:
            return {
                'risk_score': 0,
                'verdict': RiskLevel.LOW.value,
                'total_findings': 0,
                'severity_breakdown': self._get_empty_severity_breakdown()
            }
        
        # Calculate base score
        base_score = sum(finding.get('score', 0) for finding in findings)
        
        # Apply severity multipliers
        adjusted_score = self._apply_severity_multipliers(findings, base_score)
        
        # Cap the score at 100
        final_score = min(adjusted_score, 100)
        
        # Determine risk level
        risk_level = self._determine_risk_level(final_score)
        
        # Get severity breakdown
        severity_breakdown = self._get_severity_breakdown(findings)
        
        return {
            'risk_score': final_score,
            'verdict': risk_level.value,
            'total_findings': len(findings),
            'severity_breakdown': severity_breakdown,
            'findings_by_severity': self._group_findings_by_severity(findings)
        }
    
    def _apply_severity_multipliers(self, findings: List[Dict[str, Any]], base_score: int) -> int:
        """Apply severity multipliers to the base score."""
        adjusted_score = base_score
        
        for finding in findings:
            severity = finding.get('severity', 'LOW')
            multiplier = self.severity_multipliers.get(severity, 1.0)
            
            # Apply multiplier to individual finding score
            finding_score = finding.get('score', 0)
            adjusted_score += (finding_score * (multiplier - 1.0))
        
        return int(adjusted_score)
    
    def _determine_risk_level(self, score: int) -> RiskLevel:
        """Determine risk level based on score."""
        if score >= self.risk_thresholds[RiskLevel.CRITICAL]:
            return RiskLevel.CRITICAL
        elif score >= self.risk_thresholds[RiskLevel.HIGH]:
            return RiskLevel.HIGH
        elif score >= self.risk_thresholds[RiskLevel.MEDIUM]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _get_severity_breakdown(self, findings: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get breakdown of findings by severity level."""
        breakdown = self._get_empty_severity_breakdown()
        
        for finding in findings:
            severity = finding.get('severity', 'LOW')
            if severity in breakdown:
                breakdown[severity] += 1
        
        return breakdown
    
    def _get_empty_severity_breakdown(self) -> Dict[str, int]:
        """Get empty severity breakdown dictionary."""
        return {
            'LOW': 0,
            'MEDIUM': 0,
            'HIGH': 0,
            'CRITICAL': 0
        }
    
    def _group_findings_by_severity(self, findings: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Group findings by severity level."""
        grouped = {
            'LOW': [],
            'MEDIUM': [],
            'HIGH': [],
            'CRITICAL': []
        }
        
        for finding in findings:
            severity = finding.get('severity', 'LOW')
            if severity in grouped:
                grouped[severity].append(finding)
        
        return grouped
    
    def get_risk_assessment(self, score: int, findings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Get comprehensive risk assessment.
        
        Args:
            score: Calculated risk score
            findings: List of detected patterns
            
        Returns:
            Comprehensive risk assessment dictionary
        """
        risk_level = self._determine_risk_level(score)
        
        assessment = {
            'score': score,
            'level': risk_level.value,
            'confidence': self._calculate_confidence(findings),
            'recommendations': self._get_recommendations(risk_level, findings),
            'summary': self._generate_summary(score, findings)
        }
        
        return assessment
    
    def _calculate_confidence(self, findings: List[Dict[str, Any]]) -> str:
        """Calculate confidence level based on findings."""
        if not findings:
            return "No suspicious patterns detected"
        
        high_severity_count = sum(
            1 for finding in findings 
            if finding.get('severity') in ['HIGH', 'CRITICAL']
        )
        
        total_count = len(findings)
        
        if high_severity_count >= total_count * 0.5:
            return "High confidence - Multiple high-severity patterns detected"
        elif high_severity_count > 0:
            return "Medium confidence - Some high-severity patterns detected"
        else:
            return "Low confidence - Only low/medium severity patterns detected"
    
    def _get_recommendations(self, risk_level: RiskLevel, findings: List[Dict[str, Any]]) -> List[str]:
        """Get recommendations based on risk level and findings."""
        recommendations = []
        
        if risk_level == RiskLevel.CRITICAL:
            recommendations.extend([
                "🚨 CRITICAL: Do not execute this script under any circumstances",
                "Review all detected patterns before proceeding",
                "Consider running in isolated environment if absolutely necessary"
            ])
        elif risk_level == RiskLevel.HIGH:
            recommendations.extend([
                "⚠️ HIGH RISK: Exercise extreme caution",
                "Review all detected patterns thoroughly",
                "Consider sandboxed execution environment"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "⚠️ MEDIUM RISK: Review before execution",
                "Check detected patterns for legitimacy",
                "Consider source and context"
            ])
        else:
            recommendations.extend([
                "✅ LOW RISK: Generally safe to execute",
                "Review any detected patterns for context",
                "Standard security practices apply"
            ])
        
        # Add pattern-specific recommendations
        pattern_recommendations = self._get_pattern_specific_recommendations(findings)
        recommendations.extend(pattern_recommendations)
        
        return recommendations
    
    def _get_pattern_specific_recommendations(self, findings: List[Dict[str, Any]]) -> List[str]:
        """Get recommendations specific to detected patterns."""
        recommendations = []
        patterns = set(finding.get('pattern') for finding in findings)
        
        if 'exec_usage' in patterns:
            recommendations.append("🔍 Review exec/eval usage - consider alternatives")
        
        if 'subprocess_usage' in patterns:
            recommendations.append("🔍 Review subprocess calls - ensure commands are safe")
        
        if 'network_download' in patterns:
            recommendations.append("🔍 Verify network downloads - check URLs and sources")
        
        if 'sensitive_file_access' in patterns:
            recommendations.append("🔍 Review file access patterns - ensure paths are legitimate")
        
        if 'pickle_usage' in patterns:
            recommendations.append("🔍 Review pickle usage - consider safer serialization")
        
        return recommendations
    
    def _generate_summary(self, score: int, findings: List[Dict[str, Any]]) -> str:
        """Generate a summary of the risk assessment."""
        if not findings:
            return "No suspicious patterns detected. Script appears safe for execution."
        
        pattern_count = len(findings)
        high_severity_count = sum(
            1 for finding in findings 
            if finding.get('severity') in ['HIGH', 'CRITICAL']
        )
        
        summary = f"Detected {pattern_count} suspicious pattern(s)"
        
        if high_severity_count > 0:
            summary += f" including {high_severity_count} high-severity pattern(s)"
        
        summary += f" with a risk score of {score}/100."
        
        return summary 