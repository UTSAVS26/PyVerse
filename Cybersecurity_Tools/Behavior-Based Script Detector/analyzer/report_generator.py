"""
Report Generator for Behavior-Based Script Detector

This module generates various report formats for analysis results
including JSON, Markdown, and console output.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich import box


class ReportGenerator:
    """Generates reports in various formats for analysis results."""
    
    def __init__(self):
        """Initialize the report generator."""
        self.console = Console()
    
    def generate_json_report(self, filename: str, analysis_result: Dict[str, Any]) -> str:
        """
        Generate JSON report.
        
        Args:
            filename: Name of the analyzed file
            analysis_result: Analysis results dictionary
            
        Returns:
            JSON string representation of the report
        """
        report = {
            'filename': filename,
            'scan_timestamp': datetime.now().isoformat(),
            'risk_score': analysis_result.get('risk_score', 0),
            'verdict': analysis_result.get('verdict', 'Unknown'),
            'total_findings': analysis_result.get('total_findings', 0),
            'severity_breakdown': analysis_result.get('severity_breakdown', {}),
            'suspicious_patterns': analysis_result.get('findings', []),
            'assessment': analysis_result.get('assessment', {})
        }
        
        return json.dumps(report, indent=2)
    
    def generate_markdown_report(self, filename: str, analysis_result: Dict[str, Any]) -> str:
        """
        Generate Markdown report.
        
        Args:
            filename: Name of the analyzed file
            analysis_result: Analysis results dictionary
            
        Returns:
            Markdown string representation of the report
        """
        risk_score = analysis_result.get('risk_score', 0)
        verdict = analysis_result.get('verdict', 'Unknown')
        findings = analysis_result.get('findings', [])
        assessment = analysis_result.get('assessment', {})
        
        # Create markdown content
        md_content = f"""# Behavior Analysis Report

**File:** `{filename}`  
**Scan Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Risk Score:** {risk_score}/100  
**Verdict:** {verdict}

## Summary

{assessment.get('summary', 'No summary available.')}

## Risk Assessment

- **Risk Level:** {verdict}
- **Confidence:** {assessment.get('confidence', 'Unknown')}
- **Total Findings:** {len(findings)}

## Detected Patterns

"""
        
        if findings:
            md_content += "| Line | Pattern | Description | Severity | Score |\n"
            md_content += "|------|---------|-------------|----------|-------|\n"
            
            for finding in findings:
                line = finding.get('line', 'N/A')
                pattern = finding.get('pattern', 'Unknown')
                description = finding.get('description', 'No description')
                severity = finding.get('severity', 'LOW')
                score = finding.get('score', 0)
                
                md_content += f"| {line} | {pattern} | {description} | {severity} | {score} |\n"
        else:
            md_content += "No suspicious patterns detected.\n"
        
        # Add recommendations
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            md_content += "\n## Recommendations\n\n"
            for rec in recommendations:
                md_content += f"- {rec}\n"
        
        return md_content
    
    def generate_console_report(self, filename: str, analysis_result: Dict[str, Any]) -> None:
        """
        Generate and display console report using Rich.
        
        Args:
            filename: Name of the analyzed file
            analysis_result: Analysis results dictionary
        """
        risk_score = analysis_result.get('risk_score', 0)
        verdict = analysis_result.get('verdict', 'Unknown')
        findings = analysis_result.get('findings', [])
        assessment = analysis_result.get('assessment', {})
        
        # Header
        self.console.print(f"\n[bold blue]🛡️ Behavior Analysis Report[/bold blue]")
        self.console.print(f"[bold]File:[/bold] {filename}")
        self.console.print(f"[bold]Scan Date:[/bold] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Risk Score Panel
        risk_color = self._get_risk_color(risk_score)
        risk_panel = Panel(
            f"[bold]{risk_score}/100[/bold]\n{verdict}",
            title="Risk Score",
            border_style=risk_color
        )
        self.console.print(risk_panel)
        
        # Summary
        if assessment.get('summary'):
            summary_panel = Panel(
                assessment['summary'],
                title="Summary",
                border_style="blue"
            )
            self.console.print(summary_panel)
        
        # Findings Table
        if findings:
            self._display_findings_table(findings)
        else:
            self.console.print("[green]✅ No suspicious patterns detected[/green]")
        
        # Recommendations
        recommendations = assessment.get('recommendations', [])
        if recommendations:
            self._display_recommendations(recommendations)
    
    def _get_risk_color(self, score: int) -> str:
        """Get color based on risk score."""
        if score >= 80:
            return "red"
        elif score >= 60:
            return "yellow"
        elif score >= 30:
            return "blue"
        else:
            return "green"
    
    def _display_findings_table(self, findings: List[Dict[str, Any]]) -> None:
        """Display findings in a Rich table."""
        table = Table(title="Detected Suspicious Patterns", box=box.ROUNDED)
        
        table.add_column("Line", style="cyan", no_wrap=True)
        table.add_column("Pattern", style="magenta")
        table.add_column("Description", style="white")
        table.add_column("Severity", style="bold")
        table.add_column("Score", style="yellow", justify="right")
        
        for finding in findings:
            line = finding.get('line', 'N/A')
            pattern = finding.get('pattern', 'Unknown')
            description = finding.get('description', 'No description')
            severity = finding.get('severity', 'LOW')
            score = finding.get('score', 0)
            
            severity_color = self._get_severity_color(severity)
            
            table.add_row(
                str(line),
                pattern,
                description,
                f"[{severity_color}]{severity}[/{severity_color}]",
                str(score)
            )
        
        self.console.print(table)
    
    def _get_severity_color(self, severity: str) -> str:
        """Get color for severity level."""
        severity_colors = {
            'LOW': 'green',
            'MEDIUM': 'yellow',
            'HIGH': 'red',
            'CRITICAL': 'red'
        }
        return severity_colors.get(severity, 'white')
    
    def _display_recommendations(self, recommendations: List[str]) -> None:
        """Display recommendations in a Rich panel."""
        rec_text = "\n".join(recommendations)
        rec_panel = Panel(
            rec_text,
            title="Recommendations",
            border_style="blue"
        )
        self.console.print(rec_panel)
    
    def save_report(self, report_content: str, filename: str, report_type: str, output_dir: str = "reports") -> str:
        """
        Save report to file.
        
        Args:
            report_content: Report content to save
            filename: Original filename (for naming the report)
            report_type: Type of report (json, md, txt)
            output_dir: Directory to save reports
            
        Returns:
            Path to saved report file
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate report filename
        base_name = os.path.splitext(os.path.basename(filename))[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if report_type == "json":
            report_filename = f"{base_name}_analysis_{timestamp}.json"
        elif report_type == "md":
            report_filename = f"{base_name}_analysis_{timestamp}.md"
        else:
            report_filename = f"{base_name}_analysis_{timestamp}.txt"
        
        report_path = os.path.join(output_dir, report_filename)
        
        # Write report to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        return report_path
    
    def generate_batch_report(self, results: List[Dict[str, Any]]) -> str:
        """
        Generate batch report for multiple files.
        
        Args:
            results: List of analysis results
            
        Returns:
            JSON string of batch report
        """
        batch_report = {
            'scan_timestamp': datetime.now().isoformat(),
            'total_files': len(results),
            'files_analyzed': [],
            'summary': {
                'high_risk_files': 0,
                'medium_risk_files': 0,
                'low_risk_files': 0,
                'average_risk_score': 0
            }
        }
        
        total_score = 0
        for result in results:
            filename = result.get('filename', 'Unknown')
            risk_score = result.get('risk_score', 0)
            verdict = result.get('verdict', 'Unknown')
            
            file_result = {
                'filename': filename,
                'risk_score': risk_score,
                'verdict': verdict,
                'total_findings': result.get('total_findings', 0)
            }
            
            batch_report['files_analyzed'].append(file_result)
            total_score += risk_score
            
            # Count by risk level
            if 'High Risk' in verdict or 'Critical Risk' in verdict:
                batch_report['summary']['high_risk_files'] += 1
            elif 'Medium Risk' in verdict:
                batch_report['summary']['medium_risk_files'] += 1
            else:
                batch_report['summary']['low_risk_files'] += 1
        
        # Calculate average score
        if results:
            batch_report['summary']['average_risk_score'] = total_score / len(results)
        
        return json.dumps(batch_report, indent=2) 