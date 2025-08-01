#!/usr/bin/env python3
"""
Demonstration script for Behavior-Based Script Detector

This script demonstrates the capabilities of the behavior-based
script detector by analyzing example files and showing results.
"""

import sys
import os

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from analyzer.pattern_rules import PatternRules
from analyzer.score_calculator import ScoreCalculator
from analyzer.report_generator import ReportGenerator
from utils.file_loader import FileLoader


def demo_analysis():
    """Demonstrate the analysis capabilities."""
    print("🛡️ Behavior-Based Script Detector - Demonstration")
    print("=" * 60)
    
    # Initialize components
    pattern_rules = PatternRules()
    score_calculator = ScoreCalculator()
    report_generator = ReportGenerator()
    file_loader = FileLoader()
    
    # Test files
    test_files = [
        ("examples/test_safe.py", "Safe Test Script"),
        ("examples/test_malicious.py", "Malicious Test Script")
    ]
    
    for file_path, description in test_files:
        print(f"\n📁 Analyzing: {description}")
        print("-" * 40)
        
        try:
            # Load and parse file
            content, tree, atok = file_loader.load_file(file_path)
            
            # Analyze for patterns
            findings = pattern_rules.analyze_ast(tree, atok)
            
            # Calculate risk score
            score_result = score_calculator.calculate_risk_score(findings)
            
            # Display results
            print(f"Risk Score: {score_result['risk_score']}/100")
            print(f"Verdict: {score_result['verdict']}")
            print(f"Total Findings: {score_result['total_findings']}")
            
            if findings:
                print(f"\nDetected Patterns:")
                for finding in findings[:5]:  # Show first 5 findings
                    print(f"  - Line {finding['line']}: {finding['pattern']} ({finding['severity']})")
                if len(findings) > 5:
                    print(f"  ... and {len(findings) - 5} more patterns")
            else:
                print("✅ No suspicious patterns detected")
                
        except Exception as e:
            print(f"❌ Error analyzing {file_path}: {e}")
    
    print("\n" + "=" * 60)
    print("🎉 Demonstration completed!")
    print("\nTo use the detector:")
    print("  python cli/main.py --file <script.py>")
    print("  python cli/main.py --dir <directory> --threshold 60")
    print("  python cli/main.py --file <script.py> --json --markdown")


def demo_patterns():
    """Demonstrate pattern detection capabilities."""
    print("\n🔍 Pattern Detection Capabilities")
    print("=" * 40)
    
    patterns = {
        "exec_usage": "Dynamic code execution (exec, eval)",
        "subprocess_usage": "Shell command execution (subprocess, os.system)",
        "pickle_usage": "Unsafe deserialization (pickle, marshal)",
        "sensitive_file_access": "Access to sensitive system directories",
        "network_download": "Network download operations",
        "socket_usage": "Raw socket operations",
        "encoding_operations": "Encoding/decoding operations",
        "process_creation": "Process or thread creation",
        "suspicious_imports": "Suspicious module imports",
        "obfuscated_code": "Potentially obfuscated code",
        "env_manipulation": "Environment variable manipulation",
        "registry_access": "Windows registry access"
    }
    
    for pattern, description in patterns.items():
        print(f"  • {pattern}: {description}")


def demo_risk_levels():
    """Demonstrate risk level assessment."""
    print("\n⚠️ Risk Level Assessment")
    print("=" * 30)
    
    risk_levels = {
        "Low Risk (0-30)": "Generally safe to execute",
        "Medium Risk (31-60)": "Review before execution",
        "High Risk (61-80)": "Exercise extreme caution",
        "Critical Risk (81-100)": "Do not execute under any circumstances"
    }
    
    for level, description in risk_levels.items():
        print(f"  • {level}: {description}")


if __name__ == "__main__":
    demo_analysis()
    demo_patterns()
    demo_risk_levels() 