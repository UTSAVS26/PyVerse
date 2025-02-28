# Time Complexity Analyzer

## Overview
The Time Complexity Analyzer is a Python script designed to analyze the time complexity of user-provided Python code. By parsing the code and evaluating its structure, the program estimates the time complexity and provides a corresponding order of growth. This tool is particularly useful for developers and students looking to understand the efficiency of their algorithms.
## What Have I Done
In this project, I developed a program that leverages Python's Abstract Syntax Tree (AST) module to parse code input from the user. The program identifies loops and function definitions to estimate the time complexity based on common patterns. It provides clear feedback, including error handling for syntax errors, enhancing the user experience.

## What the Program Does
- Accepts a piece of Python code as input from the user.
- Parses the code using the AST module.
- Analyzes the structure of the code to identify loops and function calls.
- Estimates the time complexity and provides an order of growth (e.g., O(1), O(n), O(n^2)).
- Outputs detailed error messages in case of syntax issues.

## Libraries Used
- **ast**: A built-in Python library for parsing Python source code into its Abstract Syntax Tree representation.

## Conclusion
The Time Complexity Analyzer provides a straightforward and user-friendly way to estimate the efficiency of Python code. With its ability to handle various types of growth patterns and robust error handling, it serves as a valuable tool for anyone looking to improve their understanding of algorithmic efficiency. Future enhancements could include support for more complex constructs and deeper semantic analysis of code.