"""
File Loader for Behavior-Based Script Detector

This module handles loading and parsing Python files for analysis.
"""

import os
import ast
import asttokens
from typing import Tuple, Optional, List
from pathlib import Path


class FileLoader:
    """Handles loading and parsing Python files for analysis."""
    
    def __init__(self):
        """Initialize the file loader."""
        self.supported_extensions = {'.py'}
    
    def load_file(self, file_path: str) -> Tuple[str, ast.AST, asttokens.ASTTokens]:
        """
        Load and parse a Python file.
        
        Args:
            file_path: Path to the Python file
            
        Returns:
            Tuple of (file_content, ast_tree, ast_tokens)
            
        Raises:
            FileNotFoundError: If file doesn't exist
            SyntaxError: If file has syntax errors
            ValueError: If file is not a Python file
        """
        # Validate file path
    def validate_file(self, file_path: str) -> bool:
        """
        """
        # Prevent directory-traversal in relative paths
        from pathlib import Path
        path_obj = Path(file_path)
        if not path_obj.is_absolute():
            # resolve against current directory without requiring existence
            target = (Path.cwd() / path_obj).resolve()
            if not str(target).startswith(str(Path.cwd())):
                return False

        # existing sanity checks…
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        # …rest of validate_file’s logic…
        
        # Check file extension
        file_ext = Path(file_path).suffix.lower()
        if file_ext not in self.supported_extensions:
            raise ValueError(f"Unsupported file type: {file_ext}. Only .py files are supported.")
        
        # Read file content
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                file_content = f.read()
        
        # Parse AST
        try:
            tree = ast.parse(file_content)
            atok = asttokens.ASTTokens(file_content, tree=tree)
            return file_content, tree, atok
        except SyntaxError as e:
            raise SyntaxError(f"Syntax error in {file_path}: {e}") from e
    
    def load_directory(self, directory_path: str) -> List[Tuple[str, str, ast.AST, asttokens.ASTTokens]]:
        """
        Load all Python files from a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            List of tuples (file_path, file_content, ast_tree, ast_tokens)
        """
        if not os.path.exists(directory_path):
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not os.path.isdir(directory_path):
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        results = []
        
        for root, dirs, files in os.walk(directory_path):
            # Skip common directories that shouldn't be analyzed
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    try:
                        file_content, tree, atok = self.load_file(file_path)
                        results.append((file_path, file_content, tree, atok))
                    except (SyntaxError, ValueError) as e:
                        # Log error but continue with other files
                        print(f"Warning: Could not parse {file_path}: {e}")
        
        return results
    
    def validate_file(self, file_path: str) -> bool:
        """
        Validate if a file can be analyzed.
        
        Args:
            file_path: Path to the file
            
        Returns:
            True if file is valid for analysis
        """
        try:
            # Check if file exists
            if not os.path.exists(file_path):
                return False
            
            # Check if it's a Python file
            file_ext = Path(file_path).suffix.lower()
            if file_ext not in self.supported_extensions:
                return False
            
            # Try to parse the file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            ast.parse(content)
            return True
            
        except (FileNotFoundError, SyntaxError, UnicodeDecodeError):
            return False
    
    def get_file_info(self, file_path: str) -> dict:
        """
        Get basic information about a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Dictionary with file information
        """
        if not os.path.exists(file_path):
            return {'error': 'File not found'}
        
        stat = os.stat(file_path)
        
        return {
            'path': file_path,
            'name': os.path.basename(file_path),
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'is_file': os.path.isfile(file_path),
            'is_directory': os.path.isdir(file_path)
        }
    
    def get_directory_stats(self, directory_path: str) -> dict:
        """
        Get statistics about Python files in a directory.
        
        Args:
            directory_path: Path to the directory
            
        Returns:
            Dictionary with directory statistics
        """
        if not os.path.exists(directory_path):
            return {'error': 'Directory not found'}
        
        if not os.path.isdir(directory_path):
            return {'error': 'Path is not a directory'}
        
        stats = {
            'total_files': 0,
            'python_files': 0,
            'parseable_files': 0,
            'unparseable_files': 0,
            'total_size': 0
        }
        
        for root, dirs, files in os.walk(directory_path):
            # Skip common directories
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['__pycache__', 'node_modules', 'venv', 'env']]
            
            for file in files:
                stats['total_files'] += 1
                file_path = os.path.join(root, file)
                
                if file.endswith('.py'):
                    stats['python_files'] += 1
                    
                    # Get file size
                    try:
                        file_size = os.path.getsize(file_path)
                        stats['total_size'] += file_size
                    except OSError:
                        pass
                    
                    # Check if file is parseable
                    if self.validate_file(file_path):
                        stats['parseable_files'] += 1
                    else:
                        stats['unparseable_files'] += 1
        
        return stats 