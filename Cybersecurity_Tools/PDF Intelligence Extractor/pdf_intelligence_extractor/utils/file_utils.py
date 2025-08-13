"""
File utilities for PDF Intelligence Extractor.
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any, Optional


class FileUtils:
    """
    File handling utilities for PDF Intelligence Extractor.
    """
    
    @staticmethod
    def get_pdf_files(directory: str) -> List[str]:
        """
        Get all PDF files from a directory.
        
        Args:
            directory (str): Directory path
            
        Returns:
            List[str]: List of PDF file paths
        """
        pdf_files = []
        try:
            for file in os.listdir(directory):
                if file.lower().endswith('.pdf'):
                    pdf_files.append(os.path.join(directory, file))
        except Exception as e:
            print(f"Error reading directory {directory}: {str(e)}")
        
        return pdf_files
    
    @staticmethod
    def ensure_directory(path: str) -> None:
        """
        Ensure a directory exists, create if it doesn't.
        
        Args:
            path (str): Directory path
        """
        Path(path).mkdir(parents=True, exist_ok=True)
    
    @staticmethod
    def save_json(data: Dict[str, Any], file_path: str) -> None:
        """
        Save data to JSON file.
        
        Args:
            data (Dict[str, Any]): Data to save
            file_path (str): Output file path
        """
        try:
            FileUtils.ensure_directory(os.path.dirname(file_path))
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            raise Exception(f"Error saving JSON file: {str(e)}")
    
    @staticmethod
    def save_csv(data: List[Dict[str, Any]], file_path: str) -> None:
        """
        Save data to CSV file.
        
        Args:
            data (List[Dict[str, Any]]): Data to save
            file_path (str): Output file path
        """
        try:
            FileUtils.ensure_directory(os.path.dirname(file_path))
            if data:
                with open(file_path, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.DictWriter(f, fieldnames=data[0].keys())
                    writer.writeheader()
                    writer.writerows(data)
        except Exception as e:
            raise Exception(f"Error saving CSV file: {str(e)}")
    
    @staticmethod
    def get_file_info(file_path: str) -> Dict[str, Any]:
        """
        Get basic file information.
        
        Args:
            file_path (str): File path
            
        Returns:
            Dict[str, Any]: File information
        """
        try:
            stat = os.stat(file_path)
            return {
                'name': os.path.basename(file_path),
                'size': stat.st_size,
                'modified': stat.st_mtime,
                'path': file_path
            }
        except Exception as e:
            return {'error': str(e)}
    
    @staticmethod
    def validate_pdf_file(file_path: str) -> bool:
        """
        Validate if file is a valid PDF.
        
        Args:
            file_path (str): File path
            
        Returns:
            bool: True if valid PDF, False otherwise
        """
        try:
            if not os.path.exists(file_path):
                return False
            
            if not file_path.lower().endswith('.pdf'):
                return False
            
            # Check file size
            if os.path.getsize(file_path) == 0:
                return False
            
            return True
        except Exception:
            return False 