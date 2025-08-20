"""
PDF Text Extraction Module

Extracts text from PDF files using pdfplumber and PDFMiner as fallback.
"""

import io
from typing import Optional, List, Dict, Any
import logging
from pdfminer.high_level import extract_text_to_fp
from pdfminer.layout import LAParams
import pdfplumber

logger = logging.getLogger(__name__)


class PDFTextExtractor:
    """Extracts text from PDF files with multiple extraction methods."""
    
    def __init__(self, use_pdfplumber: bool = True):
        """
        Initialize the PDF text extractor.
        
        Args:
            use_pdfplumber: Whether to use pdfplumber as primary method
        """
        self.use_pdfplumber = use_pdfplumber
        
    def extract_text(self, pdf_path: str) -> str:
        """
        Extract text from a PDF file.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Extracted text as string
            
        Raises:
            FileNotFoundError: If PDF file doesn't exist
            Exception: If text extraction fails
        """
        try:
            if self.use_pdfplumber:
                return self._extract_with_pdfplumber(pdf_path)
            else:
                return self._extract_with_pdfminer(pdf_path)
        except Exception as e:
            logger.warning(f"Primary extraction method failed: {e}")
            # Try fallback method
            try:
                if self.use_pdfplumber:
                    return self._extract_with_pdfminer(pdf_path)
                else:
                    return self._extract_with_pdfplumber(pdf_path)
            except Exception as fallback_error:
                raise Exception(f"Both extraction methods failed: {fallback_error}")
    
    def _extract_with_pdfplumber(self, pdf_path: str) -> str:
        """Extract text using pdfplumber."""
        text = ""
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
                    
        return text
    
    def _extract_with_pdfminer(self, pdf_path: str) -> str:
        """Extract text using PDFMiner."""
        output = io.StringIO()
        
        with open(pdf_path, 'rb') as pdf_file:
            extract_text_to_fp(pdf_file, output, laparams=LAParams())
            
        return output.getvalue()
    
    def extract_text_with_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """
        Extract text with additional metadata.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing text and metadata
        """
        with pdfplumber.open(pdf_path) as pdf:
            metadata = {
                'title': pdf.metadata.get('Title', ''),
                'author': pdf.metadata.get('Author', ''),
                'subject': pdf.metadata.get('Subject', ''),
                'creator': pdf.metadata.get('Creator', ''),
                'producer': pdf.metadata.get('Producer', ''),
                'page_count': len(pdf.pages),
                'text': ""
            }
            
            # Extract text page by page
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    metadata['text'] += page_text + "\n"
                    
        return metadata
    
    def extract_text_by_pages(self, pdf_path: str) -> List[str]:
        """
        Extract text page by page.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of text strings, one per page
        """
        pages = []
        
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                pages.append(page_text if page_text else "")
                
        return pages
