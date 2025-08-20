"""
Tests for PDF Parser Module
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from flashgenie.pdf_parser.extract_text import PDFTextExtractor


class TestPDFTextExtractor:
    """Test cases for PDFTextExtractor class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.extractor = PDFTextExtractor()
        
        # Sample text for testing
        self.sample_text = """
        This is a sample document for testing.
        It contains multiple sentences and paragraphs.
        
        The mitochondria is the powerhouse of the cell.
        DNA is composed of nucleotides containing bases, phosphate, and a sugar.
        
        Photosynthesis is the process by which plants convert sunlight into energy.
        """
    
    def test_init(self):
        """Test PDFTextExtractor initialization."""
        extractor = PDFTextExtractor(use_pdfplumber=True)
        assert extractor.use_pdfplumber is True
        
        extractor = PDFTextExtractor(use_pdfplumber=False)
        assert extractor.use_pdfplumber is False
    
    @patch('flashgenie.pdf_parser.extract_text.pdfplumber')
    def test_extract_with_pdfplumber(self, mock_pdfplumber):
        """Test text extraction using pdfplumber."""
        # Mock pdfplumber
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        
        mock_page1.extract_text.return_value = "Page 1 content."
        mock_page2.extract_text.return_value = "Page 2 content."
        
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        # Test extraction
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"fake pdf content")
            pdf_path = tmp_file.name
        
        try:
            result = self.extractor._extract_with_pdfplumber(pdf_path)
            expected = "Page 1 content.\nPage 2 content.\n"
            assert result == expected
            
            # Verify pdfplumber was called correctly
            mock_pdfplumber.open.assert_called_once_with(pdf_path)
            
        finally:
            os.unlink(pdf_path)
    
    @patch('flashgenie.pdf_parser.extract_text.extract_text_to_fp')
    def test_extract_with_pdfminer(self, mock_extract):
        """Test text extraction using PDFMiner."""
        # Mock PDFMiner
        mock_extract.return_value = None
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"fake pdf content")
            pdf_path = tmp_file.name
        
        try:
            result = self.extractor._extract_with_pdfminer(pdf_path)
            # PDFMiner extraction returns empty string in our mock
            assert isinstance(result, str)
            
        finally:
            os.unlink(pdf_path)
    
    def test_extract_text_primary_method(self):
        """Test text extraction with primary method."""
        with patch.object(self.extractor, '_extract_with_pdfplumber') as mock_pdfplumber:
            mock_pdfplumber.return_value = "Extracted text"
            
            with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                tmp_file.write(b"fake pdf content")
                pdf_path = tmp_file.name
            
            try:
                result = self.extractor.extract_text(pdf_path)
                assert result == "Extracted text"
                mock_pdfplumber.assert_called_once_with(pdf_path)
                
            finally:
                os.unlink(pdf_path)
    
    def test_extract_text_fallback_method(self):
        """Test text extraction with fallback method."""
        with patch.object(self.extractor, '_extract_with_pdfplumber') as mock_pdfplumber:
            with patch.object(self.extractor, '_extract_with_pdfminer') as mock_pdfminer:
                # Primary method fails
                mock_pdfplumber.side_effect = Exception("Primary method failed")
                mock_pdfminer.return_value = "Fallback text"
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(b"fake pdf content")
                    pdf_path = tmp_file.name
                
                try:
                    result = self.extractor.extract_text(pdf_path)
                    assert result == "Fallback text"
                    mock_pdfplumber.assert_called_once_with(pdf_path)
                    mock_pdfminer.assert_called_once_with(pdf_path)
                    
                finally:
                    os.unlink(pdf_path)
    
    def test_extract_text_both_methods_fail(self):
        """Test text extraction when both methods fail."""
        with patch.object(self.extractor, '_extract_with_pdfplumber') as mock_pdfplumber:
            with patch.object(self.extractor, '_extract_with_pdfminer') as mock_pdfminer:
                # Both methods fail
                mock_pdfplumber.side_effect = Exception("Primary method failed")
                mock_pdfminer.side_effect = Exception("Fallback method failed")
                
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
                    tmp_file.write(b"fake pdf content")
                    pdf_path = tmp_file.name
                
                try:
                    with pytest.raises(Exception) as exc_info:
                        self.extractor.extract_text(pdf_path)
                    
                    assert "Both extraction methods failed" in str(exc_info.value)
                    
                finally:
                    os.unlink(pdf_path)
    
    def test_extract_text_file_not_found(self):
        """Test text extraction with non-existent file."""
        with pytest.raises(Exception):
            self.extractor.extract_text("nonexistent_file.pdf")
    
    @patch('flashgenie.pdf_parser.extract_text.pdfplumber')
    def test_extract_text_with_metadata(self, mock_pdfplumber):
        """Test text extraction with metadata."""
        # Mock pdfplumber
        mock_pdf = MagicMock()
        mock_page = MagicMock()
        
        mock_pdf.metadata = {
            'Title': 'Test Document',
            'Author': 'Test Author',
            'Subject': 'Test Subject',
            'Creator': 'Test Creator',
            'Producer': 'Test Producer'
        }
        mock_page.extract_text.return_value = "Page content."
        
        mock_pdf.pages = [mock_page]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"fake pdf content")
            pdf_path = tmp_file.name
        
        try:
            result = self.extractor.extract_text_with_metadata(pdf_path)
            
            assert result['title'] == 'Test Document'
            assert result['author'] == 'Test Author'
            assert result['subject'] == 'Test Subject'
            assert result['creator'] == 'Test Creator'
            assert result['producer'] == 'Test Producer'
            assert result['page_count'] == 1
            assert 'Page content.' in result['text']
            
        finally:
            os.unlink(pdf_path)
    
    @patch('flashgenie.pdf_parser.extract_text.pdfplumber')
    def test_extract_text_by_pages(self, mock_pdfplumber):
        """Test text extraction by pages."""
        # Mock pdfplumber
        mock_pdf = MagicMock()
        mock_page1 = MagicMock()
        mock_page2 = MagicMock()
        
        mock_page1.extract_text.return_value = "Page 1 content."
        mock_page2.extract_text.return_value = "Page 2 content."
        
        mock_pdf.pages = [mock_page1, mock_page2]
        mock_pdfplumber.open.return_value.__enter__.return_value = mock_pdf
        
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            tmp_file.write(b"fake pdf content")
            pdf_path = tmp_file.name
        
        try:
            result = self.extractor.extract_text_by_pages(pdf_path)
            
            assert len(result) == 2
            assert result[0] == "Page 1 content."
            assert result[1] == "Page 2 content."
            
        finally:
            os.unlink(pdf_path)
    
    def test_extract_text_empty_file(self):
        """Test text extraction with empty file."""
        with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as tmp_file:
            pdf_path = tmp_file.name
        
        try:
            # This should raise an exception for empty file
            with pytest.raises(Exception):
                self.extractor.extract_text(pdf_path)
                
        finally:
            os.unlink(pdf_path)
