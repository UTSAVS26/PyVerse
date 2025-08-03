"""
Test cases for InvoiceParser.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from pdf_intelligence_extractor.extractors.invoice_parser import InvoiceParser


class TestInvoiceParser:
    """Test cases for InvoiceParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = InvoiceParser()
        self.sample_invoice_text = """
        INVOICE #INV-2024-001
        
        From: ABC Company Inc.
        To: XYZ Corporation
        
        Invoice Date: 2024-01-15
        Due Date: 2024-02-15
        
        Item Description          Quantity    Price    Amount
        Web Development          10 hours    $100.00  $1,000.00
        Database Design          5 hours     $150.00  $750.00
        
        Subtotal: $1,750.00
        Tax: $175.00
        Total: $1,925.00
        """
    
    def test_get_document_type(self):
        """Test document type identification."""
        assert self.parser.get_document_type() == "invoice"
    
    def test_extract_invoice_number(self):
        """Test invoice number extraction."""
        invoice_number = self.parser.extract_invoice_number(self.sample_invoice_text)
        assert invoice_number == "INV-2024-001"
    
    def test_extract_invoice_number_no_number(self):
        """Test invoice number extraction when no number is found."""
        text_without_number = "This is an invoice without a clear invoice number."
        invoice_number = self.parser.extract_invoice_number(text_without_number)
        assert invoice_number == ""
    
    def test_extract_dates(self):
        """Test date extraction."""
        dates = self.parser.extract_dates(self.sample_invoice_text)
        assert dates.get('invoice_date') == "2024-01-15"
        assert dates.get('due_date') == "2024-02-15"
    
    def test_extract_amounts(self):
        """Test monetary amounts extraction."""
        amounts = self.parser.extract_amounts(self.sample_invoice_text)
        assert amounts.get('total') == 1925.0
        assert amounts.get('subtotal') == 1750.0
        assert amounts.get('tax') == 175.0
    
    def test_extract_parties(self):
        """Test sender and recipient extraction."""
        parties = self.parser.extract_parties(self.sample_invoice_text)
        assert parties.get('sender') == "ABC Company Inc."
        assert parties.get('recipient') == "XYZ Corporation"
    
    def test_extract_line_items(self):
        """Test line items extraction."""
        line_items = self.parser.extract_line_items(self.sample_invoice_text)
        assert len(line_items) > 0
        # Check that line items contain dollar amounts
        for item in line_items:
            assert '$' in item.get('amount', '')
    
    @patch('pdf_intelligence_extractor.extractors.base_parser.PDF_LIBS_AVAILABLE', False)
    def test_parse_invoice(self):
        """Test complete invoice parsing."""
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(self.sample_invoice_text.encode('utf-8'))
            tmp_file_path = tmp_file.name
        
        try:
            result = self.parser.parse(tmp_file_path)
            
            # Verify result structure
            assert result["document_type"] == "invoice"
            assert result["invoice_number"] == "INV-2024-001"
            assert result["dates"]["invoice_date"] == "2024-01-15"
            assert result["dates"]["due_date"] == "2024-02-15"
            assert result["amounts"]["total"] == 1925.0
            assert result["parties"]["sender"] == "ABC Company Inc."
            assert result["parties"]["recipient"] == "XYZ Corporation"
            assert len(result["line_items"]) > 0
            assert "metadata" in result
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_extract_invoice_number_various_formats(self):
        """Test invoice number extraction with various formats."""
        test_cases = [
            ("Invoice #12345", "12345"),
            ("Invoice Number: INV-001", "INV-001"),
            ("INV: ABC-2024-001", "ABC-2024-001"),
            ("Invoice 2024-001", "2024-001")
        ]
        
        for text, expected in test_cases:
            invoice_number = self.parser.extract_invoice_number(text)
            assert invoice_number == expected
    
    def test_extract_dates_various_formats(self):
        """Test date extraction with various formats."""
        text = """
        Invoice Date: 01/15/2024
        Due Date: February 15, 2024
        """
        dates = self.parser.extract_dates(text)
        assert "01/15/2024" in dates.get('invoice_date', '')
        assert "February 15, 2024" in dates.get('due_date', '')
    
    def test_extract_amounts_various_formats(self):
        """Test amount extraction with various formats."""
        text = """
        Subtotal: $1,000.00
        Tax: $100.00
        Total Amount Due: $1,100.00
        """
        amounts = self.parser.extract_amounts(text)
        assert amounts.get('subtotal') == 1000.0
        assert amounts.get('tax') == 100.0
        assert amounts.get('total') == 1100.0
    
    def test_extract_parties_various_formats(self):
        """Test party extraction with various formats."""
        text = """
        Bill From: ABC Corp
        Bill To: XYZ Inc
        """
        parties = self.parser.extract_parties(text)
        assert parties.get('sender') == "ABC Corp"
        assert parties.get('recipient') == "XYZ Inc"
    
    def test_extract_line_items_complex(self):
        """Test line items extraction with complex format."""
        text = """
        Item Description          Qty    Unit Price    Amount
        Web Development          10      $100.00       $1,000.00
        Database Design          5       $150.00       $750.00
        Testing                  8       $75.00        $600.00
        """
        line_items = self.parser.extract_line_items(text)
        assert len(line_items) >= 3
        for item in line_items:
            assert '$' in item.get('amount', '')
    
    def test_parse_empty_text(self):
        """Test parsing with empty text."""
        with patch.object(self.parser, 'extract_text_from_pdf', return_value=""):
            with patch.object(self.parser, 'extract_metadata_from_pdf', return_value={}):
                result = self.parser.parse("dummy_path")
                assert result["document_type"] == "invoice"
                assert result["invoice_number"] == ""
                assert result["dates"] == {}
                assert result["amounts"] == {}
                assert result["parties"] == {}
                assert result["line_items"] == []
    
    def test_extract_amounts_with_currency_symbols(self):
        """Test amount extraction with different currency symbols."""
        text = """
        Subtotal: $1,000.00
        Tax: $100.00
        Total: $1,100.00
        """
        amounts = self.parser.extract_amounts(text)
        assert amounts.get('subtotal') == 1000.0
        assert amounts.get('tax') == 100.0
        assert amounts.get('total') == 1100.0
    
    def test_extract_dates_with_different_formats(self):
        """Test date extraction with different date formats."""
        text = """
        Invoice Date: 15-Jan-2024
        Due Date: 15-Feb-2024
        """
        dates = self.parser.extract_dates(text)
        assert "15-Jan-2024" in dates.get('invoice_date', '')
        assert "15-Feb-2024" in dates.get('due_date', '')
    
    def test_extract_parties_with_special_characters(self):
        """Test party extraction with special characters."""
        text = """
        From: José & Associates, Inc.
        To: Müller Technologies GmbH
        """
        parties = self.parser.extract_parties(text)
        assert "José & Associates, Inc." in parties.get('sender', '')
        assert "Müller Technologies GmbH" in parties.get('recipient', '') 