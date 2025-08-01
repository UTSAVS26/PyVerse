"""
Invoice parser for PDF Intelligence Extractor.
"""

import re
from datetime import datetime
from typing import Dict, List, Any
from .base_parser import BaseParser


class InvoiceParser(BaseParser):
    """
    Parser for extracting structured data from invoice PDFs.
    """
    
    def __init__(self):
        super().__init__()
    
    def get_document_type(self) -> str:
        return "invoice"
    
    def extract_invoice_number(self, text: str) -> str:
        """Extract invoice number from text."""
        patterns = [
            r'Invoice[:\s#]+([A-Z0-9-]+)',
            r'Invoice\s+Number[:\s]+([A-Z0-9-]+)',
            r'#([A-Z0-9-]+)',
            r'INV[:\s]+([A-Z0-9-]+)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                result = matches[0].strip()
                # Filter out common words that might be captured
                if result.lower() not in ['without', 'number', 'clear']:
                    return result
        
        return ""
    
    def extract_dates(self, text: str) -> Dict[str, str]:
        """Extract invoice and due dates from text."""
        dates = {}
        
        # Date patterns
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        
        # Look for invoice date
        invoice_date_patterns = [
            r'Invoice\s+Date[:\s]+([^\n]+)',
            r'Date[:\s]+([^\n]+)',
            r'Issued[:\s]+([^\n]+)'
        ]
        
        for pattern in invoice_date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                dates['invoice_date'] = matches[0].strip()
                break
        
        # Look for due date
        due_date_patterns = [
            r'Due\s+Date[:\s]+([^\n]+)',
            r'Payment\s+Due[:\s]+([^\n]+)',
            r'Terms[:\s]+([^\n]+)'
        ]
        
        for pattern in due_date_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                dates['due_date'] = matches[0].strip()
                break
        
        return dates
    
    def extract_amounts(self, text: str) -> Dict[str, float]:
        """Extract monetary amounts from text."""
        amounts = {}
        
        # Currency patterns
        currency_patterns = [
            r'\$([0-9,]+\.?\d*)',
            r'USD[:\s]+([0-9,]+\.?\d*)',
            r'Total[:\s]*\$([0-9,]+\.?\d*)',
            r'Amount[:\s]*\$([0-9,]+\.?\d*)'
        ]
        
        # Look for total amount
        total_patterns = [
            r'Total[:\s]*\$([0-9,]+\.?\d*)',
            r'Amount\s+Due[:\s]*\$([0-9,]+\.?\d*)',
            r'Grand\s+Total[:\s]*\$([0-9,]+\.?\d*)'
        ]
        
        for pattern in total_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    amounts['total'] = float(matches[0].replace(',', ''))
                    break
                except ValueError:
                    continue
        
        # If no total found but we have subtotal and tax, calculate total
        if 'total' not in amounts and 'subtotal' in amounts and 'tax' in amounts:
            amounts['total'] = amounts['subtotal'] + amounts['tax']
        
        # Look for subtotal
        subtotal_patterns = [
            r'Subtotal[:\s]*\$([0-9,]+\.?\d*)',
            r'Sub\s+Total[:\s]*\$([0-9,]+\.?\d*)'
        ]
        
        for pattern in subtotal_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    amounts['subtotal'] = float(matches[0].replace(',', ''))
                    break
                except ValueError:
                    continue
        
        # Look for tax amount
        tax_patterns = [
            r'Tax[:\s]*\$([0-9,]+\.?\d*)',
            r'VAT[:\s]*\$([0-9,]+\.?\d*)',
            r'Sales\s+Tax[:\s]*\$([0-9,]+\.?\d*)'
        ]
        
        for pattern in tax_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                try:
                    amounts['tax'] = float(matches[0].replace(',', ''))
                    break
                except ValueError:
                    continue
        
        return amounts
    
    def extract_parties(self, text: str) -> Dict[str, str]:
        """Extract sender and recipient information."""
        parties = {}
        
        # Look for sender/from information
        sender_patterns = [
            r'From[:\s]+([^\n]+)',
            r'Sender[:\s]+([^\n]+)',
            r'Bill\s+From[:\s]+([^\n]+)'
        ]
        
        for pattern in sender_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                parties['sender'] = matches[0].strip()
                break
        
        # Look for recipient/to information
        recipient_patterns = [
            r'To[:\s]+([^\n]+)',
            r'Bill\s+To[:\s]+([^\n]+)',
            r'Recipient[:\s]+([^\n]+)'
        ]
        
        for pattern in recipient_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                parties['recipient'] = matches[0].strip()
                break
        
        return parties
    
    def extract_line_items(self, text: str) -> List[Dict[str, str]]:
        """Extract line items from invoice."""
        line_items = []
        
        # Look for table-like structures or line items
        # This is a simplified version - in practice, you'd need more sophisticated parsing
        lines = text.split('\n')
        
        for line in lines:
            # Look for patterns that might be line items
            if re.search(r'\$\d+\.?\d*', line) and len(line.strip()) > 10:
                # Try to extract item details
                parts = line.split()
                if len(parts) >= 2:
                    item = {
                        'description': ' '.join(parts[:-1]),
                        'amount': parts[-1]
                    }
                    line_items.append(item)
        
        return line_items
    
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse invoice PDF and extract structured data.
        
        Args:
            pdf_path (str): Path to the invoice PDF file
            
        Returns:
            Dict[str, Any]: Extracted invoice data
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Extract metadata
        metadata = self.extract_metadata_from_pdf(pdf_path)
        
        # Extract structured data
        invoice_number = self.extract_invoice_number(text)
        dates = self.extract_dates(text)
        amounts = self.extract_amounts(text)
        parties = self.extract_parties(text)
        line_items = self.extract_line_items(text)
        
        return {
            "document_type": self.get_document_type(),
            "invoice_number": invoice_number,
            "dates": dates,
            "amounts": amounts,
            "parties": parties,
            "line_items": line_items,
            "metadata": metadata
        } 