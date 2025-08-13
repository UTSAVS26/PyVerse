"""
PDF Intelligence Extractor
An intelligent document parser that extracts structured data from unstructured PDFs.
"""

__version__ = "1.0.0"
__author__ = "Shivansh Katiyar"

from .extractors.base_parser import BaseParser
from .extractors.resume_parser import ResumeParser
from .extractors.invoice_parser import InvoiceParser
from .extractors.research_parser import ResearchParser

__all__ = [
    "BaseParser",
    "ResumeParser", 
    "InvoiceParser",
    "ResearchParser"
] 