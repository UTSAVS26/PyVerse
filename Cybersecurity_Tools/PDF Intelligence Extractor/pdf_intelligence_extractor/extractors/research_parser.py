"""
Research paper parser for PDF Intelligence Extractor.
"""

import re
from typing import Dict, List, Any
from .base_parser import BaseParser


class ResearchParser(BaseParser):
    """
    Parser for extracting structured data from research paper PDFs.
    """
    
    def __init__(self):
        super().__init__()
    
    def get_document_type(self) -> str:
        return "research_paper"
    
    def extract_title(self, text: str) -> str:
        """Extract paper title from text."""
        # Look for title at the beginning of the document
        lines = text.split('\n')
        for line in lines[:10]:  # Check first 10 lines
            line = line.strip()
            if line and len(line) > 10 and len(line) < 200:
                # Skip common headers
                if not any(header in line.lower() for header in ['abstract', 'introduction', 'author', 'university', 'this is a research paper']):
                    return line
        
        return ""
    
    def extract_authors(self, text: str) -> List[str]:
        """Extract author names from text."""
        authors = []
        
        # Look for author patterns
        author_patterns = [
            r'Authors?[:\s]+([^\n]+)',
            r'By[:\s]+([^\n]+)',
            r'([A-Z][a-z]+ [A-Z][a-z]+(?:,?\s+[A-Z][a-z]+ [A-Z][a-z]+)*)'
        ]
        
        for pattern in author_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Split multiple authors
                author_list = re.split(r',\s*|\sand\s+', matches[0])
                authors.extend([author.strip() for author in author_list if author.strip() and len(author.strip()) > 3])
                break
        
        return authors
    
    def extract_affiliations(self, text: str) -> List[str]:
        """Extract institutional affiliations from text."""
        affiliations = []
        
        # Look for affiliation patterns
        affiliation_patterns = [
            r'([A-Z][A-Za-z\s&]+(?:University|College|Institute|School|Department))',
            r'Affiliation[:\s]+([^\n]+)',
            r'([A-Z][A-Za-z\s&]+(?:University|College|Institute|School))'
        ]
        
        for pattern in affiliation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            affiliations.extend([match.strip() for match in matches if match.strip()])
        
        return list(set(affiliations))
    
    def extract_abstract(self, text: str) -> str:
        """Extract abstract from text."""
        abstract_patterns = [
            r'Abstract[:\s]*(.*?)(?=\n\n|\n[A-Z]|Introduction|$)', 
            r'ABSTRACT[:\s]*(.*?)(?=\n\n|\n[A-Z]|Introduction|$)',
            r'Summary[:\s]*(.*?)(?=\n\n|\n[A-Z]|Introduction|$)'
        ]
        
        for pattern in abstract_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                return match.group(1).strip()
        
        return ""
    
    def extract_keywords(self, text: str) -> List[str]:
        """Extract keywords from text."""
        keywords = []
        
        # Look for keywords section
        keyword_patterns = [
            r'Keywords?[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)', 
            r'Key\s+Words[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Index\s+Terms[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in keyword_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                keywords_text = match.group(1)
                # Split by common separators
                keyword_list = re.split(r',\s*|\s*;\s*|\s*•\s*', keywords_text)
                keywords.extend([kw.strip() for kw in keyword_list if kw.strip() and len(kw.strip()) > 2])
                break
        
        return keywords
    
    def extract_references(self, text: str) -> List[str]:
        """Extract references from text."""
        references = []
        
        # Look for references section
        ref_patterns = [
            r'References?[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)', 
            r'Bibliography[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)',
            r'Cited\s+Works[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)'
        ]
        
        for pattern in ref_patterns:
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                ref_text = match.group(1)
                # Split references by common patterns
                ref_list = re.split(r'\n\d+\.\s*|\n\[[^\]]+\]\s*|\n[A-Z][a-z]+,\s+[A-Z]', ref_text)
                references.extend([ref.strip() for ref in ref_list if ref.strip() and len(ref.strip()) > 10])
                break
        
        return references
    
    def extract_doi(self, text: str) -> str:
        """Extract DOI from text."""
        doi_pattern = r'DOI[:\s]*([^\s\n]+)'
        match = re.search(doi_pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        return ""
    
    def extract_publication_info(self, text: str) -> Dict[str, str]:
        """Extract publication information."""
        pub_info = {}
        
        # Look for journal/conference name
        journal_patterns = [
            r'Journal[:\s]+([^\n]+)',
            r'Conference[:\s]+([^\n]+)',
            r'Proceedings[:\s]+([^\n]+)'
        ]
        
        for pattern in journal_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                pub_info['journal'] = match.group(1).strip()
                break
        
        # Look for publication year
        year_pattern = r'(\d{4})'
        years = re.findall(year_pattern, text)
        if years:
            pub_info['year'] = years[0]
        
        # Look for volume/issue
        volume_pattern = r'Vol\.?\s*(\d+)'
        match = re.search(volume_pattern, text, re.IGNORECASE)
        if match:
            pub_info['volume'] = match.group(1)
        
        return pub_info
    
    def parse(self, pdf_path: str) -> Dict[str, Any]:
        """
        Parse research paper PDF and extract structured data.
        
        Args:
            pdf_path (str): Path to the research paper PDF file
            
        Returns:
            Dict[str, Any]: Extracted research paper data
        """
        # Extract text from PDF
        text = self.extract_text_from_pdf(pdf_path)
        
        # Extract metadata
        metadata = self.extract_metadata_from_pdf(pdf_path)
        
        # Extract structured data
        title = self.extract_title(text)
        authors = self.extract_authors(text)
        affiliations = self.extract_affiliations(text)
        abstract = self.extract_abstract(text)
        keywords = self.extract_keywords(text)
        references = self.extract_references(text)
        doi = self.extract_doi(text)
        publication_info = self.extract_publication_info(text)
        
        return {
            "document_type": self.get_document_type(),
            "title": title,
            "authors": authors,
            "affiliations": affiliations,
            "abstract": abstract,
            "keywords": keywords,
            "references": references,
            "doi": doi,
            "publication_info": publication_info,
            "metadata": metadata
        } 