"""
AI helpers for PDF Intelligence Extractor.
"""

import re
from typing import List, Dict, Any, Optional


class AIHelpers:
    """
    AI-assisted extraction helpers.
    """
    
    @staticmethod
    def classify_document_type(text: str) -> str:
        """
        Classify document type based on content analysis.
        
        Args:
            text (str): Document text content
            
        Returns:
            str: Document type (resume, invoice, research_paper, unknown)
        """
        text_lower = text.lower()
        
        # Resume indicators
        resume_keywords = [
            'resume', 'cv', 'curriculum vitae', 'education', 'experience',
            'skills', 'objective', 'summary', 'work history', 'employment'
        ]
        
        # Invoice indicators
        invoice_keywords = [
            'invoice', 'bill', 'payment', 'total', 'amount due', 'subtotal',
            'tax', 'quantity', 'price', 'item', 'line item', 'payment terms'
        ]
        
        # Research paper indicators
        research_keywords = [
            'abstract', 'introduction', 'methodology', 'conclusion', 'references',
            'bibliography', 'doi', 'journal', 'conference', 'proceedings',
            'author', 'affiliation', 'university', 'research'
        ]
        
        # Count keyword matches
        resume_score = sum(1 for keyword in resume_keywords if keyword in text_lower)
        invoice_score = sum(1 for keyword in invoice_keywords if keyword in text_lower)
        research_score = sum(1 for keyword in research_keywords if keyword in text_lower)
        
        # Determine document type based on highest score
        scores = {
            'resume': resume_score,
            'invoice': invoice_score,
            'research_paper': research_score
        }
        
        max_score = max(scores.values())
        if max_score > 0:
            return max(scores, key=scores.get)
        
        return 'unknown'
    
    @staticmethod
    def extract_sections(text: str) -> Dict[str, str]:
        """
        Extract document sections using pattern matching.
        
        Args:
            text (str): Document text content
            
        Returns:
            Dict[str, str]: Extracted sections
        """
        sections = {}
        
        # Common section patterns
        section_patterns = {
            'abstract': r'Abstract[:\s]*(.*?)(?=\n\n|\n[A-Z]|Introduction|$)',
            'introduction': r'Introduction[:\s]*(.*?)(?=\n\n|\n[A-Z]|Methodology|$)',
            'methodology': r'Methodology[:\s]*(.*?)(?=\n\n|\n[A-Z]|Results|$)',
            'results': r'Results[:\s]*(.*?)(?=\n\n|\n[A-Z]|Conclusion|$)',
            'conclusion': r'Conclusion[:\s]*(.*?)(?=\n\n|\n[A-Z]|References|$)',
            'education': r'Education[:\s]*(.*?)(?=\n\n|\n[A-Z]|Experience|$)',
            'experience': r'Experience[:\s]*(.*?)(?=\n\n|\n[A-Z]|Skills|$)',
            'skills': r'Skills[:\s]*(.*?)(?=\n\n|\n[A-Z]|Education|$)',
            'references': r'References?[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)',
            'summary': r'Summary[:\s]*(.*?)(?=\n\n|\n[A-Z]|$)'
        }
        
        for section_name, pattern in section_patterns.items():
            match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
            if match:
                sections[section_name] = match.group(1).strip()
        
        return sections
    
    @staticmethod
    def extract_entities(text: str) -> Dict[str, List[str]]:
        """
        Extract named entities from text using pattern matching.
        
        Args:
            text (str): Document text content
            
        Returns:
            Dict[str, List[str]]: Extracted entities by type
        """
        entities = {
            'organizations': [],
            'dates': [],
            'amounts': [],
            'emails': [],
            'phones': []
        }
        
        # Organization patterns
        org_patterns = [
            r'([A-Z][A-Za-z\s&]+(?:Corp|Inc|Ltd|LLC|Company|Technologies|Solutions))',
            r'([A-Z][A-Za-z\s&]+(?:University|College|Institute|School))'
        ]
        
        for pattern in org_patterns:
            matches = re.findall(pattern, text)
            entities['organizations'].extend([match.strip() for match in matches if match.strip()])
        
        # Date patterns
        date_patterns = [
            r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
            r'(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})',
            r'(\d{4}-\d{2}-\d{2})'
        ]
        
        for pattern in date_patterns:
            matches = re.findall(pattern, text)
            entities['dates'].extend([match.strip() for match in matches if match.strip()])
        
        # Amount patterns
        amount_patterns = [
            r'\$([0-9,]+\.?\d*)',
            r'USD[:\s]+([0-9,]+\.?\d*)',
            r'([0-9,]+\.?\d*)\s*(?:USD|dollars?)'
        ]
        
        for pattern in amount_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            entities['amounts'].extend([match.strip() for match in matches if match.strip()])
        
        # Email patterns
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['emails'] = re.findall(email_pattern, text)
        
        # Phone patterns
        phone_pattern = r'(\+?1?[-.\s]?)?\(?([0-9]{3})\)?[-.\s]?([0-9]{3})[-.\s]?([0-9]{4})'
        phone_matches = re.findall(phone_pattern, text)
        entities['phones'] = [''.join(match) for match in phone_matches if any(match)]
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))
        
        return entities 