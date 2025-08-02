"""
Test cases for ResearchParser.
"""

import pytest
import tempfile
import os
from unittest.mock import patch, MagicMock
from pdf_intelligence_extractor.extractors.research_parser import ResearchParser


class TestResearchParser:
    """Test cases for ResearchParser class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.parser = ResearchParser()
        self.sample_research_text = """
        Machine Learning Applications in Natural Language Processing
        
        Authors: John Smith, Jane Doe
        Affiliations: Stanford University, MIT
        
        Abstract
        This paper presents novel approaches to natural language processing using machine learning techniques.
        We demonstrate significant improvements in text classification and sentiment analysis tasks.
        
        Keywords: machine learning, natural language processing, text classification, sentiment analysis
        
        Introduction
        Natural language processing has seen remarkable progress in recent years...
        
        References
        1. Smith, J. et al. (2023). "Advances in NLP." Journal of AI Research.
        2. Doe, J. et al. (2023). "Deep Learning for Text." Conference on AI.
        
        DOI: 10.1234/example.doi
        """
    
    def test_get_document_type(self):
        """Test document type identification."""
        assert self.parser.get_document_type() == "research_paper"
    
    def test_extract_title(self):
        """Test title extraction from research paper text."""
        title = self.parser.extract_title(self.sample_research_text)
        assert "Machine Learning Applications in Natural Language Processing" in title
    
    def test_extract_title_no_title(self):
        """Test title extraction when no clear title is found."""
        text_without_title = "This is a research paper without a clear title at the beginning."
        title = self.parser.extract_title(text_without_title)
        assert title == ""
    
    def test_extract_authors(self):
        """Test author extraction from research paper text."""
        authors = self.parser.extract_authors(self.sample_research_text)
        assert "John Smith" in authors
        assert "Jane Doe" in authors
    
    def test_extract_authors_no_authors(self):
        """Test author extraction when no authors are found."""
        text_without_authors = "This is a research paper without clear author information."
        authors = self.parser.extract_authors(text_without_authors)
        assert authors == []
    
    def test_extract_affiliations(self):
        """Test affiliation extraction from research paper text."""
        affiliations = self.parser.extract_affiliations(self.sample_research_text)
        assert "Stanford University" in affiliations
        assert "MIT" in affiliations
    
    def test_extract_abstract(self):
        """Test abstract extraction from research paper text."""
        abstract = self.parser.extract_abstract(self.sample_research_text)
        assert "novel approaches to natural language processing" in abstract.lower()
        assert "significant improvements" in abstract.lower()
    
    def test_extract_keywords(self):
        """Test keywords extraction from research paper text."""
        keywords = self.parser.extract_keywords(self.sample_research_text)
        expected_keywords = ["machine learning", "natural language processing", "text classification", "sentiment analysis"]
        for keyword in expected_keywords:
            assert keyword in keywords
    
    def test_extract_references(self):
        """Test references extraction from research paper text."""
        references = self.parser.extract_references(self.sample_research_text)
        assert len(references) > 0
        assert any("Advances in NLP" in ref for ref in references)
        assert any("Deep Learning for Text" in ref for ref in references)
    
    def test_extract_doi(self):
        """Test DOI extraction from research paper text."""
        doi = self.parser.extract_doi(self.sample_research_text)
        assert doi == "10.1234/example.doi"
    
    def test_extract_publication_info(self):
        """Test publication information extraction."""
        pub_info = self.parser.extract_publication_info(self.sample_research_text)
        # Check for year extraction
        assert "2023" in pub_info.get('year', '')
    
    @patch('pdf_intelligence_extractor.extractors.base_parser.PDF_LIBS_AVAILABLE', False)
    def test_parse_research_paper(self):
        """Test complete research paper parsing."""
        
        # Create temporary PDF file
        with tempfile.NamedTemporaryFile(suffix='.txt', delete=False) as tmp_file:
            tmp_file.write(self.sample_research_text.encode('utf-8'))
            tmp_file_path = tmp_file.name
        
        try:
            result = self.parser.parse(tmp_file_path)
            
            # Verify result structure
            assert result["document_type"] == "research_paper"
            assert "Machine Learning Applications" in result["title"]
            assert "John Smith" in result["authors"]
            assert "Jane Doe" in result["authors"]
            assert "Stanford University" in result["affiliations"]
            assert "MIT" in result["affiliations"]
            assert len(result["abstract"]) > 0
            assert len(result["keywords"]) > 0
            assert len(result["references"]) > 0
            assert result["doi"] == "10.1234/example.doi"
            assert "metadata" in result
            
        finally:
            os.unlink(tmp_file_path)
    
    def test_extract_authors_various_formats(self):
        """Test author extraction with various formats."""
        test_cases = [
            ("Authors: John Smith, Jane Doe", ["John Smith", "Jane Doe"]),
            ("By: Alice Johnson and Bob Wilson", ["Alice Johnson", "Bob Wilson"]),
            ("John Smith, Jane Doe, Mike Brown", ["John Smith", "Jane Doe", "Mike Brown"])
        ]
        
        for text, expected in test_cases:
            authors = self.parser.extract_authors(text)
            for author in expected:
                assert author in authors
    
    def test_extract_affiliations_various_formats(self):
        """Test affiliation extraction with various formats."""
        text = """
        Affiliations: Stanford University, MIT, Harvard University
        Department of Computer Science, University of California
        """
        affiliations = self.parser.extract_affiliations(text)
        assert "Stanford University" in affiliations
        assert "MIT" in affiliations
        assert "Harvard University" in affiliations
        assert "University of California" in affiliations
    
    def test_extract_abstract_various_formats(self):
        """Test abstract extraction with various formats."""
        text = """
        ABSTRACT
        This paper presents novel approaches to natural language processing.
        
        Summary
        We demonstrate significant improvements in text classification.
        """
        abstract = self.parser.extract_abstract(text)
        assert "novel approaches" in abstract.lower()
    
    def test_extract_keywords_various_formats(self):
        """Test keywords extraction with various formats."""
        text = """
        Keywords: machine learning, NLP, text analysis
        Key Words: AI, deep learning, neural networks
        Index Terms: computer vision, image processing
        """
        keywords = self.parser.extract_keywords(text)
        expected_keywords = ["machine learning", "NLP", "text analysis", "AI", "deep learning", "neural networks"]
        for keyword in expected_keywords:
            assert keyword in keywords
    
    def test_extract_references_various_formats(self):
        """Test references extraction with various formats."""
        text = """
        References
        1. Smith, J. (2023). "Title." Journal.
        2. Doe, J. (2023). "Another Title." Conference.
        Bibliography
        3. Wilson, M. (2023). "Third Title." Proceedings.
        """
        references = self.parser.extract_references(text)
        assert len(references) >= 2
        assert any("Smith, J." in ref for ref in references)
        assert any("Doe, J." in ref for ref in references)
    
    def test_extract_doi_various_formats(self):
        """Test DOI extraction with various formats."""
        test_cases = [
            ("DOI: 10.1234/example.doi", "10.1234/example.doi"),
            ("doi: 10.5678/another.doi", "10.5678/another.doi"),
            ("DOI 10.9012/third.doi", "10.9012/third.doi")
        ]
        
        for text, expected in test_cases:
            doi = self.parser.extract_doi(text)
            assert doi == expected
    
    def test_extract_publication_info_various_formats(self):
        """Test publication information extraction with various formats."""
        text = """
        Journal: Nature Machine Intelligence
        Conference: International Conference on Machine Learning
        Year: 2023
        Volume: 15
        """
        pub_info = self.parser.extract_publication_info(text)
        assert "Nature Machine Intelligence" in pub_info.get('journal', '')
        assert "2023" in pub_info.get('year', '')
        assert "15" in pub_info.get('volume', '')
    
    def test_parse_empty_text(self):
        """Test parsing with empty text."""
        with patch.object(self.parser, 'extract_text_from_pdf', return_value=""):
            with patch.object(self.parser, 'extract_metadata_from_pdf', return_value={}):
                result = self.parser.parse("dummy_path")
                assert result["document_type"] == "research_paper"
                assert result["title"] == ""
                assert result["authors"] == []
                assert result["affiliations"] == []
                assert result["abstract"] == ""
                assert result["keywords"] == []
                assert result["references"] == []
                assert result["doi"] == ""
                assert result["publication_info"] == {}
    
    def test_extract_title_with_special_characters(self):
        """Test title extraction with special characters."""
        text = """
        Machine Learning & AI: Applications in NLP
        
        Authors: José García, Müller Schmidt
        """
        title = self.parser.extract_title(text)
        assert "Machine Learning & AI" in title
    
    def test_extract_authors_with_special_characters(self):
        """Test author extraction with special characters."""
        text = """
        Authors: José García, Müller Schmidt, O'Connor, Mary-Jane
        """
        authors = self.parser.extract_authors(text)
        assert "José García" in authors
        assert "Müller Schmidt" in authors
        assert "O'Connor" in authors
        assert "Mary-Jane" in authors 